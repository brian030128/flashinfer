/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */
#include <flashinfer/attention/fused_cascade.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>

#include "fused_cascade_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

// Single fused-prefill launch covering all L cascade levels.
//
// Per-level scheduler arrays come in as *concatenated* tensors plus an
// `level_metadata` int64 array; tvm::ffi::Array<TensorView> is not a valid
// FFI type so we pack rather than pass a list of tensors.
//
// Layout of `level_metadata` (length = 4 * num_levels):
//   [4*l + 0] = batch_l         (number of requests at this level)
//   [4*l + 1] = padded_batch_l  (number of CTAs at this level)
//   [4*l + 2] = num_pages_l     (number of paged-KV indices at this level)
//   [4*l + 3] = partition_kv flag (0 or 1). When 1, kernel uses split-K
//               output indexing within partial_o[l]: writes go to
//               partial_o[l] + (o_indptr[req] + kv_tile_idx) * o_stride_n.
//               Caller runs VariableLengthMergeStates afterward to fold
//               the chunks into the first total_qo_rows of partial_o[l].
//
// `partial_o` holds final / pre-merge outputs for ALL levels (one slot
// per level, sized to cover the worst-case split-K row count). The Python
// wrapper is responsible for combining partial_o[0..L-1] into the user's
// `out` tensor afterward via merge_state_in_place.
//
// Per concatenated buffer: level l's slice is a contiguous range whose
// per-level length is determined from `level_metadata` (e.g. for
// `qo_indptr_buf`, level l owns `batch_l + 1` int32s starting after the
// prefix sum of (batch_i + 1) for i<l).
void FusedMultiLevelCascadePagedRun(
    TensorView q, TensorView paged_k_cache, TensorView paged_v_cache,
    TensorView qo_indptr_buf,                 // concat int32 [sum_l (batch_l+1)]
    TensorView paged_kv_indptr_buf,           // concat int32 [sum_l (batch_l+1)]
    TensorView paged_kv_indices_buf,          // concat int32 [sum_l num_pages_l]
    TensorView paged_kv_last_page_len_buf,    // concat int32 [sum_l batch_l]
    TensorView request_indices_buf,           // concat int32 [sum_l padded_batch_l]
    TensorView qo_tile_indices_buf,           // concat int32 [sum_l padded_batch_l]
    TensorView kv_tile_indices_buf,           // concat int32 [sum_l padded_batch_l]
    TensorView o_indptr_buf,                  // concat int32 [sum_l (batch_l+1)]
    TensorView kv_chunk_size_ptr_buf,         // int32 [num_levels]
    TensorView level_id_per_cta,              // int32 [total_ctas]
    TensorView partial_o,                     // [num_levels, max_rows_per_level, num_qo_heads, head_dim]
    TensorView partial_lse,                   // [num_levels, max_rows_per_level, num_qo_heads]
    Array<int64_t> level_metadata,
    int64_t num_levels, int64_t layout, int64_t window_left, double sm_scale,
    int64_t cta_tile_q_runtime, bool enable_pdl) {
  TVM_FFI_ICHECK_GE(num_levels, 2) << "num_levels must be >= 2";
  TVM_FFI_ICHECK_LE(num_levels, MAX_LEVELS)
      << "num_levels exceeds compile-time MAX_LEVELS=" << MAX_LEVELS;
  TVM_FFI_ICHECK_EQ(static_cast<int64_t>(level_metadata.size()), 4 * num_levels);

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }
  const int64_t* kv_cache_strides = paged_k_cache.strides().data();

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      PagedParams, [&] {
        FusedCascadeParamsArray<PagedParams, MAX_LEVELS> wrapped;
        wrapped.num_levels = static_cast<uint32_t>(num_levels);
        wrapped.cta_offset[0] = 0;

        const int64_t partial_o_stride = partial_o.stride(0);
        const int64_t partial_lse_stride = partial_lse.stride(0);
        DTypeO* partial_o_ptr = static_cast<DTypeO*>(partial_o.data_ptr());
        float* partial_lse_ptr = static_cast<float*>(partial_lse.data_ptr());

        IdType* qo_indptr_base = static_cast<IdType*>(qo_indptr_buf.data_ptr());
        IdType* paged_kv_indptr_base =
            static_cast<IdType*>(paged_kv_indptr_buf.data_ptr());
        IdType* paged_kv_indices_base =
            static_cast<IdType*>(paged_kv_indices_buf.data_ptr());
        IdType* paged_kv_last_page_len_base =
            static_cast<IdType*>(paged_kv_last_page_len_buf.data_ptr());
        IdType* request_indices_base =
            static_cast<IdType*>(request_indices_buf.data_ptr());
        IdType* qo_tile_indices_base =
            static_cast<IdType*>(qo_tile_indices_buf.data_ptr());
        IdType* kv_tile_indices_base =
            static_cast<IdType*>(kv_tile_indices_buf.data_ptr());
        IdType* o_indptr_base = static_cast<IdType*>(o_indptr_buf.data_ptr());
        IdType* kv_chunk_size_ptr_base =
            static_cast<IdType*>(kv_chunk_size_ptr_buf.data_ptr());

        int64_t qo_indptr_off = 0;
        int64_t paged_kv_indptr_off = 0;
        int64_t paged_kv_indices_off = 0;
        int64_t paged_kv_last_page_len_off = 0;
        int64_t scheduler_off = 0;  // request_indices / qo_tile_indices / kv_tile_indices
        int64_t o_indptr_off = 0;

        uint32_t total_ctas = 0;
        for (int64_t l = 0; l < num_levels; ++l) {
          int64_t batch_l = level_metadata[4 * l + 0];
          int64_t padded_batch_l = level_metadata[4 * l + 1];
          int64_t num_pages_l = level_metadata[4 * l + 2];
          // partition_kv flag from low bit of level_metadata[4*l+3]. When
          // set, the kernel uses split-K output indexing within
          // partial_o[l] (writes num_chunks rows per query). When unset,
          // each query gets one row in partial_o[l].
          bool level_partition_kv = (level_metadata[4 * l + 3] & 1) != 0;

          PagedParams& params = wrapped.data[l];
          params = PagedParams{};

          params.q = static_cast<DTypeQ*>(q.data_ptr());
          params.paged_kv = paged_kv_t<DTypeKV, IdType>(
              num_kv_heads, page_size, HEAD_DIM_VO, batch_l, kv_layout,
              static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
              static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
              paged_kv_indices_base + paged_kv_indices_off,
              paged_kv_indptr_base + paged_kv_indptr_off,
              paged_kv_last_page_len_base + paged_kv_last_page_len_off);
          params.q_indptr = qo_indptr_base + qo_indptr_off;
          // All levels write into their own slot of partial_o/partial_lse.
          // The Python wrapper consolidates split-K chunks via
          // VariableLengthMergeStates and then folds partial_o[0..L-1]
          // into the user's out tensor with merge_state_in_place.
          params.o = partial_o_ptr + l * partial_o_stride;
          params.lse = partial_lse_ptr + l * partial_lse_stride;
          params.num_qo_heads = num_qo_heads;
          params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
          params.q_stride_n = q.stride(0);
          params.q_stride_h = q.stride(1);
          params.window_left = window_left;
          params.logits_soft_cap = 0.0f;
          params.sm_scale = static_cast<float>(sm_scale);

          params.request_indices = request_indices_base + scheduler_off;
          params.qo_tile_indices = qo_tile_indices_base + scheduler_off;
          params.kv_tile_indices = kv_tile_indices_base + scheduler_off;
          params.merge_indptr = nullptr;
          params.o_indptr = o_indptr_base + o_indptr_off;
          params.kv_chunk_size_ptr = kv_chunk_size_ptr_base + l;
          params.block_valid_mask = nullptr;
          params.max_total_num_rows = 0;
          params.total_num_rows = nullptr;

          params.padded_batch_size = static_cast<uint32_t>(padded_batch_l);
          params.partition_kv = level_partition_kv;

          total_ctas += static_cast<uint32_t>(padded_batch_l);
          wrapped.cta_offset[l + 1] = static_cast<int32_t>(total_ctas);

          qo_indptr_off += batch_l + 1;
          paged_kv_indptr_off += batch_l + 1;
          paged_kv_indices_off += num_pages_l;
          paged_kv_last_page_len_off += batch_l;
          scheduler_off += padded_batch_l;
          o_indptr_off += batch_l + 1;
        }
        for (int64_t l = num_levels; l < MAX_LEVELS; ++l) {
          wrapped.data[l] = PagedParams{};
          wrapped.cta_offset[l + 1] = static_cast<int32_t>(total_ctas);
        }

        cudaError_t status = cudaSuccess;
        DISPATCH_CTA_TILE_Q(cta_tile_q_runtime, CTA_TILE_Q, {
          status = flashinfer::FusedBatchPrefillMultiLevelDispatched<
              MAX_LEVELS, CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant, PagedParams>(
              wrapped, static_cast<const int32_t*>(level_id_per_cta.data_ptr()),
              total_ctas, static_cast<uint32_t>(num_qo_heads),
              static_cast<uint32_t>(num_kv_heads), enable_pdl, stream);
        });

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "FusedMultiLevelCascadePagedRun failed: " << cudaGetErrorString(status);
        return true;
      });
}

// Folds split-K chunks per level + cross-level merge + writes final out/lse,
// all in one kernel launch. Replaces the post-prefill chain of
// merge_states (per-split-level) + out.copy_/lse.copy_ + merge_state_in_place
// (per non-final level). For uniform L=2 with one split level, that chain is
// 3 launches; this collapses to 1.
//
// `qo_len_per_level` and `num_chunks_per_level` are int32 host arrays of length
// num_levels. They're small (<= MAX_LEVELS = 4) and pass through __grid_constant__,
// so we read them once per FFI call rather than touching device memory.
void CascadePersistentMerge(TensorView partial_o, TensorView partial_lse, TensorView out,
                            TensorView lse_out, Array<int64_t> qo_len_per_level,
                            Array<int64_t> num_chunks_per_level, int64_t total_qo_rows) {
  const int64_t num_levels = static_cast<int64_t>(qo_len_per_level.size());
  TVM_FFI_ICHECK_GE(num_levels, 1);
  TVM_FFI_ICHECK_LE(num_levels, MAX_LEVELS);
  TVM_FFI_ICHECK_EQ(static_cast<int64_t>(num_chunks_per_level.size()), num_levels);

  CHECK_DIM(4, partial_o);  // [num_levels, max_rows, num_heads, head_dim]
  CHECK_DIM(3, partial_lse);
  CHECK_DIM(3, out);  // [N, num_heads, head_dim]
  CHECK_DIM(2, lse_out);

  const uint32_t num_heads = static_cast<uint32_t>(out.size(1));
  const uint32_t head_dim = static_cast<uint32_t>(out.size(2));
  TVM_FFI_ICHECK_EQ(partial_o.size(2), num_heads);
  TVM_FFI_ICHECK_EQ(partial_o.size(3), head_dim);

  ffi::CUDADeviceGuard device_guard(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());

  CascadeMergeMeta<MAX_LEVELS> meta{};
  meta.num_levels = static_cast<uint32_t>(num_levels);
  for (int64_t l = 0; l < num_levels; ++l) {
    meta.qo_len_per_level[l] = static_cast<int32_t>(qo_len_per_level[l]);
    meta.num_chunks_per_level[l] = static_cast<int32_t>(num_chunks_per_level[l]);
  }
  for (int64_t l = num_levels; l < MAX_LEVELS; ++l) {
    meta.qo_len_per_level[l] = 0;
    meta.num_chunks_per_level[l] = 1;
  }

  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(out.dtype(), c_type, [&] {
    cudaError_t status = flashinfer::CascadeMerge<MAX_LEVELS, c_type, c_type>(
        static_cast<c_type*>(partial_o.data_ptr()), static_cast<float*>(partial_lse.data_ptr()),
        partial_o.stride(0), partial_lse.stride(0), static_cast<c_type*>(out.data_ptr()),
        static_cast<float*>(lse_out.data_ptr()), meta, static_cast<uint32_t>(total_qo_rows),
        num_heads, head_dim, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "CascadePersistentMerge failed: " << cudaGetErrorString(status);
    return true;
  });
  TVM_FFI_ICHECK(success) << "CascadePersistentMerge failed: unsupported data type.";
}

/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef FLASHINFER_ATTENTION_FUSED_CASCADE_CUH_
#define FLASHINFER_ATTENTION_FUSED_CASCADE_CUH_

#include "../math.cuh"
#include "../utils.cuh"
#include "prefill.cuh"
#include "state.cuh"

namespace flashinfer {

// AI-assisted: fuses L per-level prefill workloads into one kernel launch by
// dispatching each CTA into the existing per-level device body. See
// ../../flashinfer/cascade.py:FusedMultiLevelCascadeAttentionWrapper for the
// motivation. Justified over per-level launches because back-to-back small
// per-level prefills under-fill the GPU; merging the launches lets CTAs from
// different levels coexist on the SMs.

// Holds per-level Params plus the cumulative CTA-offset table. Passed by
// value as __grid_constant__ so the kernel can index `data[level]` cheaply
// instead of dereferencing through global memory.
template <typename Params, uint32_t MAX_LEVELS>
struct FusedCascadeParamsArray {
  Params data[MAX_LEVELS];
  // cta_offset[i] = sum of padded_batch_size for levels [0, i). Size MAX_LEVELS+1.
  // Only entries [0, num_levels] are meaningful; trailing entries are unused.
  int32_t cta_offset[MAX_LEVELS + 1];
  uint32_t num_levels;
};

// Per-CTA dispatch into the unmodified BatchPrefillWithPagedKVCacheDevice body.
// `level_id_per_cta[blockIdx.x]` selects which level this CTA belongs to;
// `bx_local = blockIdx.x - cta_offset[level]` is the per-level CTA index that
// the device body would have seen if launched standalone.
template <uint32_t MAX_LEVELS, typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void FusedBatchPrefillMultiLevelKernel(
    const __grid_constant__ FusedCascadeParamsArray<Params, MAX_LEVELS> wrapped,
    const int32_t* __restrict__ level_id_per_cta) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);

  uint32_t level = static_cast<uint32_t>(level_id_per_cta[blockIdx.x]);
  int32_t bx_local = static_cast<int32_t>(blockIdx.x) - wrapped.cta_offset[level];
  BatchPrefillWithPagedKVCacheDevice<KTraits>(
      wrapped.data[level], smem_storage, threadIdx,
      /*bx=*/static_cast<uint32_t>(bx_local),
      /*kv_head_idx=*/blockIdx.z,
      /*num_kv_heads=*/gridDim.z);
}

// Launches FusedBatchPrefillMultiLevelKernel. Mirrors the smem/grid setup of
// BatchPrefillWithPagedKVCacheDispatched (prefill.cuh:2548) since all levels
// share KTraits — the only differences from that path are (a) one-time grid
// dim equal to the sum of per-level padded_batch_size, and (b) no in-launcher
// merge_states (handled by a separate kernel after this returns).
template <uint32_t MAX_LEVELS, uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t FusedBatchPrefillMultiLevelDispatched(
    const FusedCascadeParamsArray<Params, MAX_LEVELS>& wrapped,
    const int32_t* d_level_id_per_cta, uint32_t total_ctas, uint32_t num_qo_heads,
    uint32_t num_kv_heads, bool enable_pdl, cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (total_ctas == 0) {
    return cudaSuccess;
  }

  dim3 nblks(total_ctas, 1, num_kv_heads);
  dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  const int num_ctas_per_sm =
      max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                              (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
          ? 2
          : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  const uint32_t max_num_mma_kv_smem =
      (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
      ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits =
        KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                     NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                     DTypeQKAccum, typename Params::IdType, AttentionVariant>;
    if constexpr (KTraits::IsInvalid()) {
      std::ostringstream err_msg;
      err_msg << "FlashInfer Fused-Cascade Internal Error: Invalid configuration : NUM_MMA_Q="
              << NUM_MMA_Q << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
              << " NUM_WARPS_KV=" << NUM_WARPS_KV;
      FLASHINFER_ERROR(err_msg.str());
    } else {
      size_t smem_size = sizeof(typename KTraits::SharedStorage);
      auto kernel = FusedBatchPrefillMultiLevelKernel<MAX_LEVELS, KTraits, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }

      void* args[] = {(void*)&wrapped, (void*)&d_level_id_per_cta};
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, wrapped, d_level_id_per_cta));
      } else {
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
    }
  });
  return cudaSuccess;
}

// Per-row, per-head all-in-one fold + merge + write-out for fused cascade.
//
// Replaces the wrapper-side launch chain
//   merge_states (split-K fold, per split level)
//   out.copy_(partial_o[0])
//   lse.copy_(partial_lse[0])
//   merge_state_in_place (cross-level, per non-final level)
// with a single launch. For each (row, head), folds level l's num_chunks_l
// partials in-loop, merges across levels, writes the final state to user
// out/lse. Variable-depth contract: deeper levels cover a contiguous prefix
// of rows; per-level qo_len is in `meta.qo_len_per_level[l]`.
//
// Layout of inputs matches what the fused prefill writes:
//   partial_o:   [num_levels, max_rows_per_level, num_heads, head_dim]
//   partial_lse: [num_levels, max_rows_per_level, num_heads]
// Within level l, row r at chunk c lives at index `r * nc_l + c`.
template <uint32_t MAX_LEVELS>
struct CascadeMergeMeta {
  uint32_t num_levels;
  // qo_len_per_level[l] = number of query rows present at level l. Rows
  // [0, qo_len_per_level[l]) are written by level l; rows beyond that are
  // never read (variable-depth no-op).
  int32_t qo_len_per_level[MAX_LEVELS];
  // num_chunks_per_level[l] = split-K chunks at level l (1 if no split).
  int32_t num_chunks_per_level[MAX_LEVELS];
};

template <uint32_t MAX_LEVELS, uint32_t vec_size, typename DTypeIn, typename DTypeO>
__global__ void CascadeMergeKernel(
    DTypeIn* __restrict__ partial_o, float* __restrict__ partial_lse,
    int64_t partial_o_level_stride, int64_t partial_lse_level_stride,
    DTypeO* __restrict__ out, float* __restrict__ lse_out,
    const __grid_constant__ CascadeMergeMeta<MAX_LEVELS> meta, uint32_t num_heads,
    uint32_t head_dim) {
  uint32_t tx = threadIdx.x;
  uint32_t pos = blockIdx.x;       // row index
  uint32_t head_idx = blockIdx.y;  // head index

  state_t<vec_size> st;
  bool any = false;
#pragma unroll 1
  for (uint32_t l = 0; l < meta.num_levels; ++l) {
    if (pos >= static_cast<uint32_t>(meta.qo_len_per_level[l])) continue;
    uint32_t nc = static_cast<uint32_t>(meta.num_chunks_per_level[l]);
    DTypeIn* v_base = partial_o + l * partial_o_level_stride;
    float* s_base = partial_lse + l * partial_lse_level_stride;
#pragma unroll 1
    for (uint32_t c = 0; c < nc; ++c) {
      uint32_t row_idx = pos * nc + c;
      float s = s_base[row_idx * num_heads + head_idx];
      vec_t<float, vec_size> v;
      v.cast_load(v_base + (row_idx * num_heads + head_idx) * head_dim + tx * vec_size);
      st.merge(v, s, 1);
      any = true;
    }
  }

  if (!any) {
    // Defensive: in our cascade L0 always covers all rows so this branch
    // is dead in practice. Write zeros + -inf if it ever fires.
    vec_t<DTypeO, vec_size> zero;
    zero.fill(DTypeO(0.f));
    zero.store(out + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    if (lse_out != nullptr) {
      lse_out[pos * num_heads + head_idx] = -math::inf;
    }
    return;
  }

  st.normalize();
  st.o.cast_store(out + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (lse_out != nullptr) {
    lse_out[pos * num_heads + head_idx] = st.get_lse();
  }
}

template <uint32_t MAX_LEVELS, typename DTypeIn, typename DTypeO>
cudaError_t CascadeMerge(DTypeIn* partial_o, float* partial_lse, int64_t partial_o_level_stride,
                        int64_t partial_lse_level_stride, DTypeO* out, float* lse_out,
                        const CascadeMergeMeta<MAX_LEVELS>& meta, uint32_t total_qo_rows,
                        uint32_t num_heads, uint32_t head_dim, cudaStream_t stream) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    dim3 nblks(total_qo_rows, num_heads);
    dim3 nthrs(bdx);
    auto kernel = CascadeMergeKernel<MAX_LEVELS, vec_size, DTypeIn, DTypeO>;
    cudaLaunchConfig_t config{};
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, partial_o, partial_lse,
                                            partial_o_level_stride, partial_lse_level_stride, out,
                                            lse_out, meta, num_heads, head_dim));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_FUSED_CASCADE_CUH_

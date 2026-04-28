/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */
#include "fused_cascade_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

void FusedMultiLevelCascadePagedRun(
    TensorView q, TensorView paged_k_cache, TensorView paged_v_cache,
    TensorView qo_indptr_buf, TensorView paged_kv_indptr_buf,
    TensorView paged_kv_indices_buf, TensorView paged_kv_last_page_len_buf,
    TensorView request_indices_buf, TensorView qo_tile_indices_buf,
    TensorView kv_tile_indices_buf, TensorView o_indptr_buf,
    TensorView kv_chunk_size_ptr_buf, TensorView level_id_per_cta,
    TensorView partial_o, TensorView partial_lse,
    Array<int64_t> level_metadata,
    int64_t num_levels, int64_t layout, int64_t window_left, double sm_scale,
    int64_t cta_tile_q_runtime, bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_paged_run, FusedMultiLevelCascadePagedRun);

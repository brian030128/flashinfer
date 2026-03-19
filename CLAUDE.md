# SparseSpec FlashInfer Fork: On-Chip Dispatch Analysis

This is a fork of FlashInfer modified for the SparseSpec paper. All changes are relative to commit `2946e0454b300265bed2e9cb0982c0daf21098d4`.

## Problem

In speculative decoding, the attention operation has **heterogeneous arithmetic intensity** between draft (sparse, few query tokens) and verification (full, many query tokens) phases. A kernel optimized for one degrades on the other — ~85% bandwidth for its target but <50% for the other. Launching two separate kernels incurs launch overhead and underutilizes the GPU.

## Solution: Single Persistent Kernel with On-Chip Dispatch

SparseSpec launches **one cooperative kernel** that contains **two runners** with different tile configurations. The scheduler pre-partitions work items by query length, and each runner processes only its assigned work items.

### Architecture Overview

```
Single cudaLaunchCooperativeKernel
  ├── Runner1 (CTA_TILE_Q=64, verification/prefill)  ── processes work items with packed_qo_len > 16
  ├── Runner2 (CTA_TILE_Q=16, draft/decode)           ── processes work items with packed_qo_len <= 16
  ├── grid.sync()
  └── ReductionRunner (merges partial outputs)
```

### Step 1: CPU-Side Scheduling — `TwoStageHolisticPlan`

**File:** `include/flashinfer/attention/scheduler.cuh:1086-1201`

The planner (`TwoStageHolisticPlan`) creates `HolisticPlanInfo<2>` with two task sets. It partitions batch requests based on packed query length:

```cpp
const uint32_t CTA_TILE_Q_SIZES[NUM_TASKS] = {64, 16};
// ...
if (packed_qo_len > CTA_TILE_Q_SIZES[1]) {   // > 16
    idx_qo_kv_len_vec[0].push_back(...);       // → Task 0 (Runner1, verification)
} else {
    idx_qo_kv_len_vec[1].push_back(...);       // → Task 1 (Runner2, draft/decode)
}
```

Each task gets its own set of work scheduling arrays (`work_indptr`, `q_indptr`, `kv_indptr`, `batch_idx_arr`, etc.) stored at different offsets in workspace buffers. The scheduler also splits KV across SMs using a min-heap for load balancing, and each task can have different `len_kv_chunk`.

### Step 2: Two Param Sets Passed to One Kernel

**File:** `csrc/batch_attention.cu:112-180`

The host code creates `PersistentParams params[2]`, one per task. Both share the same Q/K/V/O data pointers, but each has its own work decomposition arrays (`work_indptr`, `q_indptr`, etc.). The kernel is launched as:

```cpp
BatchPagedAttentionPersistent<64, 16, HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, AttentionVariant>(
    params[0], params[1], plan_info.num_blks_x, plan_info.num_blks_y, stream);
```

### Step 3: Kernel Template — Sequential Two-Runner Execution

**File:** `include/flashinfer/attention/persistent_template.cuh:57-97`

```cpp
template <class BlockPersistentRunner1, class BlockPersistentRunner2, class BlockReductionRunner>
__global__ void PersistentKernelTemplate(
    const __grid_constant__ typename BlockPersistentRunner1::Params params_1,
    const __grid_constant__ typename BlockPersistentRunner2::Params params_2)
```

Each CTA (thread block) runs **both** runners sequentially:

1. **Runner1** executes, iterating over `params_1.work_indptr[blockIdx.y]` work items (verification workload)
2. **Runner2** executes, iterating over `params_2.work_indptr[blockIdx.y]` work items (draft workload)
3. `cg::this_grid().sync()` — full grid barrier
4. **ReductionRunner** merges partial outputs from both runners into final output

Both runners alias the **same shared memory** (reinterpret_cast to their respective SharedStorage types), so the kernel only allocates `max(sizeof(KTraits1::SharedStorage), sizeof(KTraits2::SharedStorage))`.

### Step 4: Different Kernel Traits per Runner

**File:** `include/flashinfer/attention/persistent.cuh:930-978`

#### Runner1 (Verification — large tiles, more MMA ops)
```
CTA_TILE_Q_1 = 64
NUM_WARPS_Q_1 = 1 (when CTA_TILE_Q=64)
NUM_WARPS_KV_1 = 4
NUM_MMA_Q_1 = 4     (= 64 / 16)
NUM_MMA_KV_1 = 8    (when NUM_WARPS_Q == 1, set by profiling)
```
This configuration is compute-bound friendly: large Q tile amortizes KV loading, many KV MMA tiles increase arithmetic intensity. Optimal for verification where many query tokens attend to the full KV cache.

#### Runner2 (Draft — small tiles, fewer MMA ops)
```
CTA_TILE_Q_2 = 16
NUM_WARPS_Q_2 = 1   (derived from get_num_warps_q(16))
NUM_WARPS_KV_2 = 4   (derived from get_num_warps_kv(16))
NUM_MMA_Q_2 = 1      (= 16 / 16)
NUM_MMA_KV_2 = 2     (hardcoded)
```
This configuration is memory-bound friendly: small Q tile avoids wasted computation on sparse queries (few tokens), fewer KV MMA tiles reduce per-iteration cost. Optimal for draft/decode where only 1-2 tokens attend to the KV cache.

### Step 5: Per-Work Dispatch via `work_indptr`

**File:** `include/flashinfer/attention/persistent.cuh:254-256`

Each runner's work loop iterates over its own assigned work items:
```cpp
for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1]; ++work_idx)
```

The `work_indptr` array is indexed by `blockIdx.y` (the SM/cluster index). The scheduler has already partitioned work so that:
- `params_1.work_indptr` contains only verification work items
- `params_2.work_indptr` contains only draft work items

Each CTA processes work assigned to its `blockIdx.y`. A CTA running Runner1 might process 3 verification work items, then switch to Runner2 and process 5 draft work items. The work is load-balanced across SMs by the min-heap scheduler.

### Step 6: `get_block_coord` and Per-Work Variant

**File:** `include/flashinfer/attention/persistent.cuh:32-39, 267-271`

Each work item carries full metadata via `get_block_coord`:
```cpp
auto [batch_idx, q_indptr, kv_indptr, o_indptr, q_len, kv_len,
      packed_qo_start, kv_start, kv_end, kv_head_idx, len_kv_chunk] = get_block_coord(params, work_idx);
```

A per-work `AttentionVariant` is constructed with the work index, providing custom transforms and metadata (including `request_type` used in the score kernel).

### Attention Score Kernel Variant

**File:** `include/flashinfer/attention/persistent.cuh:710-927, 980-1028`

A second kernel type, `BatchAttentionScorePersistent`, uses `PersistentAttentionScoreKernelTemplate` (two runners, no reduction). Its score runner (`BlockBatchAttentionScorePersistent`) has an additional on-chip filter:

```cpp
AttentionVariant variant(params, work_idx, nullptr);
if (variant.request_type != 2 || packed_qo_start != 0) {
    continue;  // only process verification requests here
}
```

This allows Runner1 in the score kernel to skip non-verification work at runtime, providing a second level of filtering beyond the scheduler's static partitioning.

The score kernel also supports two CTA_TILE_Q_1 options (32 or 64) dispatched from the Python side via `cta_tile_q` argument (`csrc/batch_attention_score.cu:149-163`).

### Summary of Key Files

| File | Role |
|------|------|
| `include/flashinfer/attention/persistent_template.cuh` | Kernel templates: `PersistentKernelTemplate` (2 runners + reduction), `PersistentAttentionScoreKernelTemplate` (2 runners, no reduction) |
| `include/flashinfer/attention/persistent.cuh` | Runner implementations: `BlockBatchPagedAttentionPersistent`, `BlockBatchAttentionScorePersistent`, tile trait configuration, `logits_to_scores_and_reduction` |
| `include/flashinfer/attention/scheduler.cuh` | `TwoStageHolisticPlan`: CPU-side scheduler that partitions work by packed_qo_len into two task sets |
| `csrc/batch_attention.cu` | PyTorch binding for `BatchPagedAttentionPersistent` (CTA_TILE_Q: 64/16) |
| `csrc/batch_attention_score.cu` | PyTorch binding for `BatchAttentionScorePersistent` (CTA_TILE_Q: 32 or 64 / 16) |
| `csrc/batch_attention_customize_config.jinja` | Jinja template for `PersistentParams` struct and custom attention variants |
| `csrc/batch_attention_score_customize_config.jinja` | Same for score kernel |
| `flashinfer/attention.py` | Python API with JIT module support for custom attention variants |

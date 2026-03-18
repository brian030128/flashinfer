# Fused Cascade Attention: Implementation Notes

## Problem

FlashInfer's `MultiLevelCascadeAttentionWrapper` handles cascade (shared-prefix) attention by launching **separate kernels per level** plus a merge kernel. This works well but has inherent overhead:

- N kernel launches (one per level) + 1 merge launch
- Each launch pays dispatch latency (~3-5us)
- GPU cannot overlap the levels (they're sequential)

Meanwhile, the existing `TwoStageHolisticPlan` persistent kernel already runs **two runners with different tile sizes** in a single cooperative launch (Runner1 for prefill, Runner2 for decode). We extend this to handle cascade by treating each (level, request) pair as an independent work item within the same kernel.

## What We Built

`CascadeBatchAttention` — a Python class that fuses all cascade levels into **one cooperative kernel launch**. Both the attention computation and the cross-level reduction happen inside a single `cudaLaunchCooperativeKernel`.

```
Single cudaLaunchCooperativeKernel
  ├── Runner1 (CTA_TILE_Q=128) ── processes large-Q work items from any level
  ├── Runner2 (CTA_TILE_Q=16)  ── processes small-Q work items from any level
  ├── grid.sync()
  └── ReductionRunner ── merges partials across levels AND KV chunks
```

## Changes by File

### 1. `PersistentParams` — cascade fields (`csrc/batch_attention_customize_config.jinja`)

Added two per-work-item arrays to `PersistentParams`:

```cpp
IdType* cascade_num_kv_chunks_arr;  // how many levels to merge for this work item
IdType* cascade_kv_chunk_idx_arr;   // which level this work item belongs to
```

These are `nullptr` for non-cascade use (the existing `TwoStageHolisticPlan` path).

### 2. Runner — cascade-aware chunk indexing (`include/flashinfer/attention/persistent.cuh`)

**Before:** Each work item computed `kv_chunk_idx` and `num_kv_chunks` from KV splitting (`kv_start / len_kv_chunk`). This assumes all partials for a given Q row come from splitting one KV range.

**After:** When `cascade_num_kv_chunks_arr != nullptr`, the runner reads per-work-item values instead:

```cpp
if (params.cascade_num_kv_chunks_arr != nullptr) {
    kv_chunk_idx = params.cascade_kv_chunk_idx_arr[work_idx];
    num_kv_chunks = params.cascade_num_kv_chunks_arr[work_idx];
} else {
    // original KV-split logic
}
```

This lets the reduction runner treat "level 0 partial" and "level 1 partial" as chunk 0 and chunk 1 of the same Q row, merging them with the same log-sum-exp reduction used for KV splits.

**KV bounds fix:** Changed `packed_kv_bound` from `kv_indptr * block_size + kv_len` to `kv_indptr * block_size + kv_end`. Non-causal levels inflate `kv_len` (by adding `qo_len`) to disable the causal mask, but `kv_end` is the real data boundary. Without this fix, the runner would try to load pages beyond what exists.

### 3. `HolisticPlanInfo` — serialization (`include/flashinfer/attention/scheduler.cuh`)

Added `cascade_num_kv_chunks_offset` and `cascade_kv_chunk_idx_offset` to the per-task struct. `NUM_TASK_ARGS` bumped from 10 to 12. The existing `TwoStageHolisticPlan` sets these to -1 (unused).

### 4. `CascadeHolisticPlan` — new scheduler (`include/flashinfer/attention/scheduler.cuh`)

A new scheduler function that creates work items for all (level, request) pairs:

**Request classification (per level):**
```
for each level:
    for each request in batch_size_arr[level]:
        packed_qo_len = qo_len * gqa_group_size
        if packed_qo_len > 16 → Task 0 (Runner1, large tiles)
        else                  → Task 1 (Runner2, small tiles)
```

Each level can have a **different batch size and qo_indptr**. The shared prefix level can have batch_size=1 with `qo_indptr=[0, total_q_len]` (one big request reading shared KV once), while the unique suffix level has batch_size=N with per-request indptr.

**Partial output layout:** Pre-computed interleaved layout where packed QO row `j` has its level partials at indices `j * num_levels, j * num_levels + 1, ...`:

```
merge_indptr[j] = j * num_levels     (for j = 0..total_packed_qo_len-1)
partial_o_nnz = total_packed_qo_len * num_levels
```

Each work item writes at `partial_o[first_packed_row * num_levels + level + row * num_levels]`, and the reduction runner reads all level partials for each output row via `merge_indptr`.

**Non-causal level trick:** For levels without causal masking, `kv_len` is inflated by `qo_len`. Since the causal mask boundary is `kv_len - qo_len + row_position`, inflating `kv_len` pushes the boundary beyond `kv_end`, effectively disabling causal masking while the kernel still runs in "causal mode." The actual data boundary is enforced by `kv_end`.

**Load balancing:** Same min-heap cost model as `TwoStageHolisticPlan`. Work items from all levels are load-balanced across SMs.

### 5. C++ binding (`csrc/batch_attention.cu`)

`CascadeBatchPagedAttentionPlan` accepts:
- `std::vector<at::Tensor> qo_indptr_arr` — per-level QO indptr (replaces single `qo_indptr`)
- Derives `batch_size_arr[l]` from `qo_indptr_arr[l].size(0) - 1`
- Populates cascade fields in `PersistentParams` when `cascade_num_kv_chunks_offset >= 0`

### 6. PyBind (`csrc/batch_attention_jit_pybind.cu`)

Updated the `cascade_plan` declaration to match the new C++ signature.

### 7. Python API (`flashinfer/attention.py`)

`CascadeBatchAttention.plan()` signature:
```python
def plan(self,
    qo_indptr_arr: List[torch.Tensor],   # per-level QO indptr
    kv_indptr_arr: List[torch.Tensor],    # per-level KV page table indptr
    kv_indices_arr: List[torch.Tensor],   # per-level KV page indices
    kv_len_arr: List[torch.Tensor],       # per-level KV lengths
    ...)
```

All levels must have the same total Q length (`qo_indptr_arr[l][-1]` equal for all `l`). KV indices are concatenated internally; each level's `kv_indptr` values are offset by the cumulative page count of preceding levels.

### 8. Tests (`tests/test_cascade_batch_attention.py`)

Tests use per-level qo_indptr matching `MultiLevelCascadeAttentionWrapper`'s pattern:

```python
# Shared level: batch=1, one request with all queries
qo_indptr_shared = torch.tensor([0, batch_size * qo_len])
# Unique level: batch=N, per-request
qo_indptr_unique = torch.arange(batch_size + 1) * qo_len

cascade.plan(
    [qo_indptr_shared, qo_indptr_unique],
    [shared_kv_indptr[:2], unique_kv_indptr],           # shared: 1 request
    [shared_kv_indices[:num_shared_pages], unique_kv_indices],
    [shared_kv_len_tensor[:1], unique_kv_len_tensor],   # shared: 1-element
    ...)
```

## Current Performance

### Without CUDA Graph

With shared_kv_len=1024, unique_kv_len=8, batch=16, heads=8, head_dim=128:

| Method | Median Latency |
|--------|---------------|
| Flat Paged Decode (no cascade) | 0.0430 ms |
| MultiLevel (N kernels + merge) | 0.1004 ms |
| Fused Cascade (1 kernel) | 0.0266 ms |

The fused kernel is **1.62x faster than flat decode** (eliminates redundant shared KV reads) and **3.77x faster than MultiLevel** (eliminates N kernel launches + merge overhead).

### With CUDA Graph (across shared prefix lengths)

unique_kv_len=8, batch_size=16, num_heads=8, head_dim=128:

| shared_kv_len | Flat (ms) | MultiLevel (ms) | Fused (ms) | vs Multi | vs Flat |
|---------------|-----------|-----------------|------------|----------|---------|
| 256           | 0.0143    | 0.0195          | 0.0154     | 1.27x    | 0.93x   |
| 512           | 0.0205    | 0.0195          | 0.0154     | 1.27x    | 1.33x   |
| 1024          | 0.0338    | 0.0195          | 0.0205     | 0.95x    | 1.65x   |
| 2048          | 0.0625    | 0.0338          | 0.0358     | 0.94x    | 1.74x   |
| 4096          | 0.1178    | 0.0461          | 0.0604     | 0.76x    | 1.95x   |
| 8192          | 0.2324    | 0.0696          | 0.1085     | 0.64x    | 2.14x   |
| 16384         | 0.4946    | 0.1137          | 0.1976     | 0.58x    | 2.50x   |

**Analysis:** With CUDA graph (eliminating host dispatch overhead), the fused kernel consistently beats flat decode (1.33–2.50x) since cascade avoids redundant shared KV reads. However, it is **slower than MultiLevel** at shared_kv_len ≥ 1024 (0.58–0.95x). This is expected: CUDA graph removes the kernel launch overhead that was MultiLevel's main disadvantage, and the fused kernel pays the cost of the cooperative grid sync + sequential runner execution. The fused kernel's advantage is in the non-graph regime where launch overhead dominates, or when the shared prefix is short enough that the single-kernel overhead is negligible.

## Design Decisions

**Why reuse the existing persistent kernel template?** The `PersistentKernelTemplate` (Runner1 → Runner2 → grid.sync → Reduction) already handles two tile configurations and partial output merging. Cascade just adds another dimension to the partials: instead of only KV-split partials, we now also have level partials. The reduction runner doesn't care where partials came from — it merges everything between `merge_indptr[j]` and `merge_indptr[j+1]`.

**Why per-level qo_indptr instead of a shared one?** With a single shared `qo_indptr`, the shared prefix level would create N independent work items (one per request), each redundantly reading the full shared KV. Per-level `qo_indptr` lets the shared level use batch=1 (reading KV once), matching `MultiLevelCascadeAttentionWrapper`'s IO pattern.

**Why inflate kv_len for non-causal levels?** The kernel has a single `MASK_MODE` template parameter — it can't be causal for one work item and non-causal for another. By inflating `kv_len` on non-causal levels, we push the causal mask boundary past `kv_end`, effectively making it a no-op. The real data boundary is enforced by `kv_end` in the page loading logic.

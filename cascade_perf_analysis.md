# Fused Cascade Performance Analysis

## Setup

- GPU: NVIDIA RTX A6000 (84 SMs, 768 GB/s peak BW)
- batch=16, num_qo_heads=num_kv_heads=8, head_dim=128, page_size=16, qo_len=1
- 2-level cascade: shared prefix + unique suffix (unique_kv_len=8)
- Per-level qo_indptr: shared level batch=1 (reads KV once), unique level batch=16

## End-to-End Results

| shared_kv_len | MultiLevel | Fused Cascade | Ratio | Winner |
|--------------|-----------|--------------|-------|--------|
| 256 | 0.1044 ms | 0.0287 ms | 0.27x | Fused |
| 1024 | 0.1044 ms | 0.0287 ms | 0.27x | Fused |
| 4096 | 0.1096 ms | 0.0768 ms | 0.70x | Fused |
| 8192 | 0.1096 ms | 0.1382 ms | 1.26x | MultiLevel |
| 16384 | 0.1219 ms | 0.2621 ms | 2.15x | MultiLevel |

**Fused wins at short sequences** because it avoids MultiLevel's ~100us launch overhead (3 kernel launches: prefill + decode + merge). At kv_len=256-1024, the actual compute is negligible — launch overhead dominates, and fused pays it only once.

**Fused loses at long sequences** because it underutilizes the GPU on the shared prefix level. The gap grows linearly with kv_len.

## Root Cause: No KV Splitting → 10% SM Utilization

The shared prefix level has batch=1 and packed_qo_len=16 (with gqa_group=1). The cascade scheduler creates **one work item per (qo_tile, kv_head)**:

```
work_items = ceil(packed_qo_len / CTA_TILE_Q) * num_kv_heads
           = ceil(16 / 16) * 8
           = 8
```

On an 84-SM GPU, 8 work items means **8 active SMs out of 84 (10%)**. Each SM must serially iterate through all KV tokens. There is no KV splitting — the entire kv_len is processed in one work item.

### Bandwidth comparison (shared prefix in isolation)

| kv_len | BatchPrefill | Persistent Runner2 | Ratio | Active SMs |
|--------|-------------|-------------------|-------|------------|
| 1024 | 0.052 ms (81 GB/s) | 0.028 ms (152 GB/s) | 0.53x | 8/84 |
| 4096 | 0.058 ms (287 GB/s) | 0.076 ms (221 GB/s) | 1.30x | 8/84 |
| 8192 | 0.065 ms (520 GB/s) | 0.137 ms (245 GB/s) | 2.13x | 8/84 |
| 16384 | 0.110 ms (612 GB/s) | 0.260 ms (258 GB/s) | 2.37x | 8/84 |
| 32768 | 0.198 ms (679 GB/s) | 0.507 ms (265 GB/s) | 2.56x | 8/84 |

The persistent kernel saturates at **~265 GB/s** (35% peak) because only 8 SMs are active. BatchPrefill scales to **~680 GB/s** (88% peak) by splitting KV across all 84 SMs.

At short kv_len (1024), the persistent kernel is faster despite fewer SMs because it avoids prefill's per-kernel overhead and the 8 SMs have enough work to stay busy with 32 iterations each.

### Why not Runner1?

With gqa_group=1, packed_qo_len=16, which is NOT > 16, so the shared level goes to Runner2 (CTA_TILE_Q=16). Even if we forced it to Runner1 (CTA_TILE_Q=128), Runner1 is **worse** (7-8x slower than prefill) because:

- CTA_TILE_Q=128 with only 16 actual Q positions wastes 112/128 = 87.5% of the Q tile
- Runner1's larger tiles increase register pressure without useful compute

## What BatchPrefill Does Differently

BatchPrefill's `BatchPrefillWithPagedKVCacheWrapper` uses a scheduler that **splits KV across SMs**. For batch=1, qo_len=16, kv_len=8192:

1. It divides the KV range into chunks (e.g., ~100 tokens each)
2. Creates ~80+ work items distributed across all 84 SMs
3. Each SM processes a small KV chunk and writes a partial output
4. A merge pass combines the partials

This achieves full SM occupancy and near-peak memory bandwidth.

## Fix: KV Splitting in CascadeHolisticPlan

The cascade scheduler should split long KV sequences into chunks:

```
For each (level, request, kv_head):
    kv_len = kv_len_arr[level][request]
    num_kv_chunks = ceil(kv_len / kv_chunk_size)
    for chunk_idx in range(num_kv_chunks):
        create work item:
            kv_start = chunk_idx * kv_chunk_size
            kv_end = min((chunk_idx + 1) * kv_chunk_size, kv_len)
            cascade_kv_chunk_idx = level * max_kv_chunks + chunk_idx
            cascade_num_kv_chunks = num_levels * max_kv_chunks
```

The reduction runner already handles merging multiple partials per output row — `cascade_num_kv_chunks` just increases from `num_levels` to `num_levels * max_kv_chunks`, and `merge_indptr` expands accordingly.

This would increase shared level work items from 8 to `8 * ceil(8192 / chunk_size)`, achieving full SM utilization while still reading KV only once per level.

### Expected impact

With KV splitting at kv_len=8192 and chunk_size=128:
- Work items: 8 * ceil(8192/128) = 8 * 64 = 512 (vs current 8)
- SM utilization: 84/84 = 100% (vs current 10%)
- Expected BW: ~600-680 GB/s (vs current 265 GB/s)
- Expected latency: ~0.06-0.07 ms (vs current 0.14 ms), competitive with BatchPrefill

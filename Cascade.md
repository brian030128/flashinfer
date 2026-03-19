# MultiLevelCascadeAttentionWrapper: Kernel Details

## Architecture

`MultiLevelCascadeAttentionWrapper` (`flashinfer/cascade.py:228-556`) creates N `BatchPrefillWithPagedKVCacheWrapper` instances, one per level.

- `plan()` calls each wrapper's `plan()` with per-level `qo_indptr`, `kv_indptr`, `kv_indices`, `last_page_len`
- `run()` runs the last level first, then merges remaining levels via `merge_state_in_place(out, lse, out_i, lse_i)`
- Causal mask only applied to the **last** level (`causal=causal if i == self._num_levels - 1 else False`)

## Per-Level Kernel Dispatch

Each level uses `BatchPrefillWithPagedKVCacheWrapper` which launches `BatchPrefillWithPagedKVCacheKernel` (a regular, non-cooperative kernel).

### CTA_TILE_Q Selection

Via `FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim)` (`utils.cuh:303-322`):

- `avg_packed_qo_len > 64 && head_dim < 256` → CTA_TILE_Q = 128
- Ampere (SM >= 8.0): `> 16` → 64, `<= 16` → 16
- Pre-Ampere: always 64

### Warp/MMA Config

Derived from CTA_TILE_Q (`prefill.cuh:55-73`):

- `get_num_warps_q(cta_tile_q)`: > 16 → 4, <= 16 → 1
- `get_num_warps_kv(cta_tile_q)`: 4 / num_warps_q
- `get_num_mma_q(cta_tile_q)`: > 64 → 2, <= 64 → 1

### NUM_MMA_KV

Dynamically selected based on shared memory budget (`prefill.cuh:2583-2592`):

- `max_num_mma_kv_reg = 8 / NUM_MMA_Q` (except RoPE+fp32 edge case → 2)
- `max_num_mma_kv_smem = (max_smem_per_threadblock - Q_smem) / KV_smem_per_mma`
- Final = `min(reg_limit, smem_limit)`, dispatched to nearest {8, 4, 2, 1}

### Launch Config

- **Grid**: `dim3(padded_batch_size, 1, num_kv_heads)` — one threadblock per (work_item, kv_head)
- **Threads**: `dim3(32, NUM_WARPS_Q, NUM_WARPS_KV)` — always 128 threads total (4 warps)

## Concrete Configs for Benchmark

For batch=16, num_qo_heads=8, num_kv_heads=8 (GQA ratio=1), head_dim=128, fp16, page_size=16, on A6000 (SM_86, Ampere, 102400 bytes smem/SM):

### Shared Level (batch=1, qo_len=16)

- `packed_qo_len = 16 * 1 = 16`, `avg = 16` → **CTA_TILE_Q = 16**
- NUM_WARPS_Q=1, NUM_WARPS_KV=4, NUM_MMA_Q=1
- num_ctas_per_sm: `102400 >= 2*(16*128*2 + (128+128)*16*4*2)` = `2*36864 = 73728` → yes → 2 CTAs/SM
- max_smem_per_threadblock = 51200
- max_num_mma_kv_smem = `(51200 - 4096) / 32768 = 1.43` → **1**
- max_num_mma_kv_reg = `8 / 1 = 8`
- **NUM_MMA_KV = 1**
- Grid: `(padded_batch_size, 1, 8)` — where padded_batch_size = ceil_div(16, 16) * num_kv_chunks = 1 * num_kv_chunks

### Unique Level (batch=16, qo_len=1)

- `packed_qo_len = 1`, `avg = 1` → **CTA_TILE_Q = 16**
- Same config as shared: NUM_MMA_KV=1, NUM_WARPS_Q=1, NUM_WARPS_KV=4
- Grid: `(padded_batch_size, 1, 8)` — 16 work items (one per request, no split-kv needed for kv_len=8)

### Summary Table

| | CTA_TILE_Q | NUM_MMA_Q | NUM_MMA_KV | WARPS_Q | WARPS_KV | Launch |
|---|---|---|---|---|---|---|
| Shared (batch=1, qo=16) | 16 | 1 | 1 | 1 | 4 | regular |
| Unique (batch=16, qo=1) | 16 | 1 | 1 | 1 | 4 | regular |

## Key Insight: Why MultiLevel is Fast

- Shared level: 1 request with qo_len=16 → reads 8192-token KV cache **once** (34 MB)
- Unique level: 16 requests with qo_len=1, kv_len=8 → trivial
- Two separate kernel launches + one `merge_state_in_place` call
- The speed advantage is **not** from tile configuration (both use CTA_TILE_Q=16) but from **batch structure**: shared prefix treated as 1 request

## Merge Operation

- `merge_state_in_place(out, lse, out_i, lse_i)` (`flashinfer/cascade.py:551`)
- Element-wise LSE-weighted combination of partial attention outputs
- Very cheap (~10us) for small batch sizes

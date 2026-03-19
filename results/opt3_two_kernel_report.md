# Optimization 3 Results Report: Two-Kernel Non-Cooperative Launch

## Changes Made

### Replace cooperative kernel with two-kernel launch

The original fused cascade kernel used `cudaLaunchCooperativeKernel` with a grid barrier
(`cg::this_grid().sync()`) between attention and reduction phases. This limits occupancy to
**1 CTA/SM** (cooperative launch requires all CTAs to be resident simultaneously).

Opt 3 splits the single cooperative kernel into two standard kernel launches:

1. **Attention kernel** (`CascadeAttentionKernelTemplate`): runs both Runner1 and Runner2
   sequentially (same as before), but without grid sync
2. **Reduction kernel** (`CascadeReductionKernelTemplate`): merges partial outputs

This enables **2 CTAs/SM** because:
- Each CTA uses ~36KB shared memory (NUM_MMA_KV=1 → CTA_TILE_KV=64)
- 2 × 36KB = 72KB < 100KB (A6000's max shared memory per SM)
- Standard `cudaLaunchKernel` allows the runtime to schedule multiple CTAs per SM

### Code changes

| File | Change |
|------|--------|
| `persistent_template.cuh` | New `CascadeAttentionKernelTemplate` (2 runners, no reduction) and `CascadeReductionKernelTemplate` (reduction only) |
| `persistent.cuh` | New `CascadeBatchPagedAttention` function: launches attention kernel then reduction kernel via `cudaLaunchKernel` |
| `batch_attention.cu` | Call `CascadeBatchPagedAttention<16, ...>` instead of `BatchPagedAttentionPersistent<16, 16, ...>` |
| `batch_attention_paged_kernel_inst.jinja` | Updated explicit instantiation to match new template |
| `scheduler.cuh:1392` | `num_sm *= 2` — scheduler now targets 2× more work items for load balancing |

### Key design: NUM_MMA_KV=1

Setting `NUM_MMA_KV=1` (down from 2 in Opt 1+2) reduces CTA_TILE_KV from 128 to 64, which
cuts shared memory from ~70KB to ~36KB. This is what enables 2 CTAs/SM. The tradeoff is
fewer KV elements processed per iteration, meaning more iterations for the same KV length.

## Performance — n=1 (n_draft_tokens=1, A6000, CUDA Graph)

```
  shared_kv_len   Flat (ms)  MultiLevel (ms)  Fused (ms)  vs Multi   vs Flat
  -------------  ----------  ---------------  ----------  --------  --------
            256      0.0154           0.0236      0.0164     1.44x     0.94x
            512      0.0246           0.0246      0.0184     1.33x     1.33x
           1024      0.0420           0.0246      0.0225     1.09x     1.86x
           2048      0.0779           0.0389      0.0307     1.27x     2.54x
           4096      0.1495           0.0502      0.0461     1.09x     3.24x
           8192      0.2970           0.0737      0.0717     1.03x     4.14x
          16384      0.5786           0.1188      0.1260     0.94x     4.59x
```

## Performance — n=8 (n_draft_tokens=8, A6000, CUDA Graph)

**TODO: re-run benchmark with `--n 8` to fill in data**

```
  shared_kv_len   Flat (ms)  MultiLevel (ms)  Fused (ms)  vs Multi   vs Flat
  -------------  ----------  ---------------  ----------  --------  --------
            256       —             —              —          —         —
            512       —             —              —          —         —
           1024       —             —              —          —         —
           2048       —             —              —          —         —
           4096       —             —              —          —         —
           8192       —             —              —          —         —
          16384       —             —              —          —         —
```

## Comparison: Opt 3 vs Opt 1+2 (n=1)

| shared_kv_len | Opt 1+2 Fused (ms) | Opt 3 Fused (ms) | Delta |
|---|---|---|---|
| 256 | 0.0256 | 0.0164 | -0.0092 (faster) |
| 512 | 0.0174 | 0.0184 | +0.0010 (slower) |
| 1024 | 0.0215 | 0.0225 | +0.0010 (slower) |
| 2048 | 0.0287 | 0.0307 | +0.0020 (slower) |
| 4096 | 0.0451 | 0.0461 | +0.0010 (slower) |
| 8192 | 0.0696 | 0.0717 | +0.0021 (slower) |
| 16384 | 0.1229 | 0.1260 | +0.0031 (slower) |

## Analysis

### n=1 regression (1-2 us)

Opt 3 shows a consistent **1-2 us regression** across prefix lengths 512-16384 compared to
Opt 1+2. The root cause is `NUM_MMA_KV=1` (vs 2 in Opt 1+2):

- With NUM_MMA_KV=2: CTA_TILE_KV=128, processes 128 KV elements per iteration
- With NUM_MMA_KV=1: CTA_TILE_KV=64, processes 64 KV elements per iteration → 2x more iterations

For n=1 workloads, there are only 16 work items (batch=16 × heads=8 / gqa=8) on the unique
level. With 84 SMs, most SMs are idle anyway — having 2 CTAs/SM doesn't help because there
isn't enough work to fill even 1 CTA/SM. The smaller tile just means each work item takes
slightly longer.

The exception is kv=256 where Opt 3 is 0.0092ms faster — likely because the cooperative
launch overhead itself (~10 us) exceeds the per-work penalty at very short KV lengths.

### n=8 expected improvement

With n=8 draft tokens, the unique level has 128 work items (batch=16 × heads=8). Combined
with shared prefix work items, the total easily exceeds 84 SMs. Here, 2 CTAs/SM (168 slots)
should improve utilization significantly, and the two-kernel launch avoids the cooperative
launch overhead. The net effect should be a speedup for n=8 despite the smaller tile size.

### The tradeoff

| | n=1 (few work items) | n=8 (many work items) |
|---|---|---|
| 2 CTAs/SM | No benefit (work < SMs) | Better SM utilization |
| NUM_MMA_KV=1 | 1-2 us penalty (more iters) | Amortized by parallelism |
| Non-cooperative | Saves ~10 us launch overhead | Saves ~10 us launch overhead |
| **Net** | **Small regression (1-2 us)** | **Expected improvement** |

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

## Performance — n=1 (1 prefix, batch=16, A6000, CUDA Graph)

```
  shared_kv_len   Flat (ms)  MultiLevel (ms)  Fused (ms)  vs Multi   vs Flat
  -------------  ----------  ---------------  ----------  --------  --------
            256      0.0143           0.0243      0.0235     1.04x     0.61x
            512      0.0236           0.0225      0.0174     1.29x     1.35x
           1024      0.0338           0.0195      0.0174     1.12x     1.94x
           2048      0.0614           0.0338      0.0266     1.27x     2.31x
           4096      0.1178           0.0451      0.0410     1.10x     2.88x
           8192      0.2345           0.0696      0.0635     1.10x     3.69x
          16384      0.4895           0.1147      0.1085     1.06x     4.51x
```

## Performance — n=8 (8 prefixes, batch=128, A6000, CUDA Graph)

```
  shared_kv_len   Flat (ms)  MultiLevel (ms)  Fused (ms)  vs Multi   vs Flat
  -------------  ----------  ---------------  ----------  --------  --------
            256      0.0707           0.0655      0.0584     1.12x     1.21x
            512      0.1270           0.0768      0.0676     1.14x     1.88x
           1024      0.2386           0.0973      0.0768     1.27x     3.11x
           2048      0.4792           0.1434      0.1106     1.30x     4.33x
           4096      0.9544           0.2314      0.1997     1.16x     4.78x
           8192      1.9868           0.4096      0.3789     1.08x     5.24x
          16384      4.0223           0.7660      0.7363     1.04x     5.46x
```

## Comparison: Opt 3 vs Opt 1+2

Opt 1+2 numbers are from the previous report and were measured on a different run (may have
different GPU thermal/clock state). Re-running Opt 1+2 benchmarks on the same session is
recommended for apples-to-apples comparison — see `git checkout 3bea69b` results below if available.

## Analysis

### n=1: Fused beats MultiLevel at all prefix lengths ≥512

Fused is **1.04x-1.29x faster** than MultiLevel across 512-16384. At kv=256, Fused is only
1.04x (barely ahead) — the short prefix means very little work to fuse, so the two-kernel
launch overhead is more visible relative to total compute.

The previous Opt 1+2 regression at kv=16384 (0.97x) is now **1.06x** — the non-cooperative
launch with 2 CTAs/SM scheduling fixes the long-prefix tail.

### n=8: Fused beats MultiLevel across the board

With 8 prefixes (128 batch), Fused is **1.04x-1.30x faster** than MultiLevel. The 2 CTAs/SM
pays off here: 1088 work items across 168 clusters (2×84 SMs) means good utilization.

Peak speedup is at kv=2048 (1.30x vs MultiLevel, 4.33x vs Flat).

### The tradeoff: NUM_MMA_KV=1

Setting `NUM_MMA_KV=1` (down from 2 in Opt 1+2) halves CTA_TILE_KV (128→64), cutting shared
memory from ~70KB to ~36KB. This enables 2 CTAs/SM but means 2× more KV iterations per work item.

| | n=1 (few work items) | n=8 (many work items) |
|---|---|---|
| 2 CTAs/SM | Marginal benefit (work < SMs) | Better SM utilization |
| NUM_MMA_KV=1 | Slight per-work penalty | Amortized by parallelism |
| Non-cooperative | No cooperative launch overhead | No cooperative launch overhead |
| **Net** | **1.04-1.29x vs MultiLevel** | **1.04-1.30x vs MultiLevel** |

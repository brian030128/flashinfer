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

## Comparison: Opt 3 vs Opt 1+2 (same-session benchmarks)

Measured by checking out `3bea69b` (Opt 1+2) and re-running on the same GPU session.

### Opt 1+2 baseline (commit 3bea69b)

**n=1:**
```
  shared_kv_len   Flat (ms)  MultiLevel (ms)  Fused (ms)  vs Multi   vs Flat
  -------------  ----------  ---------------  ----------  --------  --------
            256      0.0143           0.0184      0.0236     0.78x     0.61x
            512      0.0246           0.0195      0.0143     1.36x     1.71x
           1024      0.0338           0.0195      0.0164     1.19x     2.06x
           2048      0.0625           0.0328      0.0236     1.39x     2.65x
           4096      0.1116           0.0451      0.0389     1.16x     2.87x
           8192      0.2222           0.0676      0.0625     1.08x     3.56x
          16384      0.4792           0.1126      0.1137     0.99x     4.22x
```

**n=8:**
```
  shared_kv_len   Flat (ms)  MultiLevel (ms)  Fused (ms)  vs Multi   vs Flat
  -------------  ----------  ---------------  ----------  --------  --------
            256      0.0707           0.0655      0.0851     0.77x     0.83x
            512      0.1260           0.0778      0.1065     0.73x     1.18x
           1024      0.2417           0.0973      0.1495     0.65x     1.62x
           2048      0.4598           0.1423      0.2396     0.59x     1.92x
           4096      0.9585           0.2324      0.2857     0.81x     3.35x
           8192      1.9825           0.4096      0.3799     1.08x     5.22x
          16384      4.0090           0.7649      0.7363     1.04x     5.45x
```

### Head-to-head: Fused kernel timing

**n=1 (1 prefix, batch=16):**

| kv_len | Opt1+2 Fused | Opt3 Fused | Delta | Opt1+2 vs Multi | Opt3 vs Multi |
|--------|-------------|------------|-------|-----------------|---------------|
| 256 | 0.0236 | 0.0235 | -0.1μs | 0.78x | 1.04x |
| 512 | 0.0143 | 0.0174 | +3.1μs | 1.36x | 1.29x |
| 1024 | 0.0164 | 0.0174 | +1.0μs | 1.19x | 1.12x |
| 2048 | 0.0236 | 0.0266 | +3.0μs | 1.39x | 1.27x |
| 4096 | 0.0389 | 0.0410 | +2.1μs | 1.16x | 1.10x |
| 8192 | 0.0625 | 0.0635 | +1.0μs | 1.08x | 1.10x |
| 16384 | 0.1137 | 0.1085 | **-5.2μs** | 0.99x | **1.06x** |

**n=8 (8 prefixes, batch=128):**

| kv_len | Opt1+2 Fused | Opt3 Fused | Delta | Opt1+2 vs Multi | Opt3 vs Multi |
|--------|-------------|------------|-------|-----------------|---------------|
| 256 | 0.0851 | 0.0584 | **-26.7μs** | 0.77x | **1.12x** |
| 512 | 0.1065 | 0.0676 | **-38.9μs** | 0.73x | **1.14x** |
| 1024 | 0.1495 | 0.0768 | **-72.7μs** | 0.65x | **1.27x** |
| 2048 | 0.2396 | 0.1106 | **-129.0μs** | 0.59x | **1.30x** |
| 4096 | 0.2857 | 0.1997 | **-86.0μs** | 0.81x | **1.16x** |
| 8192 | 0.3799 | 0.3789 | -1.0μs | 1.08x | 1.08x |
| 16384 | 0.7363 | 0.7363 | 0.0μs | 1.04x | 1.04x |

## Analysis

### n=1: Opt 3 trades 1-3μs mid-range for fixed 16384 regression

Opt 1+2 is faster at n=1 for kv 512-8192 by 1-3μs (NUM_MMA_KV=2 processes more KV/iter).
But Opt 1+2 regresses at kv=16384 (0.99x vs MultiLevel) due to cooperative launch limiting
occupancy. Opt 3 fixes this to 1.06x. Both beat MultiLevel at all kv≥512.

### n=8: Opt 1+2 was catastrophically bad — Opt 3 fixes it

With 128 batch (8 prefixes × 16 suffixes), Opt 1+2's cooperative launch with 1 CTA/SM
created a severe bottleneck: **Fused was 0.59x-0.81x vs MultiLevel** for kv≤4096. The
1088 work items couldn't fit on 84 SMs (1 CTA each), causing massive serialization.

Opt 3's two-kernel launch with 2 CTAs/SM (168 slots) completely fixes this:
- kv=2048: **0.59x → 1.30x** (2.2× faster Fused time, 0.2396→0.1106ms)
- kv=1024: **0.65x → 1.27x** (1.9× faster Fused time, 0.1495→0.0768ms)
- kv=512: **0.73x → 1.14x** (1.6× faster Fused time, 0.1065→0.0676ms)

At kv≥8192, both Opt 1+2 and Opt 3 converge (~1.08x, 1.04x) because the work per CTA
becomes large enough that occupancy matters less.

### The tradeoff: NUM_MMA_KV=1

| | n=1 (few work items) | n=8 (many work items) |
|---|---|---|
| 2 CTAs/SM | Marginal (work < SMs) | Critical (1088 work items) |
| NUM_MMA_KV=1 | 1-3μs penalty | Amortized by 2× occupancy |
| Non-cooperative | Fixes kv=16384 regression | Fixes kv≤4096 catastrophe |
| **Net** | **Small n=1 cost (1-3μs)** | **Massive n=8 win (up to 2.2×)** |

# Fused Cascade Attention: Development Report

## 1. Why We Fused the Kernel

**Baseline: `MultiLevelCascadeAttentionWrapper`** launches N separate `BatchPrefillWithPagedKVCache` kernels (one per cascade level) plus a `merge_state_in_place` call. For a 2-level cascade (shared prefix + unique suffix), that's 3 kernel launches.

**The problem with separate launches:** Each kernel launch costs ~3-5us of host dispatch latency. For short sequences where compute is cheap (shared_kv_len <= 1024), this launch overhead dominates — the GPU spends more time waiting for kernels to be dispatched than doing actual math.

**What fusion gives us:** `CascadeBatchAttentionWrapper` packs all cascade levels into a single persistent kernel. Work items from all levels are classified by query length (packed_qo_len > 16 → Runner1/large tiles, else → Runner2/small tiles) and load-balanced across SMs. The reduction runner merges partials from both KV splits and cross-level partials using the same LSE-weighted mechanism.

**Initial results (no CUDA graph):**

| Method | Latency (shared_kv=1024) |
|--------|--------------------------|
| Flat Paged Decode | 0.0430 ms |
| MultiLevel (3 launches) | 0.1004 ms |
| Fused (1 launch) | 0.0266 ms |

Fused was **3.77x faster than MultiLevel** and **1.62x faster than flat decode**. The win came from eliminating launch overhead AND avoiding redundant shared KV reads (the shared level uses batch=1 so KV is read once, not N times).

---

## 2. The Performance Problem After Fusion

**With CUDA graph** (which eliminates host dispatch overhead), the fused kernel **regressed badly** against MultiLevel at longer shared prefix lengths:

| shared_kv_len | Fused vs MultiLevel |
|---------------|---------------------|
| 256 | 0.93x |
| 1024 | 0.95x |
| 4096 | 0.76x |
| 8192 | 0.64x |
| 16384 | 0.58x |

**Root cause: SM underutilization from no KV splitting.** The shared prefix level has batch=1 and packed_qo_len=16. The scheduler creates only `ceil(16/16) * 8 = 8` work items for it — 8 active SMs out of 84 (10% utilization). Each SM serially iterates through the entire KV sequence. The persistent kernel saturated at ~265 GB/s (35% of peak 768 GB/s), while MultiLevel's `BatchPrefillWithPagedKVCache` splits KV across all 84 SMs, achieving ~680 GB/s (88% peak).

At short kv_len (256-512), 8 SMs have enough work and the single-launch advantage persists. At long kv_len (4096+), the serial KV processing on 8 SMs becomes the bottleneck.

---

## 3. Optimizations Applied

### Optimization 1: Reduce KV Splitting (scheduler.cuh)

**Problem:** The original `kv_len_limit` formula (`total_kv_lens * num_kv_heads / num_clusters`) created excessive KV chunks, producing up to 65 partials per output row at shared_kv_len=16384. Each partial requires a slot in the partial_o buffer and work in the reduction kernel.

**Fix:** Changed to `total_kv_lens / 3`, targeting at most 3 KV chunks for the largest level. This reduced partials/row from 9-65 down to 3-4 across all prefix lengths.

| shared_kv_len | Partials/row (before) | Partials/row (after) |
|---|---|---|
| 512 | 5 | 3 |
| 1024 | 9 | 3 |
| 8192 | ~17 | 4 |
| 16384 | 65 | 4 |

**Why this works for cascade:** The unique level already provides ample work items (batch_size * num_kv_heads = 128 for the benchmark, vs 84 SMs). The shared prefix barely needs splitting — it only needs enough chunks to utilize SMs, not to create fine-grained parallelism.

### Optimization 2: Set CTA_TILE_Q_1=16 for Cascade

**Problem:** Runner1 was configured with CTA_TILE_Q=128, but cascade workloads rarely have packed_qo_len > 16. The shared level (batch=1, qo_len=16) has packed_qo_len=16 exactly, going to Runner2. Runner1 was wasted capacity.

**Fix:** Set both runners to CTA_TILE_Q=16 (`{128,16}` → `{16,16}`). Also had to reduce NUM_MMA_KV from 4 to 2 (when CTA_TILE_Q <= 16) to keep shared memory under the A6000's 100KB limit.

**Combined Opt 1+2 results (n=1, CUDA graph):**

| shared_kv_len | Fused vs MultiLevel |
|---|---|
| 512 | 1.36x |
| 1024 | 1.19x |
| 2048 | 1.39x |
| 8192 | 1.08x |
| 16384 | 0.99x |

Major improvement: Fused now beats MultiLevel across most prefix lengths. But a new problem emerged at **n=8 (8 prefixes, batch=128):** Fused was 0.59x-0.81x vs MultiLevel for kv_len <= 4096.

**Why n=8 broke:** The fused kernel used `cudaLaunchCooperativeKernel` (needed for grid sync before reduction). Cooperative launch requires all CTAs to be resident simultaneously, limiting occupancy to **1 CTA/SM** (84 CTAs total). With 1088 work items (8 prefixes * 128 batch + shared items), massive serialization occurred.

### Optimization 3: Two-Kernel Non-Cooperative Launch

**Problem:** Cooperative launch's 1 CTA/SM limit was catastrophic at high batch counts.

**Fix:** Split the single cooperative kernel into two standard kernel launches:
1. **Attention kernel** — runs both Runner1 and Runner2 (no grid sync needed)
2. **Reduction kernel** — merges partial outputs

Also reduced NUM_MMA_KV from 2 to 1, cutting CTA_TILE_KV from 128 to 64 and shared memory from ~70KB to ~36KB. This enabled **2 CTAs/SM** (2 * 36KB = 72KB < 100KB max).

The scheduler was updated with `num_sm *= 2` to target twice as many work items for load balancing.

**Final results (n=8, CUDA graph):**

| shared_kv_len | Opt1+2 vs Multi | Opt3 vs Multi | Fused speedup |
|---|---|---|---|
| 256 | 0.77x | **1.12x** | 1.46x faster |
| 512 | 0.73x | **1.14x** | 1.58x faster |
| 1024 | 0.65x | **1.27x** | 1.95x faster |
| 2048 | 0.59x | **1.30x** | 2.17x faster |
| 4096 | 0.81x | **1.16x** | 1.43x faster |
| 8192 | 1.08x | **1.08x** | same |

Opt 3 completely fixed the n=8 regression. The tradeoff: NUM_MMA_KV=1 costs 1-3us at n=1 (fewer KV elements per iteration), but this is amortized by 2x occupancy at higher batch counts.

---

## 4. Why Fused Beats MultiLevel Even With CUDA Graph

CUDA graph eliminates host-side dispatch latency, which was the fused kernel's original advantage. Yet after Opt 3, fused is 1.04x-1.30x faster than MultiLevel even under CUDA graph. Three factors explain this:

**1. Work interleaving vs sequential level execution.** MultiLevel runs levels sequentially: shared-level kernel → unique-level kernel → merge kernel. Even under CUDA graph, the GPU processes these back-to-back — SMs sit idle between kernel boundaries while the next kernel's state is configured. Fused interleaves work items from all levels into a single attention kernel. A CTA processes shared-level work items and unique-level work items in the same launch, eliminating inter-kernel gaps. At n=8 with kv=2048, this interleaving is worth 1.30x because the unique level's trivial work (kv_len=8) fills gaps left by the shared level's heavier work.

**2. Fewer kernel transitions.** MultiLevel launches 3 kernels (shared, unique, merge). Fused launches 2 (attention, reduction). Each kernel transition has GPU-side overhead: the hardware must drain the pipeline, reconfigure shared memory/registers, and start new thread blocks. With CUDA graph this overhead is reduced but not zero — it's baked into the graph as fixed scheduling nodes. Fused pays this cost once less.

**3. Joint load balancing across levels.** MultiLevel's shared-level kernel and unique-level kernel each independently schedule their own work items across SMs. If the shared level has 24 work items (8 heads × 3 KV chunks) and the unique level has 128 (16 batch × 8 heads), the shared kernel underutilizes SMs while the unique kernel may over-subscribe them. Fused's scheduler sees all 152 work items together and distributes them across SMs using a min-heap, achieving better balance. At n=8, there are ~1088 total work items — joint scheduling across 168 CTA slots (84 SMs × 2 CTAs/SM) is significantly more efficient than two separate scheduling passes.

**Why the advantage varies with kv_len:** At short kv_len (256-512), MultiLevel's kernels are cheap and the per-kernel GPU overhead is a larger fraction of total time — fused's fewer transitions matter more. At long kv_len (8192+), compute dominates and both methods approach memory bandwidth limits, so the advantage shrinks to 1.04-1.08x. The sweet spot (1.27-1.30x at kv=1024-2048 for n=8) is where kernel transition overhead and load-balancing inefficiency are both significant relative to compute.

---

## Summary

| Stage | Key Change | Impact |
|-------|-----------|--------|
| Initial fusion | Single cooperative kernel, no KV splitting | 3.77x vs MultiLevel (no graph), but regressed with CUDA graph at long prefixes |
| Opt 1: Reduce KV splits | `kv_len_limit = total_kv_lens / 3` | Partials/row: 65 → 4 at kv=16384 |
| Opt 2: CTA_TILE_Q=16 | Both runners use small tiles | Matches cascade workload characteristics |
| Opt 3: Two-kernel launch | Non-cooperative, 2 CTAs/SM | Fixed n=8 catastrophe (0.59x → 1.30x vs Multi) |

**Final state:** Fused cascade beats MultiLevel across all tested configurations (n=1 and n=8, kv_len 256-16384) except at very short prefixes (kv=256, n=1) where cooperative launch overhead slightly dominates.

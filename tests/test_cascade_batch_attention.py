"""
CUDA Graph benchmark: CascadeBatchAttention vs MultiLevelCascadeAttentionWrapper
across shared prefix lengths, with N independent shared prefixes.

Usage:
    python tests/test_cascade_batch_attention.py          # default n=1
    python tests/test_cascade_batch_attention.py --n 3    # 3 prefixes
"""

import argparse

import torch

import flashinfer
from flashinfer.attention import CascadeBatchAttention


def ceil_div(a, b):
    return (a + b - 1) // b


def build_multi_prefix_kv_cache(
    shared_kv_len,
    unique_kv_len,
    num_prefixes,
    suffixes_per_prefix,
    num_heads,
    head_dim,
    page_size,
    kv_layout="NHD",
    dtype=torch.float16,
):
    """Build a 2-level paged KV cache with N shared prefixes + per-suffix unique pages.

    Returns:
        kv_data, shared_kv_indices, shared_kv_indptr, shared_last_page_len,
        shared_kv_len_tensor, unique_kv_indices, unique_kv_indptr,
        unique_last_page_len, unique_kv_len_tensor
    """
    total_batch = num_prefixes * suffixes_per_prefix
    num_shared_pages_per_prefix = ceil_div(shared_kv_len, page_size)
    num_unique_pages_per_seq = ceil_div(unique_kv_len, page_size)
    total_shared_pages = num_prefixes * num_shared_pages_per_prefix
    total_unique_pages = total_batch * num_unique_pages_per_seq
    total_pages = total_shared_pages + total_unique_pages

    kv_data = torch.zeros(
        total_pages, 2, page_size, num_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    # Shared prefixes: each prefix p uses pages [p*nsp, (p+1)*nsp)
    for p in range(num_prefixes):
        k_shared = torch.randn(shared_kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
        v_shared = torch.randn(shared_kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
        page_start = p * num_shared_pages_per_prefix
        kv_idx = torch.arange(page_start, page_start + num_shared_pages_per_prefix,
                              device="cuda", dtype=torch.int32)
        kv_indptr = torch.tensor([0, num_shared_pages_per_prefix], device="cuda", dtype=torch.int32)
        last_page_len = torch.tensor(
            [(shared_kv_len - 1) % page_size + 1], device="cuda", dtype=torch.int32
        )
        append_indptr = torch.tensor([0, shared_kv_len], device="cuda", dtype=torch.int32)
        flashinfer.append_paged_kv_cache(
            k_shared, v_shared,
            *flashinfer.get_batch_indices_positions(
                append_indptr,
                flashinfer.get_seq_lens(kv_indptr, last_page_len, page_size),
                shared_kv_len,
            ),
            kv_data, kv_idx, kv_indptr, last_page_len, kv_layout,
        )

    # Shared level metadata (num_prefixes requests)
    shared_kv_indices_list = []
    shared_kv_indptr_list = [0]
    for p in range(num_prefixes):
        page_start = p * num_shared_pages_per_prefix
        shared_kv_indices_list.append(
            torch.arange(page_start, page_start + num_shared_pages_per_prefix,
                         device="cuda", dtype=torch.int32)
        )
        shared_kv_indptr_list.append(shared_kv_indptr_list[-1] + num_shared_pages_per_prefix)
    shared_kv_indices = torch.cat(shared_kv_indices_list)
    shared_kv_indptr = torch.tensor(shared_kv_indptr_list, device="cuda", dtype=torch.int32)
    shared_last_page_len = torch.full(
        (num_prefixes,), (shared_kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
    )
    shared_kv_len_tensor = torch.full(
        (num_prefixes,), shared_kv_len, device="cuda", dtype=torch.int32
    )

    # Unique suffixes: pages [total_shared_pages, ...)
    k_unique = torch.randn(total_batch * unique_kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
    v_unique = torch.randn(total_batch * unique_kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
    unique_kv_indices = (
        torch.arange(total_unique_pages, device="cuda", dtype=torch.int32)
        + total_shared_pages
    )
    unique_kv_indptr = (
        torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * num_unique_pages_per_seq
    )
    unique_last_page_len = torch.full(
        (total_batch,), (unique_kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
    )
    unique_append_indptr = (
        torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * unique_kv_len
    )
    flashinfer.append_paged_kv_cache(
        k_unique, v_unique,
        *flashinfer.get_batch_indices_positions(
            unique_append_indptr,
            flashinfer.get_seq_lens(unique_kv_indptr, unique_last_page_len, page_size),
            total_batch * unique_kv_len,
        ),
        kv_data, unique_kv_indices, unique_kv_indptr, unique_last_page_len, kv_layout,
    )

    unique_kv_len_tensor = torch.full(
        (total_batch,), unique_kv_len, device="cuda", dtype=torch.int32
    )

    return (
        kv_data,
        shared_kv_indices, shared_kv_indptr, shared_last_page_len,
        shared_kv_len_tensor,
        unique_kv_indices, unique_kv_indptr, unique_last_page_len,
        unique_kv_len_tensor,
    )


def benchmark_cuda_graph(num_prefixes=1, warmup=50, repeat=200):
    """CUDA Graph benchmark across shared prefix lengths with N prefixes."""
    torch.manual_seed(42)

    unique_kv_len = 8
    suffixes_per_prefix = 16
    total_batch = num_prefixes * suffixes_per_prefix
    # Llama 3.1 8B attention config (GQA: 32 query heads, 8 KV heads)
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1
    dtype = torch.bfloat16

    shared_kv_lens = [256, 512, 1024, 2048, 4096, 8192, 16384]
    results = []

    for skv_len in shared_kv_lens:
        torch.manual_seed(42)

        (
            kv_data,
            shared_kv_indices, shared_kv_indptr, shared_last_page_len,
            shared_kv_len_tensor,
            unique_kv_indices, unique_kv_indptr, unique_last_page_len,
            unique_kv_len_tensor,
        ) = build_multi_prefix_kv_cache(
            skv_len, unique_kv_len, num_prefixes, suffixes_per_prefix,
            num_kv_heads, head_dim, page_size, dtype=dtype,
        )

        q = torch.randn(total_batch * qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)

        num_shared_pages_per_prefix = ceil_div(skv_len, page_size)
        num_unique_pages = ceil_div(unique_kv_len, page_size)

        # --- Flat Decode with CUDA Graph ---
        total_kv_len = skv_len + unique_kv_len
        flat_kv_indices_list = []
        for b in range(total_batch):
            prefix_idx = b // suffixes_per_prefix
            page_start = prefix_idx * num_shared_pages_per_prefix
            flat_kv_indices_list.append(
                torch.arange(page_start, page_start + num_shared_pages_per_prefix,
                             device="cuda", dtype=torch.int32)
            )
            flat_kv_indices_list.append(
                unique_kv_indices[b * num_unique_pages : (b + 1) * num_unique_pages]
            )
        flat_kv_indices = torch.cat(flat_kv_indices_list)
        flat_pages_per_req = num_shared_pages_per_prefix + num_unique_pages
        flat_kv_indptr = (
            torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * flat_pages_per_req
        )
        flat_last_page_len = torch.full(
            (total_batch,), (total_kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
        )

        flat_decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
        )
        flat_decode.plan(
            flat_kv_indptr, flat_kv_indices, flat_last_page_len,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            q_data_type=dtype,
        )

        for _ in range(warmup):
            flat_decode.run(q, kv_data)
        torch.cuda.synchronize()

        flat_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(flat_graph):
            flat_decode.run(q, kv_data)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_events[i].record()
            flat_graph.replay()
            end_events[i].record()
        torch.cuda.synchronize()
        flat_graph_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        flat_graph_median = sorted(flat_graph_times)[len(flat_graph_times) // 2]

        # --- MultiLevel with CUDA Graph ---
        ref_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
            2, torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD"
        )
        # Shared level: each prefix owns suffixes_per_prefix query tokens
        qo_indptr_shared_ref = (
            torch.arange(num_prefixes + 1, device="cuda", dtype=torch.int32)
            * (suffixes_per_prefix * qo_len)
        )
        qo_indptr_unique_ref = (
            torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * qo_len
        )
        ref_wrapper.plan(
            [qo_indptr_shared_ref, qo_indptr_unique_ref],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_last_page_len, unique_last_page_len],
            num_qo_heads, num_kv_heads, head_dim, page_size,
            causal=True,
            q_data_type=dtype,
        )

        # Print MultiLevel dispatch info
        print(f"\n--- MultiLevel dispatch (shared_kv_len={skv_len}) ---")
        level_configs = [
            (qo_indptr_shared_ref, shared_kv_indptr, skv_len),
            (qo_indptr_unique_ref, unique_kv_indptr, unique_kv_len),
        ]
        for level_idx, (qo_indptr_l, kv_indptr_l, kv_len_l) in enumerate(level_configs):
            batch_sz = kv_indptr_l.shape[0] - 1
            total_qo = qo_indptr_l[-1].item()
            max_qo_per_req = max((qo_indptr_l[i+1] - qo_indptr_l[i]).item() for i in range(batch_sz))
            packed_qo = max_qo_per_req * (num_qo_heads // num_kv_heads)  # gqa_ratio=4
            print(f"  Level {level_idx}: batch={batch_sz}, total_qo={total_qo}, "
                  f"max_qo_per_req={max_qo_per_req}, packed_qo={packed_qo}, kv_len={kv_len_l}"
                  f" → MultiLevel uses separate kernel launch (non-cooperative, can get 2 CTAs/SM)")

        for _ in range(warmup):
            ref_wrapper.run(q, kv_data)
        torch.cuda.synchronize()

        ref_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ref_graph):
            ref_wrapper.run(q, kv_data)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_events[i].record()
            ref_graph.replay()
            end_events[i].record()
        torch.cuda.synchronize()
        ref_graph_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        ref_graph_median = sorted(ref_graph_times)[len(ref_graph_times) // 2]

        # --- Fused Cascade with CUDA Graph ---
        total_kv_pages = shared_kv_indices.shape[0] + unique_kv_indices.shape[0]
        kv_indices_buffer = torch.empty(total_kv_pages, device="cuda", dtype=torch.int32)

        qo_indptr_shared = (
            torch.arange(num_prefixes + 1, device="cuda", dtype=torch.int32)
            * (suffixes_per_prefix * qo_len)
        )
        qo_indptr_unique = (
            torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * qo_len
        )

        cascade = CascadeBatchAttention(
            num_levels=2, kv_layout="NHD", device="cuda",
            use_cuda_graph=True, kv_indices_buffer=kv_indices_buffer,
        )
        print(f"\n--- Fused Cascade plan (shared_kv_len={skv_len}) ---")
        print(f"  (C++ diagnostics on stderr)")
        cascade.plan(
            [qo_indptr_shared, qo_indptr_unique],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_kv_len_tensor, unique_kv_len_tensor],
            num_qo_heads, num_kv_heads, head_dim, head_dim,
            page_size,
            causal=True,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        print(f"  Fused: cooperative kernel → 1 CTA/SM (vs MultiLevel's potential 2 CTAs/SM)")

        out = torch.empty_like(q)
        lse = torch.empty(q.shape[0], q.shape[1], device="cuda", dtype=torch.float32)

        for _ in range(warmup):
            cascade.run(q, kv_data, out=out, lse=lse)
        torch.cuda.synchronize()

        cascade_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cascade_graph):
            cascade.run(q, kv_data, out=out, lse=lse)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_events[i].record()
            cascade_graph.replay()
            end_events[i].record()
        torch.cuda.synchronize()
        cascade_graph_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        cascade_graph_median = sorted(cascade_graph_times)[len(cascade_graph_times) // 2]

        results.append((skv_len, flat_graph_median, ref_graph_median, cascade_graph_median))

    # Print results table
    print(f"\n{'='*90}")
    print(f"CUDA Graph Benchmark — Speedup vs Shared Prefix Length")
    print(f"  num_prefixes={num_prefixes}, suffixes_per_prefix={suffixes_per_prefix}"
          f", total_batch={total_batch}")
    print(f"  unique_kv_len={unique_kv_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"{'='*90}")
    print(f"  {'shared_kv_len':>13}  {'Flat (ms)':>10}  {'MultiLevel (ms)':>15}  {'Fused (ms)':>10}  {'vs Multi':>8}  {'vs Flat':>8}")
    print(f"  {'-'*13}  {'-'*10}  {'-'*15}  {'-'*10}  {'-'*8}  {'-'*8}")
    for skv_len, flat_ms, multi_ms, fused_ms in results:
        print(f"  {skv_len:>13}  {flat_ms:>10.4f}  {multi_ms:>15.4f}  {fused_ms:>10.4f}  {multi_ms / fused_ms:>7.2f}x  {flat_ms / fused_ms:>7.2f}x")
    print(f"{'='*90}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="Number of shared prefixes")
    args = parser.parse_args()
    benchmark_cuda_graph(num_prefixes=args.n)

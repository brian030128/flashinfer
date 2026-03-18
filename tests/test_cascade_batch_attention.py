"""
Test CascadeBatchAttention (fused persistent kernel) against
MultiLevelCascadeAttentionWrapper (multi-kernel baseline).

Setup: shared prefix (1024 tokens) + unique suffixes (8 tokens each, 16 sequences).
"""

import time

import torch

import flashinfer
from flashinfer.attention import CascadeBatchAttention


def ceil_div(a, b):
    return (a + b - 1) // b


def build_cascade_kv_cache(
    shared_kv_len,
    unique_kv_len,
    batch_size,
    num_heads,
    head_dim,
    page_size,
    kv_layout="NHD",
    dtype=torch.float16,
):
    """Build a 2-level paged KV cache with shared prefix + unique suffixes.

    Returns:
        kv_data: paged KV cache tensor
        shared_kv_indices, shared_kv_indptr, shared_last_page_len
        unique_kv_indices, unique_kv_indptr, unique_last_page_len
        shared_kv_len_tensor, unique_kv_len_tensor  (per-request token counts)
    """
    num_shared_pages = ceil_div(shared_kv_len, page_size)
    num_unique_pages_per_seq = ceil_div(unique_kv_len, page_size)
    total_pages = num_shared_pages + batch_size * num_unique_pages_per_seq

    kv_data = torch.zeros(
        total_pages, 2, page_size, num_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    # Shared prefix: pages [0, num_shared_pages)
    k_shared = torch.randn(shared_kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
    v_shared = torch.randn(shared_kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
    shared_kv_indices = torch.arange(num_shared_pages, device="cuda", dtype=torch.int32)
    shared_kv_indptr = torch.tensor([0, num_shared_pages], device="cuda", dtype=torch.int32)
    shared_last_page_len = torch.tensor(
        [(shared_kv_len - 1) % page_size + 1], device="cuda", dtype=torch.int32
    )
    shared_append_indptr = torch.tensor([0, shared_kv_len], device="cuda", dtype=torch.int32)
    flashinfer.append_paged_kv_cache(
        k_shared, v_shared,
        *flashinfer.get_batch_indices_positions(
            shared_append_indptr,
            flashinfer.get_seq_lens(shared_kv_indptr, shared_last_page_len, page_size),
            shared_kv_len,
        ),
        kv_data, shared_kv_indices, shared_kv_indptr, shared_last_page_len, kv_layout,
    )

    # Unique suffixes: pages [num_shared_pages, ...)
    k_unique = torch.randn(batch_size * unique_kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
    v_unique = torch.randn(batch_size * unique_kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
    unique_kv_indices = (
        torch.arange(batch_size * num_unique_pages_per_seq, device="cuda", dtype=torch.int32)
        + num_shared_pages
    )
    unique_kv_indptr = (
        torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * num_unique_pages_per_seq
    )
    unique_last_page_len = torch.full(
        (batch_size,), (unique_kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
    )
    unique_append_indptr = (
        torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * unique_kv_len
    )
    flashinfer.append_paged_kv_cache(
        k_unique, v_unique,
        *flashinfer.get_batch_indices_positions(
            unique_append_indptr,
            flashinfer.get_seq_lens(unique_kv_indptr, unique_last_page_len, page_size),
            batch_size * unique_kv_len,
        ),
        kv_data, unique_kv_indices, unique_kv_indptr, unique_last_page_len, kv_layout,
    )

    # Token-level lengths (for CascadeBatchAttention)
    shared_kv_len_tensor = torch.full((batch_size,), shared_kv_len, device="cuda", dtype=torch.int32)
    unique_kv_len_tensor = torch.full((batch_size,), unique_kv_len, device="cuda", dtype=torch.int32)

    # For shared level, replicate indptr for all requests (each sees the same pages)
    # shared_kv_indptr for MultiLevel: [0, num_shared_pages] repeated per request
    # but actually MultiLevel uses per-level indptr of shape [batch+1]
    shared_kv_indptr_batch = (
        torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * num_shared_pages
    )
    shared_kv_indices_batch = shared_kv_indices.repeat(batch_size)
    shared_last_page_len_batch = shared_last_page_len.repeat(batch_size)

    return (
        kv_data,
        # Shared level
        shared_kv_indices_batch, shared_kv_indptr_batch, shared_last_page_len_batch,
        shared_kv_len_tensor,
        # Unique level
        unique_kv_indices, unique_kv_indptr, unique_last_page_len,
        unique_kv_len_tensor,
    )


def test_cascade_batch_attention_correctness():
    """Compare CascadeBatchAttention vs MultiLevelCascadeAttentionWrapper."""
    torch.manual_seed(42)

    shared_kv_len = 1024
    unique_kv_len = 8
    batch_size = 16
    num_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1  # decode

    (
        kv_data,
        shared_kv_indices, shared_kv_indptr, shared_last_page_len,
        shared_kv_len_tensor,
        unique_kv_indices, unique_kv_indptr, unique_last_page_len,
        unique_kv_len_tensor,
    ) = build_cascade_kv_cache(
        shared_kv_len, unique_kv_len, batch_size, num_heads, head_dim, page_size
    )

    q = torch.randn(batch_size * qo_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    qo_indptr = (torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * qo_len)

    # --- Reference: MultiLevelCascadeAttentionWrapper ---
    ref_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD"
    )

    # For shared level, qo_indptr_top maps all queries to a single "request"
    qo_indptr_top = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    ref_wrapper.plan(
        [qo_indptr_top, qo_indptr],
        [shared_kv_indptr[:2], unique_kv_indptr],  # shared: single request
        [shared_kv_indices[:shared_kv_indptr[1]], unique_kv_indices],
        [shared_last_page_len[:1], unique_last_page_len],
        num_heads, num_heads, head_dim, page_size,
        causal=True,
    )
    o_ref = ref_wrapper.run(q, kv_data)

    # --- Test: CascadeBatchAttention ---
    cascade = CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda")
    cascade.plan(
        qo_indptr,
        [shared_kv_indptr, unique_kv_indptr],
        [shared_kv_indices, unique_kv_indices],
        [shared_kv_len_tensor, unique_kv_len_tensor],
        num_heads, num_heads, head_dim, head_dim,
        page_size,
        causal=True,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_cascade, lse_cascade = cascade.run(q, kv_data)

    # Compare
    torch.testing.assert_close(o_cascade, o_ref, atol=1e-2, rtol=1e-2)
    print(f"[PASS] Correctness test: max diff = {(o_cascade - o_ref).abs().max().item():.6f}")


def benchmark_cascade(warmup=50, repeat=200):
    """Benchmark CascadeBatchAttention vs MultiLevelCascadeAttentionWrapper."""
    torch.manual_seed(42)

    shared_kv_len = 1024
    unique_kv_len = 8
    batch_size = 16
    num_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1

    (
        kv_data,
        shared_kv_indices, shared_kv_indptr, shared_last_page_len,
        shared_kv_len_tensor,
        unique_kv_indices, unique_kv_indptr, unique_last_page_len,
        unique_kv_len_tensor,
    ) = build_cascade_kv_cache(
        shared_kv_len, unique_kv_len, batch_size, num_heads, head_dim, page_size
    )

    q = torch.randn(batch_size * qo_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    qo_indptr = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * qo_len

    # --- Setup flat paged decode (no cascade, full kv per request) ---
    total_kv_len = shared_kv_len + unique_kv_len
    num_shared_pages = ceil_div(shared_kv_len, page_size)
    num_unique_pages = ceil_div(unique_kv_len, page_size)
    flat_kv_indices_list = []
    for b in range(batch_size):
        flat_kv_indices_list.append(shared_kv_indices[:num_shared_pages])
        flat_kv_indices_list.append(
            unique_kv_indices[b * num_unique_pages : (b + 1) * num_unique_pages]
        )
    flat_kv_indices = torch.cat(flat_kv_indices_list)
    flat_pages_per_req = num_shared_pages + num_unique_pages
    flat_kv_indptr = (
        torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * flat_pages_per_req
    )
    flat_last_page_len = torch.full(
        (batch_size,), (total_kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
    )

    flat_decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    flat_decode.plan(
        flat_kv_indptr, flat_kv_indices, flat_last_page_len,
        num_heads, num_heads, head_dim, page_size,
        data_type=torch.float16,
    )

    # --- Setup reference ---
    ref_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD"
    )
    qo_indptr_top = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    ref_wrapper.plan(
        [qo_indptr_top, qo_indptr],
        [shared_kv_indptr[:2], unique_kv_indptr],
        [shared_kv_indices[:shared_kv_indptr[1]], unique_kv_indices],
        [shared_last_page_len[:1], unique_last_page_len],
        num_heads, num_heads, head_dim, page_size,
        causal=True,
    )

    # --- Setup fused cascade ---
    cascade = CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda")
    cascade.plan(
        qo_indptr,
        [shared_kv_indptr, unique_kv_indptr],
        [shared_kv_indices, unique_kv_indices],
        [shared_kv_len_tensor, unique_kv_len_tensor],
        num_heads, num_heads, head_dim, head_dim,
        page_size,
        causal=True,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )

    # Pre-allocate output buffers
    out_cascade = torch.empty_like(q)
    lse_cascade = torch.empty(q.shape[0], q.shape[1], device="cuda", dtype=torch.float32)

    # --- Benchmark Flat Paged Decode (no cascade) ---
    for _ in range(warmup):
        flat_decode.run(q, kv_data)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_events[i].record()
        flat_decode.run(q, kv_data)
        end_events[i].record()
    torch.cuda.synchronize()
    flat_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    flat_median = sorted(flat_times)[len(flat_times) // 2]

    # --- Benchmark MultiLevel (reference) ---
    for _ in range(warmup):
        ref_wrapper.run(q, kv_data)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_events[i].record()
        ref_wrapper.run(q, kv_data)
        end_events[i].record()
    torch.cuda.synchronize()
    ref_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    ref_median = sorted(ref_times)[len(ref_times) // 2]

    # --- Benchmark Fused Cascade ---
    for _ in range(warmup):
        cascade.run(q, kv_data, out=out_cascade, lse=lse_cascade)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_events[i].record()
        cascade.run(q, kv_data, out=out_cascade, lse=lse_cascade)
        end_events[i].record()
    torch.cuda.synchronize()
    cascade_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    cascade_median = sorted(cascade_times)[len(cascade_times) // 2]

    print(f"\n{'='*60}")
    print(f"Cascade Attention Benchmark")
    print(f"  shared_kv_len={shared_kv_len}, unique_kv_len={unique_kv_len}")
    print(f"  batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"{'='*60}")
    print(f"  Flat Paged Decode (no cascade):  {flat_median:.4f} ms (median)")
    print(f"  MultiLevel (N kernels + merge):  {ref_median:.4f} ms (median)")
    print(f"  Fused Cascade (1 kernel):        {cascade_median:.4f} ms (median)")
    print(f"  Speedup vs MultiLevel:           {ref_median / cascade_median:.2f}x")
    print(f"  Speedup vs Flat Decode:          {flat_median / cascade_median:.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_cascade_batch_attention_correctness()
    benchmark_cascade()

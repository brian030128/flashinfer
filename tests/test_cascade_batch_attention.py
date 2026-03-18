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
    num_shared_pages = ceil_div(shared_kv_len, page_size)
    ref_wrapper.plan(
        [qo_indptr_top, qo_indptr],
        [shared_kv_indptr[:2], unique_kv_indptr],  # shared: single request
        [shared_kv_indices[:num_shared_pages], unique_kv_indices],
        [shared_last_page_len[:1], unique_last_page_len],
        num_heads, num_heads, head_dim, page_size,
        causal=True,
    )
    o_ref = ref_wrapper.run(q, kv_data)

    # --- Test: CascadeBatchAttention ---
    # Per-level qo_indptr: shared level packs all queries as 1 request,
    # unique level has per-request qo_indptr
    qo_indptr_shared = torch.tensor([0, batch_size * qo_len], device="cuda", dtype=torch.int32)
    qo_indptr_unique = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * qo_len

    cascade = CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda")
    cascade.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [shared_kv_indptr[:2], unique_kv_indptr],
        [shared_kv_indices[:num_shared_pages], unique_kv_indices],
        [shared_kv_len_tensor[:1], unique_kv_len_tensor],
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
        [shared_kv_indices[:num_shared_pages], unique_kv_indices],
        [shared_last_page_len[:1], unique_last_page_len],
        num_heads, num_heads, head_dim, page_size,
        causal=True,
    )

    # --- Setup fused cascade ---
    qo_indptr_shared = torch.tensor([0, batch_size * qo_len], device="cuda", dtype=torch.int32)
    qo_indptr_unique = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * qo_len

    cascade = CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda")
    cascade.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [shared_kv_indptr[:2], unique_kv_indptr],
        [shared_kv_indices[:num_shared_pages], unique_kv_indices],
        [shared_kv_len_tensor[:1], unique_kv_len_tensor],
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


def test_cascade_batch_attention_cuda_graph():
    """Test CascadeBatchAttention with CUDA graph capture and replay."""
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
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], device="cuda", dtype=torch.float32)

    num_shared_pages = ceil_div(shared_kv_len, page_size)

    # Pre-allocate kv_indices buffer large enough for all levels
    total_kv_pages = num_shared_pages + unique_kv_indices.shape[0]
    kv_indices_buffer = torch.empty(total_kv_pages, device="cuda", dtype=torch.int32)

    # Per-level qo_indptr: both levels use same per-request indptr for this test
    qo_indptr_shared = torch.tensor([0, batch_size * qo_len], device="cuda", dtype=torch.int32)
    qo_indptr_unique = qo_indptr

    cascade = CascadeBatchAttention(
        num_levels=2, kv_layout="NHD", device="cuda",
        use_cuda_graph=True, kv_indices_buffer=kv_indices_buffer,
    )
    cascade.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [shared_kv_indptr[:2], unique_kv_indptr],
        [shared_kv_indices[:num_shared_pages], unique_kv_indices],
        [shared_kv_len_tensor[:1], unique_kv_len_tensor],
        num_heads, num_heads, head_dim, head_dim,
        page_size, causal=False,
        q_data_type=torch.float16, kv_data_type=torch.float16,
    )

    # Warmup
    cascade.run(q, kv_data, out=out, lse=lse)
    torch.cuda.synchronize()

    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cascade.run(q, kv_data, out=out, lse=lse)

    # Replay
    graph.replay()
    torch.cuda.synchronize()
    o_graph = out.clone()

    # Compare against non-graph run
    cascade_no_graph = CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda")
    cascade_no_graph.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [shared_kv_indptr[:2], unique_kv_indptr],
        [shared_kv_indices[:num_shared_pages], unique_kv_indices],
        [shared_kv_len_tensor[:1], unique_kv_len_tensor],
        num_heads, num_heads, head_dim, head_dim,
        page_size, causal=False,
        q_data_type=torch.float16, kv_data_type=torch.float16,
    )
    o_ref, _ = cascade_no_graph.run(q, kv_data)

    torch.testing.assert_close(o_graph, o_ref, atol=1e-3, rtol=1e-3)
    print(f"[PASS] CUDA graph test: max diff = {(o_graph - o_ref).abs().max().item():.6f}")

    # Replay again to verify stability
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, o_ref, atol=1e-3, rtol=1e-3)
    print(f"[PASS] CUDA graph replay test: max diff = {(out - o_ref).abs().max().item():.6f}")


def compute_attention_metrics(batch_size, qo_len, kv_len, num_heads, head_dim, dtype_bytes, latency_ms):
    """Compute achieved bandwidth, throughput, and arithmetic intensity for attention.

    Returns:
        (achieved_bw_gbps, achieved_tflops, arithmetic_intensity)
    """
    # Bytes: KV read + Q read + O write
    kv_bytes = batch_size * kv_len * num_heads * head_dim * 2 * dtype_bytes  # K + V
    q_bytes = batch_size * qo_len * num_heads * head_dim * dtype_bytes
    o_bytes = batch_size * qo_len * num_heads * head_dim * dtype_bytes
    total_bytes = kv_bytes + q_bytes + o_bytes

    # FLOPs: 2 for QK^T + 2 for attn*V
    total_flops = batch_size * num_heads * qo_len * kv_len * head_dim * 4

    latency_s = latency_ms / 1000.0
    achieved_bw_gbps = total_bytes / latency_s / 1e9
    achieved_tflops = total_flops / latency_s / 1e12
    arithmetic_intensity = total_flops / total_bytes

    return achieved_bw_gbps, achieved_tflops, arithmetic_intensity


def _benchmark_median(fn, warmup=50, repeat=200):
    """Run fn with warmup and return median latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sorted(times)[len(times) // 2]


def benchmark_per_level(warmup=50, repeat=200):
    """Benchmark each cascade level in isolation to measure utilization."""
    torch.manual_seed(42)

    batch_size = 16
    num_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1
    dtype = torch.float16
    dtype_bytes = 2

    levels = [
        ("Shared Prefix", 8192),
        ("Unique Suffix", 8),
    ]

    # Get GPU info
    props = torch.cuda.get_device_properties(0)
    device_name = props.name
    # Peak memory bandwidth (GB/s) - approximate from known specs
    # For unknown GPUs, just report achieved numbers
    peak_bw_gbps = None
    name_lower = device_name.lower()
    if "a100" in name_lower:
        peak_bw_gbps = 2039 if "sxm" in name_lower else 1555
    elif "h100" in name_lower:
        peak_bw_gbps = 3352 if "sxm" in name_lower else 2039
    elif "4090" in name_lower:
        peak_bw_gbps = 1008
    elif "3090" in name_lower:
        peak_bw_gbps = 936

    print(f"\nPer-Level Decode Benchmarks (batch={batch_size}, heads={num_heads}, "
          f"head_dim={head_dim}, qo_len={qo_len})")
    print(f"GPU: {device_name}")
    if peak_bw_gbps:
        print(f"Peak memory bandwidth: {peak_bw_gbps} GB/s")
    print("─" * 64)

    for level_name, kv_len in levels:
        # Compute metrics constants
        _, _, arith_intensity = compute_attention_metrics(
            batch_size, qo_len, kv_len, num_heads, head_dim, dtype_bytes, 1.0
        )

        # Build a simple flat paged KV cache for this level
        num_pages_per_seq = ceil_div(kv_len, page_size)
        total_pages = batch_size * num_pages_per_seq
        kv_data = torch.randn(
            total_pages, 2, page_size, num_heads, head_dim,
            device="cuda", dtype=dtype,
        )
        q = torch.randn(batch_size * qo_len, num_heads, head_dim, device="cuda", dtype=dtype)
        qo_indptr = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * qo_len

        kv_indices = torch.arange(total_pages, device="cuda", dtype=torch.int32)
        kv_indptr = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * num_pages_per_seq
        last_page_len = torch.full(
            (batch_size,), (kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
        )

        # --- BatchDecodeWithPagedKVCache ---
        decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
        )
        decode_wrapper.plan(
            kv_indptr, kv_indices, last_page_len,
            num_heads, num_heads, head_dim, page_size,
            data_type=dtype,
        )
        decode_median = _benchmark_median(
            lambda _dw=decode_wrapper, _q=q, _kv=kv_data: _dw.run(_q, _kv), warmup, repeat
        )
        decode_bw, decode_tflops, _ = compute_attention_metrics(
            batch_size, qo_len, kv_len, num_heads, head_dim, dtype_bytes, decode_median
        )

        # --- Fused Cascade (2-level, with trivial second level) ---
        # CascadeBatchAttention requires num_levels >= 2, so add a dummy level
        # with 1 KV token per request (negligible overhead).
        dummy_pages_per_seq = 1
        dummy_total_pages = batch_size * dummy_pages_per_seq
        dummy_start_page = total_pages  # after real pages
        kv_data_with_dummy = torch.randn(
            total_pages + dummy_total_pages, 2, page_size, num_heads, head_dim,
            device="cuda", dtype=dtype,
        )
        # Copy real data into the combined buffer
        kv_data_with_dummy[:total_pages] = kv_data
        dummy_kv_indices = torch.arange(
            dummy_start_page, dummy_start_page + dummy_total_pages,
            device="cuda", dtype=torch.int32,
        )
        dummy_kv_indptr = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * dummy_pages_per_seq
        dummy_kv_len_tensor = torch.ones(batch_size, device="cuda", dtype=torch.int32)

        kv_len_tensor = torch.full((batch_size,), kv_len, device="cuda", dtype=torch.int32)
        cascade = CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda")
        cascade.plan(
            [qo_indptr, qo_indptr],  # same qo_indptr for both levels
            [kv_indptr, dummy_kv_indptr],
            [kv_indices, dummy_kv_indices],
            [kv_len_tensor, dummy_kv_len_tensor],
            num_heads, num_heads, head_dim, head_dim,
            page_size,
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        out_cascade = torch.empty_like(q)
        lse_cascade = torch.empty(q.shape[0], q.shape[1], device="cuda", dtype=torch.float32)
        cascade_median = _benchmark_median(
            lambda _c=cascade, _q=q, _kv=kv_data_with_dummy, _o=out_cascade, _l=lse_cascade: _c.run(_q, _kv, out=_o, lse=_l),
            warmup, repeat,
        )
        cascade_bw, cascade_tflops, _ = compute_attention_metrics(
            batch_size, qo_len, kv_len, num_heads, head_dim, dtype_bytes, cascade_median
        )

        # Print results
        print(f"\n{level_name} (kv_len={kv_len}):")
        print(f"  Arithmetic intensity:         {arith_intensity:.2f} FLOPs/byte")

        def _fmt(name, median, bw, tflops):
            bw_str = f"{bw:.2f} GB/s"
            if peak_bw_gbps:
                bw_str += f" ({bw / peak_bw_gbps * 100:.0f}% peak)"
            print(f"  {name:30s} {median:.4f} ms | {bw_str} | {tflops:.2f} TFLOPS")

        _fmt("BatchDecode:", decode_median, decode_bw, decode_tflops)
        _fmt("Fused Cascade (2-level*):", cascade_median, cascade_bw, cascade_tflops)

    print("─" * 64)


def benchmark_diagnosis(warmup=50, repeat=200):
    """Diagnose fused cascade slowdown: redundant KV reads + SM utilization."""
    torch.manual_seed(42)

    batch_size = 16
    num_heads = 8
    head_dim = 128
    page_size = 16
    dtype = torch.float16
    dtype_bytes = 2
    shared_kv_len = 8192
    unique_kv_len = 8

    props = torch.cuda.get_device_properties(0)
    num_sms = props.multi_processor_count

    print(f"\nFused Cascade Slowdown Diagnosis (batch={batch_size}, heads={num_heads}, head_dim={head_dim})")
    print(f"GPU: {props.name} ({num_sms} SMs)")
    print("=" * 64)

    # ── 1. Shared Prefix Level ──────────────────────────────────────

    # Build paged KV for shared prefix
    num_shared_pages = ceil_div(shared_kv_len, page_size)
    kv_data_shared = torch.randn(
        num_shared_pages, 2, page_size, num_heads, head_dim,
        device="cuda", dtype=dtype,
    )
    shared_kv_indices = torch.arange(num_shared_pages, device="cuda", dtype=torch.int32)

    # Shared-as-MultiLevel: batch=1, qo_len=batch_size (prefill-style)
    q_all = torch.randn(batch_size, num_heads, head_dim, device="cuda", dtype=dtype)
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    prefill_qo_indptr = torch.tensor([0, batch_size], device="cuda", dtype=torch.int32)
    prefill_kv_indptr = torch.tensor([0, num_shared_pages], device="cuda", dtype=torch.int32)
    prefill_last_page_len = torch.tensor(
        [(shared_kv_len - 1) % page_size + 1], device="cuda", dtype=torch.int32
    )
    prefill_wrapper.plan(
        prefill_qo_indptr, prefill_kv_indptr, shared_kv_indices, prefill_last_page_len,
        num_heads, num_heads, head_dim, page_size,
        causal=False,
    )
    prefill_median = _benchmark_median(
        lambda: prefill_wrapper.run(q_all, kv_data_shared), warmup, repeat
    )
    prefill_bw, _, _ = compute_attention_metrics(
        1, batch_size, shared_kv_len, num_heads, head_dim, dtype_bytes, prefill_median
    )

    # Shared-as-Fused: batch=16, qo_len=1 (decode-style, each reads full KV)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    # Each of 16 requests points to the same shared pages
    decode_kv_indices = shared_kv_indices.repeat(batch_size)
    decode_kv_indptr = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * num_shared_pages
    decode_last_page_len = torch.full(
        (batch_size,), (shared_kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
    )
    decode_wrapper.plan(
        decode_kv_indptr, decode_kv_indices, decode_last_page_len,
        num_heads, num_heads, head_dim, page_size,
        data_type=dtype,
    )
    decode_shared_median = _benchmark_median(
        lambda: decode_wrapper.run(q_all, kv_data_shared), warmup, repeat
    )
    decode_shared_bw, _, _ = compute_attention_metrics(
        batch_size, 1, shared_kv_len, num_heads, head_dim, dtype_bytes, decode_shared_median
    )

    # Data sizes
    kv_bytes_once = 1 * shared_kv_len * num_heads * head_dim * 2 * dtype_bytes
    kv_bytes_16x = batch_size * shared_kv_len * num_heads * head_dim * 2 * dtype_bytes

    print(f"\n1. Shared Prefix Level (kv_len={shared_kv_len})")
    print(f"   MultiLevel approach:  batch=1,  qo_len={batch_size} -> reads KV once  ({kv_bytes_once / 1e6:.0f} MB)")
    print(f"   Fused approach:       batch={batch_size}, qo_len=1  -> reads KV {batch_size}x   ({kv_bytes_16x / 1e6:.0f} MB)")
    print(f"   Data amplification:   {batch_size}x")
    print()
    print(f"   BatchPrefill (batch=1, qo={batch_size}):    {prefill_median:.4f} ms | {prefill_bw:.2f} GB/s")
    print(f"   BatchDecode  (batch={batch_size}, qo=1):    {decode_shared_median:.4f} ms | {decode_shared_bw:.2f} GB/s")

    # ── 2. Unique Suffix Level ──────────────────────────────────────

    num_unique_pages = ceil_div(unique_kv_len, page_size)
    kv_data_unique = torch.randn(
        batch_size * num_unique_pages, 2, page_size, num_heads, head_dim,
        device="cuda", dtype=dtype,
    )
    unique_kv_indices = torch.arange(batch_size * num_unique_pages, device="cuda", dtype=torch.int32)
    unique_kv_indptr = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * num_unique_pages
    unique_last_page_len = torch.full(
        (batch_size,), (unique_kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
    )

    decode_unique = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    decode_unique.plan(
        unique_kv_indptr, unique_kv_indices, unique_last_page_len,
        num_heads, num_heads, head_dim, page_size,
        data_type=dtype,
    )
    unique_median = _benchmark_median(
        lambda: decode_unique.run(q_all, kv_data_unique), warmup, repeat
    )
    unique_bw, _, _ = compute_attention_metrics(
        batch_size, 1, unique_kv_len, num_heads, head_dim, dtype_bytes, unique_median
    )

    print(f"\n2. Unique Suffix Level (kv_len={unique_kv_len})")
    print(f"   BatchDecode  (batch={batch_size}, qo=1):    {unique_median:.4f} ms | {unique_bw:.2f} GB/s")

    # ── 3. SM Utilization ───────────────────────────────────────────

    # Runner2 (CTA_TILE_Q=16, NUM_MMA_KV=2): approximate chunk size
    # chunk_size ≈ NUM_MMA_KV * page_size = 2 * 16 = 32 tokens
    num_mma_kv_runner2 = 2
    chunk_size_approx = num_mma_kv_runner2 * page_size

    shared_work_items = batch_size * num_heads * ceil_div(shared_kv_len, chunk_size_approx)
    unique_work_items = batch_size * num_heads * ceil_div(unique_kv_len, chunk_size_approx)

    print(f"\n3. SM Utilization (fused kernel)")
    print(f"   Total SMs:       {num_sms}")
    print(f"   Shared level:    {shared_work_items} work items -> {shared_work_items / num_sms:.1f} items/SM")
    print(f"   Unique level:    {unique_work_items} work items   -> {unique_work_items / num_sms:.1f} items/SM")

    # ── 4. Time Breakdown ───────────────────────────────────────────

    merge_overhead_est = 0.01  # ~10us estimate for merge kernel
    multilevel_est = prefill_median + unique_median + merge_overhead_est
    fused_est = decode_shared_median + unique_median  # + sync + reduction (unknown)

    print(f"\n4. Time Breakdown")
    print(f"   MultiLevel:  shared_prefill + unique_decode + merge ~= "
          f"{prefill_median:.4f} + {unique_median:.4f} + ~{merge_overhead_est:.2f} = ~{multilevel_est:.4f} ms")
    print(f"   Fused:       shared_decode_x{batch_size} + unique_decode + sync+reduce ~= "
          f"{decode_shared_median:.4f} + {unique_median:.4f} + ? = {fused_est:.4f} ms (lower bound)")
    print(f"   Gap explained by: {batch_size}x redundant KV reads on shared prefix")
    print("=" * 64)


if __name__ == "__main__":
    test_cascade_batch_attention_correctness()
    test_cascade_batch_attention_cuda_graph()
    benchmark_cascade()
    benchmark_diagnosis()

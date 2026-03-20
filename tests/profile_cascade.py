"""
Profile script for ncu: runs Fused and MultiLevel each once (no CUDA graph).
Usage:
    ncu --set full -o profile_fused  python tests/profile_cascade.py --mode fused --kv 16384 --n 1
    ncu --set full -o profile_multi  python tests/profile_cascade.py --mode multi --kv 16384 --n 1
Or just run directly for a quick sanity check:
    python tests/profile_cascade.py --mode both --kv 16384 --n 1
"""
import argparse
import torch
import flashinfer
from flashinfer.attention import CascadeBatchAttention


def ceil_div(a, b):
    return (a + b - 1) // b


def setup(shared_kv_len, num_prefixes):
    torch.manual_seed(42)

    unique_kv_len = 8
    suffixes_per_prefix = 16
    total_batch = num_prefixes * suffixes_per_prefix
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1
    dtype = torch.bfloat16

    num_shared_pages_per_prefix = ceil_div(shared_kv_len, page_size)
    num_unique_pages_per_seq = ceil_div(unique_kv_len, page_size)
    total_shared_pages = num_prefixes * num_shared_pages_per_prefix
    total_unique_pages = total_batch * num_unique_pages_per_seq
    total_pages = total_shared_pages + total_unique_pages

    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    # Shared level metadata
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

    # Unique level metadata
    unique_kv_indices = (
        torch.arange(total_unique_pages, device="cuda", dtype=torch.int32) + total_shared_pages
    )
    unique_kv_indptr = (
        torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * num_unique_pages_per_seq
    )
    unique_last_page_len = torch.full(
        (total_batch,), (unique_kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
    )
    unique_kv_len_tensor = torch.full(
        (total_batch,), unique_kv_len, device="cuda", dtype=torch.int32
    )

    q = torch.randn(total_batch * qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    return dict(
        kv_data=kv_data, q=q,
        shared_kv_indices=shared_kv_indices, shared_kv_indptr=shared_kv_indptr,
        shared_last_page_len=shared_last_page_len, shared_kv_len_tensor=shared_kv_len_tensor,
        unique_kv_indices=unique_kv_indices, unique_kv_indptr=unique_kv_indptr,
        unique_last_page_len=unique_last_page_len, unique_kv_len_tensor=unique_kv_len_tensor,
        total_batch=total_batch, num_prefixes=num_prefixes,
        suffixes_per_prefix=suffixes_per_prefix,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=page_size, qo_len=qo_len, dtype=dtype,
    )


def run_multi(d):
    ref = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD"
    )
    qo_indptr_shared = (
        torch.arange(d["num_prefixes"] + 1, device="cuda", dtype=torch.int32)
        * (d["suffixes_per_prefix"] * d["qo_len"])
    )
    qo_indptr_unique = (
        torch.arange(d["total_batch"] + 1, device="cuda", dtype=torch.int32) * d["qo_len"]
    )
    ref.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [d["shared_kv_indptr"], d["unique_kv_indptr"]],
        [d["shared_kv_indices"], d["unique_kv_indices"]],
        [d["shared_last_page_len"], d["unique_last_page_len"]],
        d["num_qo_heads"], d["num_kv_heads"], d["head_dim"], d["page_size"],
        causal=True, q_data_type=d["dtype"],
    )
    # Warmup
    for _ in range(3):
        ref.run(d["q"], d["kv_data"])
    torch.cuda.synchronize()
    # Profiled run
    ref.run(d["q"], d["kv_data"])
    torch.cuda.synchronize()
    print("MultiLevel done")


def run_fused(d):
    total_kv_pages = d["shared_kv_indices"].shape[0] + d["unique_kv_indices"].shape[0]
    kv_indices_buffer = torch.empty(total_kv_pages, device="cuda", dtype=torch.int32)
    qo_indptr_shared = (
        torch.arange(d["num_prefixes"] + 1, device="cuda", dtype=torch.int32)
        * (d["suffixes_per_prefix"] * d["qo_len"])
    )
    qo_indptr_unique = (
        torch.arange(d["total_batch"] + 1, device="cuda", dtype=torch.int32) * d["qo_len"]
    )
    cascade = CascadeBatchAttention(
        num_levels=2, kv_layout="NHD", device="cuda",
        use_cuda_graph=True, kv_indices_buffer=kv_indices_buffer,
    )
    cascade.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [d["shared_kv_indptr"], d["unique_kv_indptr"]],
        [d["shared_kv_indices"], d["unique_kv_indices"]],
        [d["shared_kv_len_tensor"], d["unique_kv_len_tensor"]],
        d["num_qo_heads"], d["num_kv_heads"], d["head_dim"], d["head_dim"],
        d["page_size"], causal=True,
        q_data_type=d["dtype"], kv_data_type=d["dtype"],
    )
    out = torch.empty_like(d["q"])
    lse = torch.empty(d["q"].shape[0], d["q"].shape[1], device="cuda", dtype=torch.float32)
    # Warmup
    for _ in range(3):
        cascade.run(d["q"], d["kv_data"], out=out, lse=lse)
    torch.cuda.synchronize()
    # Profiled run
    cascade.run(d["q"], d["kv_data"], out=out, lse=lse)
    torch.cuda.synchronize()
    print("Fused done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fused", "multi", "both"], default="both")
    parser.add_argument("--kv", type=int, default=16384)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    d = setup(args.kv, args.n)

    if args.mode in ("multi", "both"):
        run_multi(d)
    if args.mode in ("fused", "both"):
        run_fused(d)

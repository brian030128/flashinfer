"""
Correctness test: CascadeBatchAttentionWrapper vs MultiLevelCascadeAttentionWrapper.

Verifies that the fused cascade kernel produces the same output (within tolerance)
as the reference multi-level cascade implementation.

Usage:
    pytest tests/test_cascade_correctness.py -v
    python tests/test_cascade_correctness.py
"""

import pytest
import torch

import flashinfer
from flashinfer.attention import CascadeBatchAttentionWrapper


def ceil_div(a, b):
    return (a + b - 1) // b


def build_cascade_kv_cache(
    shared_kv_len,
    unique_kv_len,
    num_prefixes,
    suffixes_per_prefix,
    num_kv_heads,
    head_dim,
    page_size,
    dtype=torch.bfloat16,
):
    """Build a 2-level paged KV cache and return all metadata."""
    total_batch = num_prefixes * suffixes_per_prefix
    num_shared_pages_per_prefix = ceil_div(shared_kv_len, page_size)
    num_unique_pages_per_seq = ceil_div(unique_kv_len, page_size)
    total_shared_pages = num_prefixes * num_shared_pages_per_prefix
    total_unique_pages = total_batch * num_unique_pages_per_seq
    total_pages = total_shared_pages + total_unique_pages

    kv_data = torch.zeros(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    # Fill shared prefix pages
    for p in range(num_prefixes):
        k_shared = torch.randn(shared_kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v_shared = torch.randn(shared_kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
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
            kv_data, kv_idx, kv_indptr, last_page_len, "NHD",
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

    # Unique suffix pages
    k_unique = torch.randn(total_batch * unique_kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_unique = torch.randn(total_batch * unique_kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    unique_kv_indices = (
        torch.arange(total_unique_pages, device="cuda", dtype=torch.int32) + total_shared_pages
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
        kv_data, unique_kv_indices, unique_kv_indptr, unique_last_page_len, "NHD",
    )

    unique_kv_len_tensor = torch.full(
        (total_batch,), unique_kv_len, device="cuda", dtype=torch.int32
    )

    return dict(
        kv_data=kv_data,
        shared_kv_indices=shared_kv_indices,
        shared_kv_indptr=shared_kv_indptr,
        shared_last_page_len=shared_last_page_len,
        shared_kv_len_tensor=shared_kv_len_tensor,
        unique_kv_indices=unique_kv_indices,
        unique_kv_indptr=unique_kv_indptr,
        unique_last_page_len=unique_last_page_len,
        unique_kv_len_tensor=unique_kv_len_tensor,
    )


def run_multilevel_reference(q, kv_data, d, num_prefixes, suffixes_per_prefix, qo_len,
                             num_qo_heads, num_kv_heads, head_dim, page_size, dtype):
    """Run MultiLevelCascadeAttentionWrapper as reference."""
    ref_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD"
    )
    total_batch = num_prefixes * suffixes_per_prefix
    qo_indptr_shared = (
        torch.arange(num_prefixes + 1, device="cuda", dtype=torch.int32)
        * (suffixes_per_prefix * qo_len)
    )
    qo_indptr_unique = (
        torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    ref_wrapper.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [d["shared_kv_indptr"], d["unique_kv_indptr"]],
        [d["shared_kv_indices"], d["unique_kv_indices"]],
        [d["shared_last_page_len"], d["unique_last_page_len"]],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True, q_data_type=dtype,
    )
    # MultiLevelCascadeAttentionWrapper.run returns just output tensor
    return ref_wrapper.run(q, kv_data)


def run_fused_cascade(q, kv_data, d, num_prefixes, suffixes_per_prefix, qo_len,
                      num_qo_heads, num_kv_heads, head_dim, page_size, dtype):
    """Run CascadeBatchAttentionWrapper (fused)."""
    total_batch = num_prefixes * suffixes_per_prefix
    cascade = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda",
    )
    qo_indptr_shared = (
        torch.arange(num_prefixes + 1, device="cuda", dtype=torch.int32)
        * (suffixes_per_prefix * qo_len)
    )
    qo_indptr_unique = (
        torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    cascade.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [d["shared_kv_indptr"], d["unique_kv_indptr"]],
        [d["shared_kv_indices"], d["unique_kv_indices"]],
        [d["shared_kv_len_tensor"], d["unique_kv_len_tensor"]],
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size, causal=True,
        q_data_type=dtype, kv_data_type=dtype,
    )
    out, lse = cascade.run(q, kv_data)
    return out


@pytest.mark.parametrize("shared_kv_len", [128, 512, 2048, 8192])
@pytest.mark.parametrize("num_prefixes", [1, 4])
@pytest.mark.parametrize("unique_kv_len", [5, 32])
def test_cascade_correctness(shared_kv_len, num_prefixes, unique_kv_len):
    torch.manual_seed(42)
    suffixes_per_prefix = 16
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1
    dtype = torch.bfloat16
    total_batch = num_prefixes * suffixes_per_prefix

    d = build_cascade_kv_cache(
        shared_kv_len, unique_kv_len, num_prefixes, suffixes_per_prefix,
        num_kv_heads, head_dim, page_size, dtype=dtype,
    )

    q = torch.randn(total_batch * qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    ref_out = run_multilevel_reference(
        q, d["kv_data"], d, num_prefixes, suffixes_per_prefix, qo_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
    )
    fused_out = run_fused_cascade(
        q, d["kv_data"], d, num_prefixes, suffixes_per_prefix, qo_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
    )

    # bf16 attention has numerical error; use generous tolerance
    torch.testing.assert_close(fused_out, ref_out, atol=1e-2, rtol=1e-2)


def run_single_test(shared_kv_len, num_prefixes, unique_kv_len):
    """Run a single test with debug output."""
    torch.manual_seed(42)
    suffixes_per_prefix = 16
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1
    dtype = torch.bfloat16
    total_batch = num_prefixes * suffixes_per_prefix

    d = build_cascade_kv_cache(
        shared_kv_len, unique_kv_len, num_prefixes, suffixes_per_prefix,
        num_kv_heads, head_dim, page_size, dtype=dtype,
    )

    q = torch.randn(total_batch * qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    ref_out = run_multilevel_reference(
        q, d["kv_data"], d, num_prefixes, suffixes_per_prefix, qo_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
    )
    fused_out = run_fused_cascade(
        q, d["kv_data"], d, num_prefixes, suffixes_per_prefix, qo_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
    )

    abs_diff = (fused_out.float() - ref_out.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    close_frac = (abs_diff < 0.01).float().mean().item()
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
          f"frac_close(<0.01)={close_frac:.4f}")
    print(f"  ref_out  range: [{ref_out.min().item():.4f}, {ref_out.max().item():.4f}]")
    print(f"  fused_out range: [{fused_out.min().item():.4f}, {fused_out.max().item():.4f}]")
    # Check a few elements
    for i in range(min(3, total_batch)):
        print(f"  batch[{i}] ref[0,:3]  = {ref_out[i, 0, :3].tolist()}")
        print(f"  batch[{i}] fused[0,:3]= {fused_out[i, 0, :3].tolist()}")
    return max_diff


if __name__ == "__main__":
    configs = [
        (256, 1, 5),
        (256, 1, 32),
        (2048, 1, 5),
        (2048, 4, 5),
        (8192, 1, 5),
    ]
    all_pass = True
    for skv, npfx, ukv in configs:
        print(f"Testing shared_kv_len={skv}, num_prefixes={npfx}, unique_kv_len={ukv}:")
        max_diff = run_single_test(skv, npfx, ukv)
        passed = max_diff < 0.02
        print(f"  {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False
    print(f"\n{'All correctness tests passed!' if all_pass else 'SOME TESTS FAILED'}")

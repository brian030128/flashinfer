"""
Copyright (c) 2026 by FlashInfer team.

Tests for FusedMultiLevelCascadeAttentionWrapper.

Compares the fused single-launch implementation against the existing
MultiLevelCascadeAttentionWrapper (which launches one prefill kernel per
level + merge_state_in_place between levels).
"""

from typing import List

import pytest
import torch

import flashinfer


def ceil_div(a, b):
    return (a + b - 1) // b


def _build_paged_kv(num_pages, page_size, num_kv_heads, head_dim, device):
    """Allocates a unified [num_pages, 2, page_size, num_kv_heads, head_dim] kv buffer."""
    return torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )


def _setup_two_level_inputs(
    batch_size: int,
    shared_kv_pages: int,
    unique_kv_pages_per_req: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    device,
):
    """Produces (q, kv_data, plan_args) for a 2-level shared-prefix cascade.

    Level 0: a single "request" of size `batch_size` over `shared_kv_pages`.
    Level 1: `batch_size` per-row requests, each over `unique_kv_pages_per_req`.
    """
    total_unique_pages = batch_size * unique_kv_pages_per_req
    total_pages = shared_kv_pages + total_unique_pages
    kv_data = _build_paged_kv(total_pages, page_size, num_kv_heads, head_dim, device)

    shared_kv_indices = torch.arange(0, shared_kv_pages, dtype=torch.int32, device=device)
    shared_kv_indptr = torch.tensor([0, shared_kv_pages], dtype=torch.int32, device=device)
    shared_last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)

    unique_kv_indices = (
        torch.arange(0, total_unique_pages, dtype=torch.int32, device=device)
        + shared_kv_pages
    )
    unique_kv_indptr = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=device
    ) * unique_kv_pages_per_req
    unique_last_page_len = torch.full(
        (batch_size,), page_size, dtype=torch.int32, device=device
    )

    qo_indptr_top = torch.tensor([0, batch_size], dtype=torch.int32, device=device)
    qo_indptr_bottom = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)

    return {
        "qo_indptr_arr": [qo_indptr_top, qo_indptr_bottom],
        "kv_indptr_arr": [shared_kv_indptr, unique_kv_indptr],
        "kv_indices_arr": [shared_kv_indices, unique_kv_indices],
        "kv_last_page_arr": [shared_last_page_len, unique_last_page_len],
        "kv_data": kv_data,
    }


def _run_baseline(inputs, q, num_qo_heads, num_kv_heads, head_dim, page_size, kv_layout):
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(2, workspace, kv_layout)
    wrapper.plan(
        inputs["qo_indptr_arr"],
        inputs["kv_indptr_arr"],
        inputs["kv_indices_arr"],
        inputs["kv_last_page_arr"],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    return wrapper.run(q, inputs["kv_data"])


def _run_fused(inputs, q, num_qo_heads, num_kv_heads, head_dim, page_size, kv_layout):
    wrapper = flashinfer.FusedMultiLevelCascadeAttentionWrapper(2, kv_layout=kv_layout)
    wrapper.plan(
        inputs["qo_indptr_arr"],
        inputs["kv_indptr_arr"],
        inputs["kv_indices_arr"],
        inputs["kv_last_page_arr"],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    return wrapper.run(q, inputs["kv_data"])


@pytest.mark.parametrize("batch_size", [1, 4, 7])
@pytest.mark.parametrize("shared_kv_pages", [8, 64])
@pytest.mark.parametrize("unique_kv_pages_per_req", [1, 4])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_dim", [128])
def test_fused_cascade_two_level_decode_matches_baseline(
    batch_size,
    shared_kv_pages,
    unique_kv_pages_per_req,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    kv_layout = "NHD"
    torch.manual_seed(0)

    inputs = _setup_two_level_inputs(
        batch_size,
        shared_kv_pages,
        unique_kv_pages_per_req,
        page_size,
        num_kv_heads,
        head_dim,
        device,
    )
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device)

    baseline = _run_baseline(
        inputs, q, num_qo_heads, num_kv_heads, head_dim, page_size, kv_layout
    )
    fused = _run_fused(
        inputs, q, num_qo_heads, num_kv_heads, head_dim, page_size, kv_layout
    )

    # fp16 attention with random KV — relax tolerances vs. the exact match the
    # other cascade tests use, since merge ordering can perturb the last bits.
    torch.testing.assert_close(fused, baseline, rtol=2e-2, atol=2e-2)


def _setup_n_level_inputs(
    num_levels: int,
    batch_size: int,
    pages_per_level: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    device,
):
    """N-level cascade where every level except the last uses a single
    "request" covering all batch rows (shared-prefix style stack)."""
    total_pages = num_levels * pages_per_level if num_levels - 1 == 0 else (
        (num_levels - 1) * pages_per_level + batch_size * pages_per_level
    )
    kv_data = _build_paged_kv(total_pages, page_size, num_kv_heads, head_dim, device)

    qo_indptr_arr = []
    kv_indptr_arr = []
    kv_indices_arr = []
    kv_last_page_arr = []

    page_cursor = 0
    for l in range(num_levels):
        if l < num_levels - 1:
            # Shared-style level: single group over all batch rows.
            qo_indptr_arr.append(
                torch.tensor([0, batch_size], dtype=torch.int32, device=device)
            )
            kv_indices_arr.append(
                torch.arange(
                    page_cursor, page_cursor + pages_per_level, dtype=torch.int32, device=device
                )
            )
            kv_indptr_arr.append(
                torch.tensor([0, pages_per_level], dtype=torch.int32, device=device)
            )
            kv_last_page_arr.append(
                torch.tensor([page_size], dtype=torch.int32, device=device)
            )
            page_cursor += pages_per_level
        else:
            # Final level: per-row request, each with its own pages.
            qo_indptr_arr.append(
                torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
            )
            kv_indices_arr.append(
                torch.arange(
                    page_cursor,
                    page_cursor + batch_size * pages_per_level,
                    dtype=torch.int32,
                    device=device,
                )
            )
            kv_indptr_arr.append(
                torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
                * pages_per_level
            )
            kv_last_page_arr.append(
                torch.full((batch_size,), page_size, dtype=torch.int32, device=device)
            )
            page_cursor += batch_size * pages_per_level

    return {
        "qo_indptr_arr": qo_indptr_arr,
        "kv_indptr_arr": kv_indptr_arr,
        "kv_indices_arr": kv_indices_arr,
        "kv_last_page_arr": kv_last_page_arr,
        "kv_data": kv_data,
    }


def _setup_variable_depth_inputs(
    rows_per_level: List[int],
    pages_per_level: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    device,
):
    """Builds a cascade where deeper levels cover *fewer* rows.

    rows_per_level[0] >= rows_per_level[1] >= ... — the row count must be
    monotone-decreasing because the fused wrapper requires deeper levels to
    cover a contiguous prefix of the root level's rows (the parent project
    sorts deepest-branch queries first in its tree-draft tensor layout).

    Each level uses `pages_per_level` pages per participating row
    (per-row requests, like the "unique" branch level in a 2-level cascade).
    """
    num_levels = len(rows_per_level)
    total_pages = sum(rows_per_level) * pages_per_level
    kv_data = _build_paged_kv(total_pages, page_size, num_kv_heads, head_dim, device)

    qo_indptr_arr = []
    kv_indptr_arr = []
    kv_indices_arr = []
    kv_last_page_arr = []

    page_cursor = 0
    for l in range(num_levels):
        rows_l = rows_per_level[l]
        # Per-row requests at every level so the test exercises a non-trivial
        # tile schedule (rather than degenerating to one big group).
        qo_indptr_arr.append(
            torch.arange(0, rows_l + 1, dtype=torch.int32, device=device)
        )
        kv_indices_arr.append(
            torch.arange(
                page_cursor,
                page_cursor + rows_l * pages_per_level,
                dtype=torch.int32,
                device=device,
            )
        )
        kv_indptr_arr.append(
            torch.arange(0, rows_l + 1, dtype=torch.int32, device=device) * pages_per_level
        )
        kv_last_page_arr.append(
            torch.full((rows_l,), page_size, dtype=torch.int32, device=device)
        )
        page_cursor += rows_l * pages_per_level

    return {
        "qo_indptr_arr": qo_indptr_arr,
        "kv_indptr_arr": kv_indptr_arr,
        "kv_indices_arr": kv_indices_arr,
        "kv_last_page_arr": kv_last_page_arr,
        "kv_data": kv_data,
        "rows_per_level": rows_per_level,
    }


def _reference_variable_depth(inputs, q, num_qo_heads, num_kv_heads, head_dim, page_size):
    """Per-row reference using single-level cascade attention per query row.

    For each row, gather its KV ranges across every level it participates in,
    concatenate, and run a flat attention. Handles variable-depth correctly
    because LSE-merging across levels is mathematically equivalent to just
    concatenating the KV.
    """
    device = q.device
    rows_per_level = inputs["rows_per_level"]
    total_rows = rows_per_level[0]
    out = torch.zeros_like(q)

    kv_data = inputs["kv_data"]  # [num_pages, 2, page_size, num_kv_heads, head_dim]

    # Group-query: replicate Q heads across KV heads.
    gqa = num_qo_heads // num_kv_heads
    sm_scale = 1.0 / (head_dim**0.5)

    for row in range(total_rows):
        # Gather all KV pages this row attends to across every level it's in.
        all_k = []
        all_v = []
        for l, rows_l in enumerate(rows_per_level):
            if row >= rows_l:
                continue  # row doesn't participate at this level
            # Per-row request: kv_indptr_arr[l][row : row+2] gives this row's pages.
            kv_indptr = inputs["kv_indptr_arr"][l]
            kv_indices = inputs["kv_indices_arr"][l]
            last_page_len = inputs["kv_last_page_arr"][l]
            page_start = int(kv_indptr[row].item())
            page_end = int(kv_indptr[row + 1].item())
            for p_idx in range(page_start, page_end):
                page = int(kv_indices[p_idx].item())
                # NHD layout: kv_data[page, k_or_v, page_pos, num_kv_heads, head_dim].
                k_page = kv_data[page, 0]  # [page_size, num_kv_heads, head_dim]
                v_page = kv_data[page, 1]
                # Last page may be partially valid.
                valid_len = (
                    page_size if p_idx < page_end - 1 else int(last_page_len[row].item())
                )
                all_k.append(k_page[:valid_len])
                all_v.append(v_page[:valid_len])
        if not all_k:
            continue
        K = torch.cat(all_k, dim=0)  # [kv_len_total, num_kv_heads, head_dim]
        V = torch.cat(all_v, dim=0)
        # Flat attention for this row.
        q_row = q[row]  # [num_qo_heads, head_dim]
        # GQA expansion.
        K_full = K.repeat_interleave(gqa, dim=1)  # [kv_len, num_qo_heads, head_dim]
        V_full = V.repeat_interleave(gqa, dim=1)
        scores = torch.einsum("hd,nhd->nh", q_row.float(), K_full.float()) * sm_scale
        weights = torch.softmax(scores, dim=0)  # [kv_len, num_qo_heads]
        out[row] = torch.einsum("nh,nhd->hd", weights, V_full.float()).to(q.dtype)

    return out


@pytest.mark.parametrize(
    "rows_per_level",
    [
        [4, 4],         # uniform 2-level
        [4, 2],         # variable: 2 of 4 reach level 1
        [6, 4, 2],      # 3-level: pyramidal
        [8, 4, 2, 1],   # 4-level: aggressive narrowing
    ],
)
@pytest.mark.parametrize("head_dim", [128])
def test_fused_cascade_variable_depth(rows_per_level, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    kv_layout = "NHD"
    torch.manual_seed(0)

    page_size = 16
    pages_per_level = 4
    num_qo_heads = num_kv_heads = 8

    inputs = _setup_variable_depth_inputs(
        rows_per_level, pages_per_level, page_size, num_kv_heads, head_dim, device
    )

    total_rows = rows_per_level[0]
    q = torch.randn(total_rows, num_qo_heads, head_dim, dtype=torch.float16, device=device)

    fused_wrapper = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        len(rows_per_level), kv_layout=kv_layout
    )
    fused_wrapper.plan(
        inputs["qo_indptr_arr"],
        inputs["kv_indptr_arr"],
        inputs["kv_indices_arr"],
        inputs["kv_last_page_arr"],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    fused = fused_wrapper.run(q, inputs["kv_data"])

    reference = _reference_variable_depth(
        inputs, q, num_qo_heads, num_kv_heads, head_dim, page_size
    )

    torch.testing.assert_close(fused, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_levels", [2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("head_dim", [128])
def test_fused_cascade_n_level_matches_baseline(num_levels, batch_size, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    kv_layout = "NHD"
    torch.manual_seed(0)

    page_size = 16
    pages_per_level = 8
    num_qo_heads = num_kv_heads = 8

    inputs = _setup_n_level_inputs(
        num_levels, batch_size, pages_per_level, page_size, num_kv_heads, head_dim, device
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device)

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    baseline_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        num_levels, workspace, kv_layout
    )
    baseline_wrapper.plan(
        inputs["qo_indptr_arr"],
        inputs["kv_indptr_arr"],
        inputs["kv_indices_arr"],
        inputs["kv_last_page_arr"],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    baseline = baseline_wrapper.run(q, inputs["kv_data"])

    fused_wrapper = flashinfer.FusedMultiLevelCascadeAttentionWrapper(num_levels, kv_layout=kv_layout)
    fused_wrapper.plan(
        inputs["qo_indptr_arr"],
        inputs["kv_indptr_arr"],
        inputs["kv_indices_arr"],
        inputs["kv_last_page_arr"],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    fused = fused_wrapper.run(q, inputs["kv_data"])

    torch.testing.assert_close(fused, baseline, rtol=2e-2, atol=2e-2)

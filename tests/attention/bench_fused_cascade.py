"""
Microbenchmark: FusedMultiLevelCascadeAttentionWrapper vs MultiLevelCascadeAttentionWrapper.

Compares wall-clock time per call (one cascade attention forward pass) for the
fused single-launch implementation against the baseline that launches one
prefill kernel per level + a merge_state_in_place between levels.

Usage:
    CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES=0 \
        uv run python tests/attention/bench_fused_cascade.py

Reports median ms/call and speedup (baseline / fused) for each config.
"""

import statistics
from typing import List, Optional

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time


def _build_paged_kv(num_pages, page_size, num_kv_heads, head_dim, device):
    return torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )


def _make_uniform_inputs(
    num_levels: int,
    batch_size: int,
    pages_per_level: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    device,
):
    """Uniform depth: every query participates at every level. Earlier
    levels are shared (one group per level), the last level is per-row."""
    total_pages = (num_levels - 1) * pages_per_level + batch_size * pages_per_level
    kv_data = _build_paged_kv(total_pages, page_size, num_kv_heads, head_dim, device)

    qo_indptr_arr, kv_indptr_arr, kv_indices_arr, kv_last_page_arr = [], [], [], []
    page_cursor = 0
    for l in range(num_levels):
        if l < num_levels - 1:
            # Shared level: one group covering all batch rows.
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


def _make_variable_depth_inputs(
    rows_per_level: List[int],
    pages_per_level: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    device,
):
    """Variable depth: rows_per_level[l] queries participate at level l.
    Per-row requests at every level (worst case for the baseline since every
    level has many small requests)."""
    total_pages = sum(rows_per_level) * pages_per_level
    kv_data = _build_paged_kv(total_pages, page_size, num_kv_heads, head_dim, device)

    qo_indptr_arr, kv_indptr_arr, kv_indices_arr, kv_last_page_arr = [], [], [], []
    page_cursor = 0
    for rows_l in rows_per_level:
        qo_indptr_arr.append(torch.arange(0, rows_l + 1, dtype=torch.int32, device=device))
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
    }


def _bench_call(fn, inputs, label):
    times = bench_gpu_time(
        fn,
        dry_run_iters=5,
        repeat_iters=50,
        cold_l2_cache=False,
    )
    return statistics.median(times)


def _run_config(
    name: str,
    inputs,
    num_levels: int,
    total_qo_rows: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    device,
):
    kv_layout = "NHD"
    q = torch.randn(
        total_qo_rows, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    baseline = flashinfer.MultiLevelCascadeAttentionWrapper(num_levels, workspace, kv_layout)
    baseline.plan(
        inputs["qo_indptr_arr"],
        inputs["kv_indptr_arr"],
        inputs["kv_indices_arr"],
        inputs["kv_last_page_arr"],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )

    fused = flashinfer.FusedMultiLevelCascadeAttentionWrapper(num_levels, kv_layout=kv_layout)
    fused.plan(
        inputs["qo_indptr_arr"],
        inputs["kv_indptr_arr"],
        inputs["kv_indices_arr"],
        inputs["kv_last_page_arr"],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )

    baseline_ms = _bench_call(lambda: baseline.run(q, inputs["kv_data"]), inputs, "baseline")
    fused_ms = _bench_call(lambda: fused.run(q, inputs["kv_data"]), inputs, "fused")
    speedup = baseline_ms / fused_ms
    print(
        f"{name:45s}  baseline={baseline_ms*1000:7.1f} us  "
        f"fused={fused_ms*1000:7.1f} us  speedup={speedup:5.2f}x"
    )
    return speedup


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    num_qo_heads = 8
    num_kv_heads = 8
    head_dim = 128
    page_size = 16

    print("=" * 100)
    print("Uniform-depth cascades (every query participates at every level)")
    print("=" * 100)
    speedups: List[float] = []

    for num_levels in [2, 3, 4]:
        for batch_size in [1, 4, 16, 64]:
            for pages_per_level in [4, 16, 64]:  # ~64 / 256 / 1024 KV tokens per level
                inputs = _make_uniform_inputs(
                    num_levels,
                    batch_size,
                    pages_per_level,
                    page_size,
                    num_kv_heads,
                    head_dim,
                    device,
                )
                name = (
                    f"uniform L={num_levels} B={batch_size:>2d} "
                    f"pages/lvl={pages_per_level:>3d}"
                )
                s = _run_config(
                    name,
                    inputs,
                    num_levels,
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    page_size,
                    device,
                )
                speedups.append(s)

    print()
    print("=" * 100)
    print("Variable-depth cascades (deeper levels cover fewer rows)")
    print("=" * 100)

    variable_topologies = [
        [4, 4],          # uniform 2-level (control)
        [4, 2],          # 2-level: half drop out at level 1
        [8, 4],          # 2-level: half drop out, larger batch
        [16, 4],         # 2-level: aggressive drop-out
        [8, 4, 2],       # 3-level pyramid
        [16, 8, 2],      # 3-level pyramid, wider root
        [8, 4, 2, 1],    # 4-level: deep narrow tip
        [16, 8, 4, 1],   # 4-level: wide root, narrow tip
    ]

    for rows_per_level in variable_topologies:
        for pages_per_level in [4, 16, 64]:
            inputs = _make_variable_depth_inputs(
                rows_per_level,
                pages_per_level,
                page_size,
                num_kv_heads,
                head_dim,
                device,
            )
            name = f"varD rows={rows_per_level}  pages/lvl={pages_per_level:>3d}"
            s = _run_config(
                name,
                inputs,
                len(rows_per_level),
                rows_per_level[0],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                device,
            )
            speedups.append(s)

    print()
    print("=" * 100)
    n_at_least_baseline = sum(1 for s in speedups if s >= 0.95)  # within 5% counts as parity
    n_faster = sum(1 for s in speedups if s > 1.0)
    print(
        f"Summary: {len(speedups)} configs, "
        f"{n_at_least_baseline} at-or-above baseline (>=0.95x), "
        f"{n_faster} strictly faster (>1.00x)."
    )
    print(
        f"Median speedup: {statistics.median(speedups):.2f}x  |  "
        f"Min: {min(speedups):.2f}x  |  Max: {max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()

"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

import os

import jinja2
import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import dtype_map, filename_safe_dtype_map, write_if_different


def get_fused_cascade_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    max_levels: int,
) -> str:
    return (
        f"fused_cascade_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"max_levels_{max_levels}"
    )


def gen_fused_cascade_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    max_levels: int = 4,
) -> JitSpec:
    """JIT spec for the fused multi-level cascade prefill kernel.

    The fused module bakes in:
    - dtypes (q/kv/o/idx) and head_dim_qk/vo
    - MAX_LEVELS template bound (default 4)
    - mask = NON_CAUSAL, pos enc = NONE, no sliding window, no soft cap

    These constraints exist because v1 targets the shared-prefix tree-draft
    use case; relaxing them is a follow-up that mirrors the dispatch fan-out
    in gen_batch_prefill_module.
    """
    uri = get_fused_cascade_uri(
        dtype_q, dtype_kv, dtype_o, dtype_idx, head_dim_qk, head_dim_vo, max_levels
    )

    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    with open(jit_env.FLASHINFER_CSRC_DIR / "fused_cascade_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    kwargs = {
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[dtype_idx],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "max_levels": max_levels,
        # All-false flags match the v1 kernel template instantiation in
        # FusedBatchPrefillMultiLevelDispatched (POS_ENCODING_MODE=NONE etc).
        "variant_name": "DefaultAttention<false, false, false, false>",
        "variant_decl": "// DefaultAttention is included via variants.cuh in the config header.",
    }

    config_inc = config_templ.render(**kwargs)
    write_if_different(gen_directory / "fused_cascade_config.inc", config_inc)

    source_paths = []
    for filename in ["fused_cascade.cu", "fused_cascade_jit_binding.cu"]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)
        source_paths.append(dest_path)

    return gen_jit_spec(uri, source_paths)

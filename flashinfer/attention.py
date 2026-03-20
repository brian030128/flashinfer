"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import math
from typing import List, Optional, Tuple, Union

import torch

from .jit import gen_batch_attention_module
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_kv_layout,
    _unpack_paged_kv_cache,
)


@functools.cache
def get_holistic_attention_module(*args):
    return gen_batch_attention_module(*args).build_and_load()


class BatchAttention:
    def __init__(
        self,
        kv_layout: str = "NHD",
        device: str = "cuda",
    ):
        _check_kv_layout(kv_layout)
        self._kv_layout = kv_layout

        self.float_workspace_buffer = torch.empty(
            384 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.page_locked_int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cpu"),
            pin_memory=True,
        )

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        causal: bool = False,
        sm_scale: float = None,
        logits_soft_cap: Optional[float] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        use_profiler: bool = False,
    ) -> None:
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        self._logits_soft_cap = logits_soft_cap

        # get jit module
        get_module_args = (
            q_data_type,
            kv_data_type,
            q_data_type,
            kv_indptr.dtype,
            head_dim_qk,
            head_dim_vo,
            PosEncodingMode["NONE"].value,
            logits_soft_cap > 0.0,
            use_profiler,  # different compiler path
        )
        self.module = get_holistic_attention_module(*get_module_args)

        qo_indptr_host = qo_indptr.to(torch.device("cpu"), non_blocking=True)
        kv_indptr_host = kv_indptr.to(torch.device("cpu"), non_blocking=True)
        kv_len_arr_host = kv_len_arr.to(torch.device("cpu"), non_blocking=True)
        torch.cuda.synchronize()

        batch_size = kv_len_arr.shape[0]
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._use_profiler = use_profiler

        # No addtional buf allocated for CUDA graph tensor
        # Allocate outside FlashInfer
        self._kv_indices = kv_indices
        self._plan_info = self.module.plan(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self.page_locked_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim_vo,
            causal,
        )

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        logits_soft_cap: float = 0.0,
        profiler_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        if logits_soft_cap > 0.0 and self._logits_soft_cap <= 0.0:
            raise ValueError(
                "logits_soft_cap used in kernel run but not provided in plan(). This will cause template deduction error."
            )

        k_cache, v_cache = _unpack_paged_kv_cache(kv_cache, self._kv_layout)
        if out is None:
            out = torch.empty_like(q)
        if lse is None:
            # lse shape: [batch_size, num_qo_heads]
            lse = torch.empty(
                q.shape[0], q.shape[1], device=q.device, dtype=torch.float32
            )
        head_dim_qk = q.shape[2]
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(head_dim_qk)

        # profiler_buffer is optional
        profiler_args = (profiler_buffer,) if self._use_profiler else ()

        self.module.run(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._kv_indices,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            self._sm_scale,
            logits_soft_cap,
            # ADDITIONAL_FUNC_PARAMS
            # PROFILER_FUNC_PARAMS
            *profiler_args,
        )

        return out, lse


class CascadeBatchAttentionWrapper:
    """Fused multi-level cascade attention using a single persistent kernel.

    All cascade levels are processed in one cooperative kernel launch. The reduction
    runner merges cross-level and cross-chunk partials in a single pass after grid sync.

    Requires num_levels >= 2. All levels share the same Q tensor and qo_indptr.
    Causal masking is typically applied only to the last level.
    """

    def __init__(
        self,
        num_levels: int,
        kv_layout: str = "NHD",
        device: str = "cuda",
        use_cuda_graph: bool = False,
        kv_indices_buffer: Optional[torch.Tensor] = None,
    ):
        assert num_levels >= 2, "CascadeBatchAttentionWrapper requires num_levels >= 2"
        _check_kv_layout(kv_layout)
        self._num_levels = num_levels
        self._kv_layout = kv_layout
        self._use_cuda_graph = use_cuda_graph

        if use_cuda_graph:
            if kv_indices_buffer is None:
                raise ValueError(
                    "kv_indices_buffer must be provided when use_cuda_graph=True"
                )
            self._kv_indices_buf = kv_indices_buffer
        else:
            self._kv_indices_buf = None

        self.float_workspace_buffer = torch.empty(
            384 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.page_locked_int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cpu"),
            pin_memory=True,
        )

    def plan(
        self,
        qo_indptr_arr: List[torch.Tensor],
        kv_indptr_arr: List[torch.Tensor],
        kv_indices_arr: List[torch.Tensor],
        kv_len_arr: List[torch.Tensor],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        causal: bool = False,
        sm_scale: Optional[float] = None,
        logits_soft_cap: Optional[float] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
    ) -> None:
        """Plan cascade attention.

        Parameters
        ----------
        qo_indptr_arr : List[torch.Tensor]
            Per-level QO indptr arrays. Each level can have a different batch size,
            but all must have the same total Q length (last element).
        kv_indptr_arr : List[torch.Tensor]
            Per-level KV page table indptr arrays.
        kv_indices_arr : List[torch.Tensor]
            Per-level KV page indices tensors.
        kv_len_arr : List[torch.Tensor]
            Per-level KV length arrays.
        causal : bool
            If True, causal masking is applied to the LAST level only.
        """
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        self._logits_soft_cap = logits_soft_cap

        get_module_args = (
            q_data_type,
            kv_data_type,
            q_data_type,
            kv_indptr_arr[0].dtype,
            head_dim_qk,
            head_dim_vo,
            PosEncodingMode["NONE"].value,
            logits_soft_cap > 0.0,
            False,  # use_profiler
        )
        self.module = get_holistic_attention_module(*get_module_args)

        # Move to host
        qo_indptr_host_arr = [t.to("cpu", non_blocking=True) for t in qo_indptr_arr]
        kv_indptr_host_arr = [t.to("cpu", non_blocking=True) for t in kv_indptr_arr]
        kv_len_host_arr = [t.to("cpu", non_blocking=True) for t in kv_len_arr]
        torch.cuda.synchronize()

        self._page_size = page_size
        self._sm_scale = sm_scale
        # Cascade always uses CAUSAL kernel mode (non-causal levels inflate kv_len)
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads

        # Concatenate kv_indices from all levels
        kv_indices_cat = torch.cat(kv_indices_arr, dim=0)
        if self._use_cuda_graph:
            if len(kv_indices_cat) > len(self._kv_indices_buf):
                raise ValueError(
                    f"kv_indices ({len(kv_indices_cat)}) exceeds "
                    f"kv_indices_buffer size ({len(self._kv_indices_buf)})"
                )
            self._kv_indices_buf[: len(kv_indices_cat)].copy_(
                kv_indices_cat, non_blocking=True
            )
            self._kv_indices = self._kv_indices_buf
        else:
            self._kv_indices = kv_indices_cat

        # Per-level causal flags: only last level is causal (if causal=True)
        causal_flags = [0] * self._num_levels
        if causal:
            causal_flags[-1] = 1

        # Number of pages per level (for computing level offsets)
        kv_indices_num_pages = [t.shape[0] for t in kv_indices_arr]

        self._plan_info = self.module.cascade_plan(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self.page_locked_int_workspace_buffer,
            qo_indptr_host_arr,
            kv_indptr_host_arr,
            kv_len_host_arr,
            causal_flags,
            kv_indices_num_pages,
            self._num_levels,
            num_qo_heads,
            num_kv_heads,
            head_dim_vo,
        )

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        logits_soft_cap: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run fused cascade attention.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor [total_q_len, num_qo_heads, head_dim].
        kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Paged KV cache shared across all levels.
        """
        k_cache, v_cache = _unpack_paged_kv_cache(kv_cache, self._kv_layout)
        if out is None:
            out = torch.empty_like(q)
        if lse is None:
            lse = torch.empty(
                q.shape[0], q.shape[1], device=q.device, dtype=torch.float32
            )
        head_dim_qk = q.shape[2]
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(head_dim_qk)

        # Reuse the existing run function — it handles cascade via plan_info
        self.module.run(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._kv_indices,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            self._sm_scale,
            logits_soft_cap,
        )

        return out, lse

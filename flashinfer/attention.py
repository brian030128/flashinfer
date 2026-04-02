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

from .api_logging import flashinfer_api
from .jit import gen_batch_attention_module
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_kv_layout,
    _unpack_paged_kv_cache,
    determine_attention_backend,
)
from .prefill import BatchPrefillWithPagedKVCacheWrapper
from .jit.attention.variants import attention_sink_decl
from .jit.utils import filename_safe_dtype_map


@functools.cache
def get_holistic_attention_module(*args):
    return gen_batch_attention_module(*args).build_and_load()


class BatchAttention:
    @flashinfer_api
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

    @flashinfer_api
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

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
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
        sm_scale = self._sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim_qk)
        if k_scale is not None:
            sm_scale *= k_scale
        if v_scale is None:
            v_scale = 1.0
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
            v_scale,
            sm_scale,
            logits_soft_cap,
            # ADDITIONAL_FUNC_PARAMS
            # PROFILER_FUNC_PARAMS
            *profiler_args,
        )

        return out, lse


class CascadeBatchAttentionWrapper:
    """Fused multi-level cascade attention using two non-cooperative kernel launches.

    All cascade levels are processed in one attention kernel launch, then a second
    reduction kernel merges cross-level and cross-chunk partials.

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

        qo_indptr_host_arr = [t.to("cpu", non_blocking=True) for t in qo_indptr_arr]
        kv_indptr_host_arr = [t.to("cpu", non_blocking=True) for t in kv_indptr_arr]
        kv_len_host_arr = [t.to("cpu", non_blocking=True) for t in kv_len_arr]
        torch.cuda.synchronize()

        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads

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

        causal_flags = [0] * self._num_levels
        if causal:
            causal_flags[-1] = 1

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

    def fast_cascade_plan(
        self,
        qo_indptr_host_arr: List[torch.Tensor],
        kv_indptr_host_arr: List[torch.Tensor],
        kv_indices_arr: List[torch.Tensor],
        kv_len_host_arr: List[torch.Tensor],
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
        """Fast plan that skips GPU->CPU transfers and torch.cuda.synchronize().

        Like plan(), but accepts pre-computed CPU tensors for qo_indptr, kv_indptr,
        and kv_len. Requires that plan() or fast_cascade_plan() was called at least
        once before (so self.module is cached).
        """
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        self._logits_soft_cap = logits_soft_cap

        # Reuse self.module from prior plan() call — no get_holistic_attention_module()
        # No GPU->CPU copy, no torch.cuda.synchronize()

        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads

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

        causal_flags = [0] * self._num_levels
        if causal:
            causal_flags[-1] = 1

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

    def plan_for_draft(
        self,
        max_draft_depth: int,
        first_call: bool = False,
        qo_indptr_host_arr: Optional[List[torch.Tensor]] = None,
        kv_indptr_host_arr: Optional[List[torch.Tensor]] = None,
        kv_indices_arr: Optional[List[torch.Tensor]] = None,
        kv_len_host_arr: Optional[List[torch.Tensor]] = None,
        num_qo_heads: int = 0,
        num_kv_heads: int = 0,
        head_dim_qk: int = 0,
        head_dim_vo: int = 0,
        page_size: int = 1,
        causal: bool = False,
        sm_scale: Optional[float] = None,
        logits_soft_cap: Optional[float] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
    ) -> None:
        """Plan once for max draft depth. Call update_draft_step() per step.

        Plans cascade attention for the worst-case suffix length (max_draft_depth),
        then identifies level-2 work items in the workspace buffer so that
        update_draft_step() can patch kv_len/kv_end without re-running the
        full scheduling.

        Args:
            max_draft_depth: Maximum draft suffix length (= speculative_num_steps).
            first_call: If True, uses plan() to JIT-compile the module.
                        If False, uses fast_cascade_plan() (no sync).
        """
        common = dict(
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk, head_dim_vo=head_dim_vo,
            page_size=page_size, causal=causal, sm_scale=sm_scale,
            logits_soft_cap=logits_soft_cap, q_data_type=q_data_type,
            kv_data_type=kv_data_type,
        )
        if first_call:
            # plan() accepts GPU or CPU tensors (CPU is a no-op for .to("cpu"))
            self.plan(
                qo_indptr_arr=qo_indptr_host_arr,
                kv_indptr_arr=kv_indptr_host_arr,
                kv_indices_arr=kv_indices_arr,
                kv_len_arr=kv_len_host_arr,
                **common,
            )
        else:
            self.fast_cascade_plan(
                qo_indptr_host_arr=qo_indptr_host_arr,
                kv_indptr_host_arr=kv_indptr_host_arr,
                kv_indices_arr=kv_indices_arr,
                kv_len_host_arr=kv_len_host_arr,
                **common,
            )

        # Extract byte offsets for task 1 (decode-like) arrays.
        # plan_info layout: [num_blks_x, num_blks_y, task0(12 fields), task1(12 fields), shared(8)]
        # Task 1 starts at index 14. Fields per task (NUM_TASK_ARGS=12):
        #   q_indptr(0), kv_indptr(1), partial_indptr(2), q_len(3), kv_len(4),
        #   q_start(5), kv_start(6), kv_end(7), kv_head_idx(8), work_indptr(9),
        #   cascade_num_kv_chunks(10), cascade_kv_chunk_idx(11)
        TASK1_BASE = 2 + 12  # = 14
        kv_len_byte_offset = self._plan_info[TASK1_BASE + 4]
        kv_end_byte_offset = self._plan_info[TASK1_BASE + 7]
        kv_indptr_byte_offset = self._plan_info[TASK1_BASE + 1]
        work_indptr_byte_offset = self._plan_info[TASK1_BASE + 9]

        self._draft_kv_len_start = kv_len_byte_offset // 4  # int32 index
        self._draft_kv_end_start = kv_end_byte_offset // 4

        # Read total_works from page-locked (CPU pinned) buffer — no GPU sync.
        # cascade_plan writes to page_locked first, then async DMA to GPU.
        page_locked_buf = self.page_locked_int_workspace_buffer.view(torch.int32)
        num_clusters = self._plan_info[1]  # num_blks_y = num_clusters
        work_indptr_start = work_indptr_byte_offset // 4
        total_works = int(page_locked_buf[work_indptr_start + num_clusters])

        # Filter: only patch Level 2 (unique suffix) work items.
        # The C++ scheduler sets kv_indptr = kv_indptr_h[level][i] + kv_indices_level_offsets[level].
        # Level 0 offsets start at 0; Level 1 offsets start at len(kv_indices_arr[0]).
        # So Level 2 items have kv_indptr >= level2_offset.
        # This is critical for configs where both levels land in Task 1
        # (e.g., topk=2, GQA=4: packed_qo_len = 8 < 16 for BOTH levels).
        kv_indptr_start = kv_indptr_byte_offset // 4
        work_kv_indptrs = page_locked_buf[kv_indptr_start : kv_indptr_start + total_works]
        level2_offset = kv_indices_arr[0].shape[0]
        level2_mask = work_kv_indptrs >= level2_offset
        self._draft_level2_indices = torch.where(level2_mask)[0].to(
            device=self.int_workspace_buffer.device
        )

    def update_draft_step(self, step_kv_len: int, step_kv_end: int) -> None:
        """Patch kv_len and kv_end for level-2 work items. No scheduling recomputation.

        In CUDA graph mode each step has its own wrapper with exact plan data,
        so _draft_level2_indices is not set — skip patching.

        Args:
            step_kv_len: kv_len_for_work for this step (= step_offset + qo_len).
            step_kv_end: effective_kv_len for this step (= step_offset).
        """
        if not hasattr(self, "_draft_level2_indices"):
            return
        buf = self.int_workspace_buffer.view(torch.int32)
        idx = self._draft_level2_indices
        buf[self._draft_kv_len_start + idx] = step_kv_len
        buf[self._draft_kv_end_start + idx] = step_kv_end

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        logits_soft_cap: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            1.0,  # v_scale
            self._sm_scale,
            logits_soft_cap,
        )

        return out, lse


class BatchAttentionWithAttentionSinkWrapper(BatchPrefillWithPagedKVCacheWrapper):
    r"""
    Wrapper for prefill and decode attention with paged KV-cache that adds support for
    attention sinks. This class extends `BatchPrefillWithPagedKVCacheWrapper`, providing
    a convenient interface for using attention sinks during prefill or decode attention.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        head_dim_qk: int = 128,
        head_dim_vo: int = 128,
        window_left: int = -1,
    ) -> None:
        # trtllm is separate code path
        assert backend in ["fa2", "fa3", "auto"]
        if backend == "auto":
            # dispatch backend before init jit module
            backend = determine_attention_backend(
                float_workspace_buffer.device,
                PosEncodingMode[pos_encoding_mode].value,
                use_fp16_qk_reduction,  # use_fp16_qk_reduction
                custom_mask_buf is not None,  # use_custom_mask
                q_data_type,
                kv_data_type,
            )

        jit_args = [
            f"batch_prefill_attention_sink_{filename_safe_dtype_map[q_data_type]}_swa_{window_left >= 0}_{backend}",  # uri
            q_data_type,  # dtype_q
            kv_data_type,  # dtype_kv
            q_data_type,  # dtype_o
            torch.int32,  # idtype
            head_dim_qk,  # hidden_dim_qk
            head_dim_vo,  # hidden_dim_vo
            ["sink"],  # additional_tensor_names
            ["float"],  # additional_tensor_dtypes
            ["sm_scale"],  # additional_scalar_names
            ["double"],  # additional_scalar_dtypes
            "AttentionSink",
            attention_sink_decl[backend],
        ]
        jit_kwargs = {
            "use_sliding_window": window_left >= 0,
            "use_fp16_qk_reduction": use_fp16_qk_reduction,
            "pos_encoding_mode": PosEncodingMode[pos_encoding_mode].value,
        }

        super().__init__(
            float_workspace_buffer=float_workspace_buffer,
            kv_layout=kv_layout,
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buf,
            paged_kv_indptr_buf=paged_kv_indptr_buf,
            paged_kv_indices_buf=paged_kv_indices_buf,
            paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
            custom_mask_buf=custom_mask_buf,
            mask_indptr_buf=mask_indptr_buf,
            backend=backend,
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )

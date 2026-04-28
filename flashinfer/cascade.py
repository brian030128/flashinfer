"""
Copyright (c) 2023 by FlashInfer team.

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
from typing import List, Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .decode import BatchDecodeWithPagedKVCacheWrapper
from .jit.cascade import gen_cascade_module
from .jit.fused_cascade import gen_fused_cascade_module
from .prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    _unpack_paged_kv_cache,
    single_prefill_with_kv_cache,
)
from .utils import (
    MaskMode,
    TensorLayout,
    _check_kv_layout,
    _get_cache_buf,
    canonicalize_torch_dtype,
    register_custom_op,
    register_fake_op,
)


@functools.cache
def get_fused_cascade_module(
    dtype_q,
    dtype_kv,
    dtype_o,
    dtype_idx,
    head_dim_qk,
    head_dim_vo,
    max_levels,
):
    return gen_fused_cascade_module(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        max_levels,
    ).build_and_load()


@functools.cache
def get_cascade_module():
    return gen_cascade_module().build_and_load()


@flashinfer_api
@register_custom_op("flashinfer::merge_state", mutates_args=())
def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Merge the attention output ``V`` and the logsumexp value ``S`` from the two
    KV-segments.
    Check :ref:`our tutorial <recursive-attention>` on the mathematical details.

    Parameters
    ----------
    v_a : torch.Tensor
        The attention output from the KV segment ``A``, shape:
        ``[seq_len, num_heads, head_dim]``.
    s_a : torch.Tensor
        The logsumexp value from the KV segment ``A``. expected to be a float32 tensor,
        shape: ``[seq_len, num_heads]``.
    v_b : torch.Tensor
        The attention output from the KV segment ``B``,
        shape: ``[seq_len, num_heads, head_dim]``.
    s_b : torch.Tensor
        The logsumexp value from the KV segment ``B``, expected to be a float32 tensor,
        shape: ``[seq_len, num_heads]``

    Returns
    -------
    V : torch.Tensor
        The merged attention output (equivalent to attention with merged KV-segment
        ``[A: B]``), shape: ``[seq_len, num_heads, head_dim]``.
    S : torch.Tensor
        The logsumexp value from the merged KV-segment ``[A: B]``, shape:
        ``[seq_len, num_heads]``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_merged, s_merged = flashinfer.merge_state(va, sa, vb, sb)
    >>> v_merged.shape
    torch.Size([2048, 32, 128])
    >>> s_merged.shape
    torch.Size([2048, 32])
    """
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    v_merged = torch.empty_like(v_a)
    s_merged = torch.empty_like(s_a)
    get_cascade_module().merge_state(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


@register_fake_op("flashinfer::merge_state")
def _fake_merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    v = torch.empty_like(v_a)
    s = torch.empty_like(s_a)
    return v, s


@flashinfer_api
@register_custom_op("flashinfer::merge_state_in_place", mutates_args=("v", "s"))
def merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    r"""Merge the self-attention state ``(v, s)`` with another state
    ``(v_other, s_other)`` in-place.

    Parameters
    ----------
    v : torch.Tensor
        The partial attention output to be updated in-place, shape:
        ``(seq_len, num_heads, head_dim)``.
    s : torch.Tensor
        The partial logsumexpr value to be updated in-place, expected to be a float32
        tensor, shape: ``(seq_len, num_heads)``.
    v_other : torch.Tensor
        The other attention output to be merged, shape:
        ``(seq_len, num_heads, head_dim)``.
    s_other : torch.Tensor
        The other logsumexp value to be merged, expected to be a float32 tensor,
        shape: ``(seq_len, num_heads)``.
    mask : Optional[torch.Tensor]
        The boolean mask tensor for whether to merge the state for a corresponding sequence
        or not. Useful for CUDA graphs. If not specified (default), will merge states for
        all sequences.
        shape: ``[seq_len]``

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> v = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> s = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_other = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> s_other = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> flashinfer.merge_state_in_place(v, s, v_other, s_other)
    """
    s = s.to(torch.float32)
    s_other = s_other.to(torch.float32)
    get_cascade_module().merge_state_in_place(v, s, v_other, s_other, mask)


@register_fake_op("flashinfer::merge_state_in_place")
def _fake_merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::merge_states", mutates_args=())
def merge_states(v: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Merge multiple attention states (v, s).

    Parameters
    ----------
    v : torch.Tensor
        The attention output from the KV segments, shape:
        ``[seq_len, num_states, num_heads, head_dim]``.
    s : torch.Tensor
        The logsumexp value from the KV segments, shape:
        ``[seq_len, num_states, num_heads]``, expected
        to be a float32 tensor.

    Returns
    -------
    V : torch.Tensor
        The merged attention output, shape: ``[seq_len, num_heads, head_dim]``.
    S : torch.Tensor
        The logsumexp value from the merged KV-segments, shape:
        ``[seq_len, num_heads]``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> num_states = 100
    >>> v = torch.randn(seq_len, num_states, num_heads, head_dim).half().to("cuda:0")
    >>> s = torch.randn(seq_len, num_states, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_merged, s_merged = flashinfer.merge_states(v, s)
    >>> v_merged.shape
    torch.Size([2048, 32, 128])
    >>> s_merged.shape
    torch.Size([2048, 32])
    """
    device = v.device
    s = s.to(torch.float32)
    seq_len, _, num_heads, head_dim = v.size()
    v_merged = torch.empty(seq_len, num_heads, head_dim, dtype=v.dtype, device=device)
    s_merged = torch.empty(seq_len, num_heads, dtype=torch.float32, device=device)
    get_cascade_module().merge_states(v, s, v_merged, s_merged)
    return v_merged, s_merged


@register_fake_op("flashinfer::merge_states")
def _fake_merge_states(
    v: torch.Tensor, s: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len, _, num_heads, head_dim = v.size()
    v_merged = torch.empty(seq_len, num_heads, head_dim, dtype=v.dtype)
    s_merged = torch.empty(seq_len, num_heads, dtype=torch.float32)
    return v_merged, s_merged


class MultiLevelCascadeAttentionWrapper:
    r"""Attention wrapper for memory efficient multi-level cascade inference, this API assumes all
    levels KV-Cache are stored in a unified paged table.

    Please check :ref:`cascade-inference-data-layout` for data layout in cascade inference.
    Note that it's not always beneficial to increase the number of levels because of the overhead
    of merging attention results.

    The idea of cascade inference is introduced in our `blog post <https://flashinfer.ai/2024/02/02/cascade-inference.html>`_.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
    ...     2, workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_kv_num_pages = 512
    >>> unique_kv_num_pages = 128
    >>> total_num_pages = shared_kv_num_pages + unique_kv_num_pages
    >>> shared_kv_page_indices = torch.arange(shared_kv_num_pages).int().to("cuda:0")
    >>> shared_kv_page_indptr = torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device="cuda:0")
    >>> unique_kv_page_indices = torch.arange(shared_kv_num_pages, total_num_pages).int().to("cuda:0")
    >>> unique_kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> shared_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda:0")
    >>> # 1 <= kv_last_page_len <= page_size
    >>> unique_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> qo_indptr_arr = [
    ...     torch.tensor([0, batch_size], dtype=torch.int32, device="cuda:0"),  # top-level for shared KV-Cache
    ...     torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0")    # bottom-level for unique KV-Cache
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> wrapper.plan(
    ...     qo_indptr_arr,
    ...     [shared_kv_page_indptr, unique_kv_page_indptr],
    ...     [shared_kv_page_indices, unique_kv_page_indices],
    ...     [shared_kv_last_page_len, unique_kv_last_page_len],
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = wrapper.run(q, kv_cache_at_layer[i])
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    See Also
    --------
    BatchPrefillWithPagedKVCacheWrapper
    """

    @flashinfer_api
    def __init__(
        self,
        num_levels,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_indptr_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_indices_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_last_page_len_buf_arr: Optional[List[torch.Tensor]] = None,
    ) -> None:
        r"""Constructor of :class:`MultiLevelCascadeAttentionWrapper`.

        Parameters
        ----------
        num_levels : int
            The number of levels in the cascade attention.
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        use_cuda_graph : bool
            Whether to use CUDA graph to capture the kernels, if enabled, the auxiliary data structures
            will be stored in provided buffers.
        qo_indptr_buf_arr : Optional[List[torch.Tensor]]
            An array of qo indptr buffers for each level, the array length should be equal to
            the number of levels.
            The last element of each tensor should be the total number of queries/outputs.
        paged_kv_indptr_buf_arr : Optional[List[torch.Tensor]]
            An array of paged kv-cache indptr buffers for each level, the array length should be
            equal to the number of levels.
        paged_kv_indices_buf_arr : Optional[List[torch.Tensor]]
            An array of paged kv-cache indices buffers for each level, the array length should be
            equal to the number of levels.
        paged_kv_last_page_len_buf_arr : Optional[List[torch.Tensor]]
            An array of paged kv-cache last page length buffers for each level, the array length
            should be equal to the number of levels.
        """
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            self._batch_prefill_wrappers = [
                BatchPrefillWithPagedKVCacheWrapper(
                    float_workspace_buffer,
                    kv_layout,
                    use_cuda_graph=True,
                    qo_indptr_buf=qo_indptr_buf,
                    paged_kv_indptr_buf=paged_kv_indptr_buf,
                    paged_kv_indices_buf=paged_kv_indices_buf,
                    paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
                )
                for (
                    qo_indptr_buf,
                    paged_kv_indptr_buf,
                    paged_kv_indices_buf,
                    paged_kv_last_page_len_buf,
                ) in zip(
                    qo_indptr_buf_arr,
                    paged_kv_indptr_buf_arr,
                    paged_kv_indices_buf_arr,
                    paged_kv_last_page_len_buf_arr,
                    strict=True,
                )
            ]
        else:
            self._batch_prefill_wrappers = [
                BatchPrefillWithPagedKVCacheWrapper(float_workspace_buffer, kv_layout)
                for _ in range(num_levels)
            ]
        self._num_levels = num_levels
        self._kv_layout = kv_layout

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffers: List[torch.Tensor],
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffers : List[torch.Tensor]
            The array of new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        for wrapper, int_workspace_buffer in zip(
            self._batch_prefill_wrappers, int_workspace_buffers, strict=True
        ):
            wrapper.reset_workspace_buffer(float_workspace_buffer, int_workspace_buffer)

    @flashinfer_api
    def plan(
        self,
        qo_indptr_arr: List[torch.Tensor],
        paged_kv_indptr_arr: List[torch.Tensor],
        paged_kv_indices_arr: List[torch.Tensor],
        paged_kv_last_page_len: List[torch.Tensor],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: str = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
    ):
        r"""Create auxiliary data structures for multi-level cascade attention for multiple
        forward calls within the same decode step. Please check
        :ref:`cascade-inference-data-layout` for data layout in cascade inference.

        Parameters
        ----------
        qo_indptr_arr : List[torch.Tensor]
            An array of qo indptr tensors for each level, the array length should be equal to
            the number of levels.
            The last element of each tensor should be the total number of queries/outputs.
        paged_kv_indptr_arr : List[torch.Tensor]
            An array of paged kv-cache indptr tensors for each level, the array length should be
            equal to the number of levels.
        paged_kv_indices_arr : List[torch.Tensor]
            An array of paged kv-cache indices tensors for each level, the array length should be
            equal to the number of levels.
        paged_kv_last_page_len : List[torch.Tensor]
            An array of paged kv-cache last page length tensors for each level, the array length
            should be equal to the number of levels.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim : int
            The dimension of the heads.
        page_size : int
            The page size of the paged kv-cache.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor. If None, will be set to torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        """
        for i, (
            wrapper,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) in enumerate(
            zip(
                self._batch_prefill_wrappers,
                qo_indptr_arr,
                paged_kv_indptr_arr,
                paged_kv_indices_arr,
                paged_kv_last_page_len,
                strict=True,
            )
        ):
            wrapper.plan(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=causal if i == self._num_levels - 1 else False,
                pos_encoding_mode=pos_encoding_mode,
                use_fp16_qk_reduction=use_fp16_qk_reduction,
                sm_scale=sm_scale,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
                q_data_type=q_data_type,
                kv_data_type=kv_data_type,
            )

    begin_forward = plan

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
    ):
        r"""Compute multi-level cascade attention.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.
        """
        out, lse = self._batch_prefill_wrappers[-1].run(
            q,
            paged_kv_cache,
            return_lse=True,
        )
        for wrapper in self._batch_prefill_wrappers[:-1]:
            out_i, lse_i = wrapper.run(q, paged_kv_cache, return_lse=True)
            merge_state_in_place(out, lse, out_i, lse_i)

        return out

    forward = run


class FusedMultiLevelCascadeAttentionWrapper:
    r"""Single-launch fused variant of :class:`MultiLevelCascadeAttentionWrapper`.

    The default wrapper launches one ``BatchPrefillWithPagedKVCache`` kernel per
    cascade level plus a ``merge_state_in_place`` between every pair — ``2*L - 1``
    launches per call. For shared-prefix tree drafting (the parent fast-draft
    project's tree-draft setting) per-level work is small, so back-to-back
    launches dominate latency and starve the GPU.

    This wrapper collapses the L per-level prefill kernels into a single launch
    via ``include/flashinfer/attention/fused_cascade.cuh``. The merge step is
    still a separate kernel (one ``merge_state_in_place`` per non-final level),
    matching the FlashAttention "split-K + reduction" pattern.

    v1 limitations:

    * ``causal=False`` only — final-level causal mask not supported in v1.
    * No sliding window, no logits soft cap, no RoPE in-kernel.
    * No KV split (each level's KV is processed as one chunk).
    * ``num_levels`` must be in ``[2, max_levels]``; ``max_levels`` defaults to 4.

    The plan/run API mirrors :class:`MultiLevelCascadeAttentionWrapper` so this
    class is a drop-in replacement under those constraints.
    """

    @staticmethod
    def _schedule_level_two_pool(
        qo_indptr_h: List[int],
        gqa_group_size: int,
    ) -> Tuple[
        List[int], List[int],   # pool_64: request_indices, qo_tile_indices
        List[int], List[int],   # pool_16: request_indices, qo_tile_indices
        List[int],              # o_indptr (shared, per-request)
    ]:
        """Two-pool work-tile scheduler. Per group with `n` queries:
        - packed = n * gqa_group_size
        - packed <= 16: 1 tile in pool_16 (qo_tile_idx=0)
        - 16 < packed <= 64: 1 tile in pool_64 (qo_tile_idx=0)
        - packed > 64: floor(packed/64) pool_64 tiles followed by 1 leftover
          tile (pool_16 if leftover<=16 else pool_64). qo_tile_idx is the
          tile's offset *in its pool's tile-grid*, so the kernel's
          `qo_packed_idx_base = qo_tile_idx * CTA_TILE_Q` maps to the
          correct packed-row range. e.g. packed=72 split as
          1*pool_64(qo_tile_idx=0, covers packed 0..63) + 1*pool_16
          (qo_tile_idx=4, covers packed 64..79).
        """
        req64: List[int] = []
        tile64: List[int] = []
        req16: List[int] = []
        tile16: List[int] = []
        o_indptr: List[int] = [0]
        batch = len(qo_indptr_h) - 1
        for i in range(batch):
            qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i]
            packed = qo_len * gqa_group_size
            if packed <= 16:
                req16.append(i)
                tile16.append(0)
            elif packed <= 64:
                req64.append(i)
                tile64.append(0)
            else:
                full = packed // 64
                leftover = packed - full * 64
                for t in range(full):
                    req64.append(i)
                    tile64.append(t)
                if leftover > 0:
                    if leftover <= 16:
                        # pool_16 tile starts at packed = full*64; in pool_16's
                        # tile grid (CTA_TILE_Q=16) that's tile index full*4.
                        req16.append(i)
                        tile16.append(full * 4)
                    else:
                        # pool_64 tile covers leftover<=64 with padding.
                        req64.append(i)
                        tile64.append(full)
            o_indptr.append(o_indptr[-1] + qo_len)
        return req64, tile64, req16, tile16, o_indptr

    def __init__(
        self,
        num_levels: int,
        float_workspace_buffer: Optional[torch.Tensor] = None,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        # Accepted for API compatibility with MultiLevelCascadeAttentionWrapper;
        # unused because the fused wrapper schedules in Python and allocates
        # workspace lazily inside plan().
        qo_indptr_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_indptr_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_indices_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_last_page_len_buf_arr: Optional[List[torch.Tensor]] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        max_levels: int = 4,
    ) -> None:
        if num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {num_levels}")
        if num_levels > max_levels:
            raise ValueError(
                f"num_levels={num_levels} exceeds max_levels={max_levels}; "
                "increase max_levels (causes a recompile of the fused module)."
            )
        _check_kv_layout(kv_layout)
        self._num_levels = num_levels
        self._max_levels = max_levels
        self._kv_layout = kv_layout
        # Pick a device for plan-time tensor construction. Priority:
        # explicit `device` kwarg, then the workspace buffer's device, then
        # we infer from the first tensor passed into plan().
        if device is not None:
            self._device: Optional[torch.device] = torch.device(device)
        elif float_workspace_buffer is not None:
            self._device = float_workspace_buffer.device
        else:
            self._device = None
        self._cached_module = None
        self._float_workspace_buffer = float_workspace_buffer
        # Filled in by plan().
        self._planned = False

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @flashinfer_api
    def plan(
        self,
        qo_indptr_arr: List[torch.Tensor],
        paged_kv_indptr_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_indices_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_last_page_len: Optional[List[torch.Tensor]] = None,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        page_size: Optional[int] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = torch.float16,
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        *,
        # Aliases used by CascadeBatchAttentionWrapper-style callers
        # (see fast-draft/tests/bench_tree_attn.py). Either naming style works.
        kv_indptr_arr: Optional[List[torch.Tensor]] = None,
        kv_indices_arr: Optional[List[torch.Tensor]] = None,
        kv_len_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_last_page_len_arr: Optional[List[torch.Tensor]] = None,
        head_dim_qk: Optional[int] = None,
        head_dim_vo: Optional[int] = None,
        use_profiler: bool = False,
    ) -> None:
        # Resolve naming-style aliases.
        if paged_kv_indptr_arr is None:
            paged_kv_indptr_arr = kv_indptr_arr
        if paged_kv_indices_arr is None:
            paged_kv_indices_arr = kv_indices_arr
        if paged_kv_last_page_len is None:
            paged_kv_last_page_len = paged_kv_last_page_len_arr
        if head_dim is None:
            if head_dim_qk is not None and head_dim_vo is not None:
                if head_dim_qk != head_dim_vo:
                    raise ValueError(
                        f"head_dim_qk={head_dim_qk} must equal head_dim_vo={head_dim_vo} "
                        "(fused cascade kernel uses a single head_dim)."
                    )
                head_dim = head_dim_qk
            elif head_dim_qk is not None:
                head_dim = head_dim_qk
            elif head_dim_vo is not None:
                head_dim = head_dim_vo

        if paged_kv_indptr_arr is None or paged_kv_indices_arr is None:
            raise ValueError(
                "Must provide paged_kv_indptr_arr/paged_kv_indices_arr "
                "(or kv_indptr_arr/kv_indices_arr aliases)."
            )

        # Derive paged_kv_last_page_len from kv_len_arr if not provided directly.
        # CascadeBatchAttention-style callers pass total per-request KV LENGTHS
        # via kv_len_arr; we convert to last_page_len = ((kv_len - 1) % ps) + 1.
        if paged_kv_last_page_len is None:
            if kv_len_arr is None:
                raise ValueError(
                    "Must provide either paged_kv_last_page_len or kv_len_arr."
                )
            paged_kv_last_page_len = []
            for kv_len in kv_len_arr:
                kv_len_i32 = kv_len.to(torch.int32)
                # Element-wise: (kv_len - 1) % page_size + 1, clamped >= 1.
                last = torch.where(
                    kv_len_i32 > 0,
                    ((kv_len_i32 - 1) % page_size) + 1,
                    torch.ones_like(kv_len_i32),
                )
                paged_kv_last_page_len.append(last)

        if head_dim is None:
            raise ValueError("Must provide head_dim (or head_dim_qk + head_dim_vo).")
        if num_qo_heads is None or num_kv_heads is None or page_size is None:
            raise ValueError(
                "num_qo_heads, num_kv_heads, and page_size are required."
            )

        if causal:
            raise NotImplementedError(
                "causal=True is not supported in v1 of FusedMultiLevelCascadeAttentionWrapper "
                "(see class docstring); fall back to MultiLevelCascadeAttentionWrapper."
            )
        if window_left != -1:
            raise NotImplementedError(
                "Sliding window is not supported in v1 of FusedMultiLevelCascadeAttentionWrapper."
            )
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise NotImplementedError(
                "logits_soft_cap is not supported in v1 of FusedMultiLevelCascadeAttentionWrapper."
            )
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads={num_qo_heads} must be divisible by num_kv_heads={num_kv_heads}"
            )
        if (
            len(qo_indptr_arr) != self._num_levels
            or len(paged_kv_indptr_arr) != self._num_levels
            or len(paged_kv_indices_arr) != self._num_levels
            or len(paged_kv_last_page_len) != self._num_levels
        ):
            raise ValueError(
                f"all per-level arrays must have length num_levels={self._num_levels}"
            )

        q_dtype = canonicalize_torch_dtype(q_data_type)
        kv_dtype = (
            canonicalize_torch_dtype(kv_data_type) if kv_data_type is not None else q_dtype
        )

        device = qo_indptr_arr[0].device
        gqa_group_size = num_qo_heads // num_kv_heads

        # The scheduler needs host-side qo_indptr to enumerate per-request tiles.
        # qo_indptr is small (batch+1 entries) so .cpu() is cheap.
        qo_indptr_h_arr = [t.cpu().to(torch.int32).tolist() for t in qo_indptr_arr]

        # For variable-depth cascades, deeper levels may have fewer participating
        # rows than the root level (e.g. only "deep branch" queries reach level 3).
        # We assume rows are contiguous from index 0 (caller orders Q so deep
        # queries come first), so partial_o sized for the *root* level covers
        # every level's writes. Per-level rows that don't participate are left
        # at LSE=-inf (zeroed in run()) so they merge as no-ops.
        total_qo_rows = max(qh[-1] for qh in qo_indptr_h_arr)
        for l, qh in enumerate(qo_indptr_h_arr):
            if qh[-1] > total_qo_rows:
                raise ValueError(
                    f"qo_indptr_arr[{l}][-1]={qh[-1]} exceeds the max across levels "
                    f"({total_qo_rows}); deeper levels must cover a contiguous prefix "
                    "of the rows handled by the root level."
                )

        # Two-pool scheduler: one set of buffers per cta_tile_q (16, 64).
        # Per-pool state, indexed by pool_idx ∈ {0=pool_64, 1=pool_16}.
        # Each pool gets its own request_indices/qo_tile_indices/kv_tile_indices/
        # level_id_per_cta/level_metadata/kv_chunk_size_ptr concatenations.
        # The user inputs (qo_indptr / paged_kv_*) and the per-request o_indptr
        # are SHARED across pools because both pools reference the same problem
        # data; only the tile schedule differs.
        per_pool_request_chunks: List[List[torch.Tensor]] = [[], []]
        per_pool_tile_chunks: List[List[torch.Tensor]] = [[], []]
        per_pool_kv_tile_chunks: List[List[torch.Tensor]] = [[], []]
        per_pool_level_id_chunks: List[List[torch.Tensor]] = [[], []]
        per_pool_level_metadata: List[List[int]] = [[], []]
        per_pool_kv_chunk_size_values: List[List[int]] = [[], []]
        per_pool_total_ctas = [0, 0]
        o_indptr_chunks: List[torch.Tensor] = []

        for l in range(self._num_levels):
            req64, tile64, req16, tile16, o_ind = self._schedule_level_two_pool(
                qo_indptr_h_arr[l], gqa_group_size
            )
            batch_l = len(qo_indptr_h_arr[l]) - 1
            num_pages_l = paged_kv_indices_arr[l].numel()
            o_indptr_chunks.append(
                torch.tensor(o_ind, dtype=torch.int32, device=device)
            )
            for pool_idx, (req, tile) in enumerate([(req64, tile64), (req16, tile16)]):
                padded_batch_l = len(req)
                per_pool_request_chunks[pool_idx].append(
                    torch.tensor(req, dtype=torch.int32, device=device)
                )
                per_pool_tile_chunks[pool_idx].append(
                    torch.tensor(tile, dtype=torch.int32, device=device)
                )
                per_pool_kv_tile_chunks[pool_idx].append(
                    torch.zeros(padded_batch_l, dtype=torch.int32, device=device)
                )
                per_pool_level_id_chunks[pool_idx].append(
                    torch.full(
                        (padded_batch_l,), l, dtype=torch.int32, device=device
                    )
                )
                per_pool_kv_chunk_size_values[pool_idx].append(0x7FFFFFFF)
                per_pool_level_metadata[pool_idx].extend(
                    [batch_l, padded_batch_l, num_pages_l, 0]
                )
                per_pool_total_ctas[pool_idx] += padded_batch_l

        if per_pool_total_ctas[0] == 0 and per_pool_total_ctas[1] == 0:
            raise ValueError("Total padded CTA count is 0 — no work to do.")

        # Concatenate user-provided tensors once (shared across pools).
        def _concat_int32(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.cat(
                [t.to(device=device, dtype=torch.int32) for t in tensors], dim=0
            )

        qo_indptr_buf = _concat_int32(qo_indptr_arr)
        paged_kv_indptr_buf = _concat_int32(paged_kv_indptr_arr)
        paged_kv_indices_buf = _concat_int32(paged_kv_indices_arr)
        paged_kv_last_page_len_buf = _concat_int32(paged_kv_last_page_len)
        o_indptr_buf = torch.cat(o_indptr_chunks, dim=0)

        # Concatenate per-pool scheduler outputs.
        def _maybe_cat(chunks: List[torch.Tensor]) -> torch.Tensor:
            return (
                torch.cat(chunks, dim=0)
                if any(c.numel() for c in chunks)
                else torch.empty(0, dtype=torch.int32, device=device)
            )

        pool_64_request_indices_buf = _maybe_cat(per_pool_request_chunks[0])
        pool_64_qo_tile_indices_buf = _maybe_cat(per_pool_tile_chunks[0])
        pool_64_kv_tile_indices_buf = _maybe_cat(per_pool_kv_tile_chunks[0])
        pool_64_level_id_per_cta = _maybe_cat(per_pool_level_id_chunks[0])
        pool_64_kv_chunk_size_ptr_buf = torch.tensor(
            per_pool_kv_chunk_size_values[0], dtype=torch.int32, device=device
        )
        pool_64_level_metadata = per_pool_level_metadata[0]

        pool_16_request_indices_buf = _maybe_cat(per_pool_request_chunks[1])
        pool_16_qo_tile_indices_buf = _maybe_cat(per_pool_tile_chunks[1])
        pool_16_kv_tile_indices_buf = _maybe_cat(per_pool_kv_tile_chunks[1])
        pool_16_level_id_per_cta = _maybe_cat(per_pool_level_id_chunks[1])
        pool_16_kv_chunk_size_ptr_buf = torch.tensor(
            per_pool_kv_chunk_size_values[1], dtype=torch.int32, device=device
        )
        pool_16_level_metadata = per_pool_level_metadata[1]

        # Pre-allocate partial output buffers for levels 1..L-1 only.
        # Level 0 writes directly into the user's `out`/`lse` buffer (or a
        # fresh allocation in run() if the caller didn't supply one),
        # eliminating the trailing memcpy that otherwise dominates per-call
        # cost in small-batch decode regimes (the bench_tree_attn workload).
        # Each per-level slice [l] is contiguous because level-major layout
        # matches the per-level prefill device body's hardcoded
        # `o_stride_n = num_qo_heads * head_dim` (see prefill.cuh:1820).
        partial_o = torch.empty(
            max(self._num_levels - 1, 1),
            total_qo_rows,
            num_qo_heads,
            head_dim,
            dtype=q_dtype,
            device=device,
        )
        partial_lse = torch.empty(
            max(self._num_levels - 1, 1),
            total_qo_rows,
            num_qo_heads,
            dtype=torch.float32,
            device=device,
        )

        # Stash shared per-level inputs.
        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self._o_indptr_buf = o_indptr_buf
        self._partial_o = partial_o
        self._partial_lse = partial_lse

        # Stash per-pool scheduler outputs. Pool index 0 = CTA_TILE_Q=64,
        # pool index 1 = CTA_TILE_Q=16. Each pool may have 0 work if no
        # tiles fall into it; in that case we skip its launch in run().
        self._pool_64_request_indices_buf = pool_64_request_indices_buf
        self._pool_64_qo_tile_indices_buf = pool_64_qo_tile_indices_buf
        self._pool_64_kv_tile_indices_buf = pool_64_kv_tile_indices_buf
        self._pool_64_kv_chunk_size_ptr_buf = pool_64_kv_chunk_size_ptr_buf
        self._pool_64_level_id_per_cta = pool_64_level_id_per_cta
        self._pool_64_level_metadata = pool_64_level_metadata
        self._pool_64_total_ctas = per_pool_total_ctas[0]

        self._pool_16_request_indices_buf = pool_16_request_indices_buf
        self._pool_16_qo_tile_indices_buf = pool_16_qo_tile_indices_buf
        self._pool_16_kv_tile_indices_buf = pool_16_kv_tile_indices_buf
        self._pool_16_kv_chunk_size_ptr_buf = pool_16_kv_chunk_size_ptr_buf
        self._pool_16_level_id_per_cta = pool_16_level_id_per_cta
        self._pool_16_level_metadata = pool_16_level_metadata
        self._pool_16_total_ctas = per_pool_total_ctas[1]

        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._sm_scale = (
            sm_scale if sm_scale is not None else 1.0 / (head_dim**0.5)
        )
        # Variable-depth: not every row participates at every level. We need
        # to pre-init partial_o/partial_lse so that the unwritten rows merge as
        # no-ops. For uniform depth we can skip the pre-init entirely (saves
        # two kernel launches per run() — ~10 µs at decode time).
        self._uniform_depth = all(
            qh[-1] == total_qo_rows for qh in qo_indptr_h_arr
        )
        self._total_qo_rows = total_qo_rows
        self._window_left = window_left
        self._q_dtype = q_dtype
        self._kv_dtype = kv_dtype
        self._device = device
        self._planned = True

        # JIT-compile the fused module (once per dtype/head_dim/max_levels combo).
        self._cached_module = get_fused_cascade_module(
            q_dtype,
            kv_dtype,
            q_dtype,  # output dtype = input dtype
            torch.int32,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            self._max_levels,
        )

    begin_forward = plan

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run fused multi-level cascade attention.

        Returns ``out`` by default (matches MultiLevelCascadeAttentionWrapper).
        If ``return_lse=True`` or either ``out``/``lse`` buffer is supplied,
        returns ``(out, lse)`` (matches CascadeBatchAttentionWrapper).
        ``out``/``lse`` buffers, when provided, are written in-place.
        """
        if not self._planned:
            raise RuntimeError("plan() must be called before run().")
        # Track whether the caller passed buffers (return tuple iff yes).
        wants_lse = return_lse or (out is not None) or (lse is not None)

        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)

        # Allocate output buffers if not provided. Level 0 of the fused kernel
        # writes here directly; the post-kernel merge then accumulates partials
        # 1..L-1 in place.
        if out is None:
            out = torch.empty(
                self._total_qo_rows,
                self._num_qo_heads,
                self._head_dim,
                dtype=self._q_dtype,
                device=self._device,
            )
        if lse is None:
            lse = torch.empty(
                self._total_qo_rows,
                self._num_qo_heads,
                dtype=torch.float32,
                device=self._device,
            )

        if not self._uniform_depth:
            # Variable-depth cascade: pre-fill so rows a level skips merge as
            # no-ops. For uniform depth every row gets written by every level,
            # so the pre-init is dead work (and costs two kernel launches).
            self._partial_lse.fill_(float("-inf"))
            self._partial_o.zero_()
            # Also init out/lse for rows level 0 itself doesn't cover.
            lse.fill_(float("-inf"))
            out.zero_()

        # Two-pool dispatch: launch CTA_TILE_Q=64 kernel first (carries the
        # large-prefix work), then CTA_TILE_Q=16 (per-row work + small
        # leftovers). Skip a launch if its pool has no tiles. Each launch
        # runs the same fused kernel template with its CTA_TILE_Q baked in,
        # so the compiler gets unrestricted scheduling on each path.
        if self._pool_64_total_ctas > 0:
            self._cached_module.fused_paged_run(
                q,
                k_cache,
                v_cache,
                self._qo_indptr_buf,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                self._pool_64_request_indices_buf,
                self._pool_64_qo_tile_indices_buf,
                self._pool_64_kv_tile_indices_buf,
                self._o_indptr_buf,
                self._pool_64_kv_chunk_size_ptr_buf,
                self._pool_64_level_id_per_cta,
                out,
                lse,
                self._partial_o,
                self._partial_lse,
                self._pool_64_level_metadata,
                self._num_levels,
                TensorLayout[self._kv_layout].value,
                self._window_left,
                self._sm_scale,
                64,
                enable_pdl,
            )
        if self._pool_16_total_ctas > 0:
            self._cached_module.fused_paged_run(
                q,
                k_cache,
                v_cache,
                self._qo_indptr_buf,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                self._pool_16_request_indices_buf,
                self._pool_16_qo_tile_indices_buf,
                self._pool_16_kv_tile_indices_buf,
                self._o_indptr_buf,
                self._pool_16_kv_chunk_size_ptr_buf,
                self._pool_16_level_id_per_cta,
                out,
                lse,
                self._partial_o,
                self._partial_lse,
                self._pool_16_level_metadata,
                self._num_levels,
                TensorLayout[self._kv_layout].value,
                self._window_left,
                self._sm_scale,
                16,
                enable_pdl,
            )

        # Accumulate partials 1..L-1 into out/lse via in-place merges.
        for l in range(self._num_levels - 1):
            merge_state_in_place(
                out, lse, self._partial_o[l], self._partial_lse[l]
            )

        if wants_lse:
            return out, lse
        return out

    forward = run


class BatchDecodeWithSharedPrefixPagedKVCacheWrapper:
    r"""Wrapper class for decode attention with shared-prefix paged kv-cache for batch
    of requests. The shared-prefix KV-Cache was stored in a standalone tensors, and the
    unique KV-Cache of each request was stored in a paged KV-Cache data structure.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Warning
    -------
    This API will be deprecated in the future, please use
    :class:`MultiLevelCascadeAttentionWrapper` instead.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_prefix_len = 8192
    >>> unique_kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> unique_kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= kv_last_page_len <= page_size
    >>> unique_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> unique_kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_k_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_v_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> wrapper.begin_forward(
    ...     unique_kv_page_indptr,
    ...     unique_kv_page_indices,
    ...     unique_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     data_type=torch.float16
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     k_shared = shared_k_data_at_layer[i]
    ...     v_shared = shared_v_data_at_layer[i]
    ...     unique_kv_cache = unique_kv_cache_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = wrapper.forward(q, k_shared, v_shared, unique_kv_cache)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's shared prefix batch decode attention creates
    some auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    @flashinfer_api
    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        self._batch_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._batch_decode_wrapper.reset_workspace_buffer(
            float_workspace_buffer, int_workspace_buffer
        )

    @flashinfer_api
    def begin_forward(
        self,
        unique_kv_indptr: torch.Tensor,
        unique_kv_indices: torch.Tensor,
        unique_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        data_type: str = "float16",
    ) -> None:
        r"""Plan shared-prefix batch decode attention for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``
        num_qo_heads : int
            The number of query/output heads
        num_kv_heads : int
            The number of key/value heads
        head_dim : int
            The dimension of the heads
        page_size : int
            The page size of the paged kv cache
        data_type : Union[str, torch.dtype]
            The data type of the paged kv cache

        Note
        ----
        The :meth:`begin_forward` method should be called before any :meth:`forward` or
        :meth:`forward_return_lse` calls,
        auxiliary data structures will be created during this call and cached for
        multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.


        See Also
        --------
        MultiLevelCascadeAttentionWrapper
        """
        self._batch_decode_wrapper.begin_forward(
            unique_kv_indptr,
            unique_kv_indices,
            unique_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            data_type=data_type,
        )

    @flashinfer_api
    def forward(
        self,
        q: torch.Tensor,
        k_shared: torch.Tensor,
        v_shared: torch.Tensor,
        unique_kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute batch decode attention between queries and shared-prefix paged
        kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
        k_shared : torch.Tensor
            The shared prefix key tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        v_shared : torch.Tensor
            The shared prefix value tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        unique_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The request-independent suffix paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.

        Returns
        -------
        V : torch.Tensor
            The attention output, shape: ``[batch_size, num_heads, head_dim]``
        """
        V_shared, S_shared = single_prefill_with_kv_cache(
            q,
            k_shared,
            v_shared,
            causal=False,
            pos_encoding_mode="NONE",
            kv_layout=self._kv_layout,
            sm_scale=self._batch_decode_wrapper._sm_scale,
            rope_scale=self._batch_decode_wrapper._rope_scale,
            rope_theta=self._batch_decode_wrapper._rope_theta,
            return_lse=True,
        )
        V_unique, S_unique = self._batch_decode_wrapper.forward_return_lse(
            q,
            unique_kv_cache,
            pos_encoding_mode="NONE",
        )
        merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
        return V_shared

    @flashinfer_api
    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect"""
        pass


class BatchPrefillWithSharedPrefixPagedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with shared-prefix paged kv-cache for
    batch of requests.

    Check :ref:`our tutorial<kv-layout>` for paged kv-cache layout.

    Warning
    -------
    This API will be deprecated in the future, please use
    :class:`MultiLevelCascadeAttentionWrapper` instead.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_prefix_len = 8192
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> paged_kv_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= paged_kv_last_page_len <= page_size
    >>> paged_kv_last_page_len= torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_k_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_v_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.begin_forward(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    ...     kv_cache = kv_cache_at_layer[i]
    ...     k_shared = shared_k_data_at_layer[i]
    ...     v_shared = shared_v_data_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.forward(
    ...         q, k_shared, v_shared, kv_cache, causal=True
    ...     )
    ...     outputs.append(o)
    ...
    s[0].shape>>> # clear auxiliary data structures
    >>> prefill_wrapper.end_forward()
    >>> outputs[0].shape
    torch.Size([100, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's shared-prefix batch prefill/append attention
    operators creates some auxiliary data structures, these data structures can be
    reused across multiple prefill/append attention calls (e.g. different Transformer
    layers). This wrapper class manages the lifecycle of these data structures.
    """

    @flashinfer_api
    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithSharedPrefixPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        self._batch_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._batch_prefill_wrapper.reset_workspace_buffer(
            float_workspace_buffer, int_workspace_buffer
        )

    @flashinfer_api
    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
    ) -> None:
        r"""Create auxiliary data structures for shared-prefix batch prefill/append
        attention for multiple forward calls within the same prefill/append step.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        paged_kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        paged_kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[qo_indptr[-1]]``.
        paged_kv_last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged
            kv-cache, shape: ``[batch_size]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim : int
            The dimension of the heads.
        page_size : int
            The page size of the paged kv-cache.

        Note
        ----
        The :meth:`begin_forward` method should be called before any :meth:`forward`
        or :meth:`forward_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        self._batch_prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
        )

    @flashinfer_api
    def forward(
        self,
        q: torch.Tensor,
        k_shared: torch.Tensor,
        v_shared: torch.Tensor,
        unique_kv_cache: torch.Tensor,
        causal: bool = False,
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Compute batch prefill/append attention between query and shared-prefix paged
        kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        k_shared : torch.Tensor
            The shared prefix key tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        v_shared ; torch.Tensor
            The shared prefix value tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        unique_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The request-independent suffix paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.

        causal : bool
            Whether to apply causal mask on the attention matrix.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        sm_scale : Optional[float]
            The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.

        Returns
        -------
        V : torch.Tensor
            The attention output, shape: ``[qo_indptr[-1], num_heads, head_dim]``.

        See Also
        --------
        MultiLevelCascadeAttentionWrapper
        """
        V_shared, S_shared = single_prefill_with_kv_cache(
            q,
            k_shared,
            v_shared,
            causal=False,
            pos_encoding_mode="NONE",
            kv_layout=self._kv_layout,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            return_lse=True,
        )
        V_unique, S_unique = self._batch_prefill_wrapper.forward_return_lse(
            q,
            unique_kv_cache,
            causal=causal,
            pos_encoding_mode="NONE",
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
        return V_shared

    @flashinfer_api
    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect"""
        pass

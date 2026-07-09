# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE weight-gradient (wgrad) op wrapper.

Mirrors the Triton ``fused_moe_wgrad`` contract so the same routing metadata
(``sorted_token_ids`` + ``block_start``/``blocks_per_expert`` built with
``block_size == WGRAD_BLOCK_M``) can be reused verbatim.
"""

from __future__ import annotations

import torch

from .kernels.moe_wgrad_flydsl_v2 import compile_moe_wgrad_v2, WGRAD_BLOCK_M
from .kernels.tensor_shim import ptr_arg, _run_compiled

__all__ = ["flydsl_moe_wgrad", "WGRAD_BLOCK_M"]


def flydsl_moe_wgrad(
    x: torch.Tensor,
    grad: torch.Tensor,
    dw: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    block_start: torch.Tensor,
    blocks_per_expert: torch.Tensor,
    top_k: int,
    mul_routed_weight: bool,
    block_n: int = 128,
    block_k: int = 128,
    warps_n: int = 2,
    warps_k: int = 2,
) -> None:
    """Compute ``dw[e] = sum_{slots of e} grad[slot]^T @ x[slot // top_k]`` in place.

    See ``aiter.ops.triton.moe.moe_wgrad.fused_moe_wgrad`` for the argument contract.

    ``top_k`` is passed to the kernel as a compile-time constant so the
    ``slot // top_k`` gather divide lowers to a shift/magic-multiply instead of
    the runtime software-division sequence. ``block_n``/``block_k`` and
    ``warps_n``/``warps_k`` select the workgroup tile; the defaults
    (``128x128`` over ``2x2`` warps) are a strong general config on CDNA4.
    """
    num_experts, N, K = dw.shape
    num_valid_tokens = topk_ids.numel()

    assert x.dtype == grad.dtype == dw.dtype == torch.bfloat16
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()

    topk_w = topk_weights.reshape(-1).to(torch.float32)
    if not topk_w.is_contiguous():
        topk_w = topk_w.contiguous()

    exe = compile_moe_wgrad_v2(
        dtype="bf16",
        mul_routed_weight=bool(mul_routed_weight),
        block_n=int(block_n),
        block_k=int(block_k),
        warps_n=int(warps_n),
        warps_k=int(warps_k),
        topk=int(top_k),
    )

    _run_compiled(
        exe,
        ptr_arg(dw),
        ptr_arg(x),
        ptr_arg(grad),
        ptr_arg(topk_w),
        ptr_arg(sorted_token_ids),
        ptr_arg(block_start),
        ptr_arg(blocks_per_expert),
        int(N),
        int(K),
        int(num_valid_tokens),
        int(top_k),
        int(num_experts),
        torch.cuda.current_stream(),
    )

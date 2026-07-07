# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional, Tuple

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton._triton_kernels.moe.moe_wgrad import (
    _fused_moe_wgrad_kernel,
    _fused_moe_wgrad_persistent_kernel,
)

_LOGGER = AiterTritonLogger()

_USE_MOE_WGRAD_PERSISTENT_KERNEL = False


def moe_wgrad_set_use_persistent_kernel(value: bool):
    """Toggle the persistent-CTA wgrad kernel (mirrors ``moe_set_use_persistent_kernel``)."""
    global _USE_MOE_WGRAD_PERSISTENT_KERNEL
    _USE_MOE_WGRAD_PERSISTENT_KERNEL = value


def moe_wgrad_block_offsets(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert sorted-block layout for the wgrad kernel (sync-free, on-device).

    Mirrors the block layout produced by ``moe_align_block_size``: experts are laid out
    in id order, each padded up to a multiple of ``block_size``.

    Returns
    -------
    (block_start, blocks_per_expert):
        ``block_start[e]`` is the first sorted-block index owned by expert ``e`` and
        ``blocks_per_expert[e]`` is how many blocks it spans, both ``int32`` on device.
    """
    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=num_experts)
    blocks_per_expert = ((counts + block_size - 1) // block_size).to(torch.int32)
    block_start = torch.zeros_like(blocks_per_expert)
    if num_experts > 1:
        block_start[1:] = torch.cumsum(blocks_per_expert, dim=0)[:-1].to(torch.int32)
    return block_start, blocks_per_expert


def fused_moe_wgrad(
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
    compute_type: tl.dtype,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Fused weight-gradient for a permute-free MoE grouped GEMM.

    Computes ``dw[e] = sum_{slots of e} grad[slot]^T @ x[slot // top_k]`` in place,
    gathering both operands along the contraction (token-slot) axis via
    ``sorted_token_ids`` -- the same align buffer used by the forward gather-GEMM. No
    activation/grad reordering is materialized and no host synchronization is required.

    Parameters
    ----------
    x:
        Activations ``[num_tokens, K]``.
    grad:
        Upstream gradient rows ``[num_tokens * top_k, N]`` (one row per routed slot).
    dw:
        Output weight gradient ``[num_experts, N, K]`` (written in place).
    topk_weights:
        Router weights ``[num_tokens, top_k]``; only read when ``mul_routed_weight``.
    topk_ids:
        Routing indices ``[num_tokens, top_k]`` (used only for ``num_valid_tokens``).
    sorted_token_ids:
        Expert-grouped, block-padded routed-slot ids from ``moe_align_block_size``.
    block_start, blocks_per_expert:
        Per-expert sorted-block layout from ``moe_wgrad_block_offsets`` (built with the
        same ``block_size`` == ``BLOCK_SIZE_M`` as the align buffers).
    top_k:
        MoE routing top-k.
    mul_routed_weight:
        If True, scale gathered grad rows by ``topk_weights`` (matches ``fused_moe``).
    compute_type:
        Triton output dtype for ``dw``.
    config:
        Kernel tile config; ``BLOCK_SIZE_M`` must equal the align ``block_size``.
    """
    num_experts, N, K = dw.shape
    num_valid_tokens = topk_ids.numel()

    _LOGGER.info(
        f"FUSED_MOE_WGRAD:  x={tuple(x.shape)}  grad={tuple(grad.shape)}  "
        f"dw={tuple(dw.shape)}  top_k={top_k}  mul_routed_weight={mul_routed_weight}"
    )

    if config is None:
        config = get_default_moe_wgrad_config()

    if _USE_MOE_WGRAD_PERSISTENT_KERNEL:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
        cfg = dict(config)
        group_size_m = cfg.pop("GROUP_SIZE_M", 1)
        trans_free = cfg.pop("TRANS_FREE", False)
        grid = lambda META: (  # noqa: E731
            min(
                NUM_SMS,
                num_experts
                * triton.cdiv(N, META["BLOCK_SIZE_N"])
                * triton.cdiv(K, META["BLOCK_SIZE_K"]),
            ),
        )
        _fused_moe_wgrad_persistent_kernel[grid](
            x,
            grad,
            dw,
            topk_weights,
            sorted_token_ids,
            block_start,
            blocks_per_expert,
            num_experts,
            N,
            K,
            num_valid_tokens,
            x.stride(0),
            x.stride(1),
            grad.stride(0),
            grad.stride(1),
            dw.stride(0),
            dw.stride(1),
            dw.stride(2),
            top_k,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            compute_type=compute_type,
            TRANS_FREE=trans_free,
            GROUP_SIZE_M=group_size_m,
            NUM_SMS=NUM_SMS,
            NUM_XCDS=get_num_xcds(),
            **cfg,
        )
        return

    cfg = dict(config)
    cfg.pop("GROUP_SIZE_M", None)
    trans_free = cfg.pop("TRANS_FREE", False)

    grid = lambda META: (  # noqa: E731
        num_experts
        * triton.cdiv(N, META["BLOCK_SIZE_N"])
        * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )

    _fused_moe_wgrad_kernel[grid](
        x,
        grad,
        dw,
        topk_weights,
        sorted_token_ids,
        block_start,
        blocks_per_expert,
        N,
        K,
        num_valid_tokens,
        x.stride(0),
        x.stride(1),
        grad.stride(0),
        grad.stride(1),
        dw.stride(0),
        dw.stride(1),
        dw.stride(2),
        top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=compute_type,
        TRANS_FREE=trans_free,
        **cfg,
    )


def get_default_moe_wgrad_config(block_size_m: int = 32) -> Dict[str, Any]:
    """Default tile config. ``BLOCK_SIZE_M`` must match the align ``block_size``.

    ``GROUP_SIZE_M`` only affects the persistent kernel (grouped ``N x K`` tile ordering);
    it is ignored by the non-persistent launch.
    """
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "CONTRACT_M": 64,
        # Gather grad pre-transposed so tl.dot needs no tl.trans: this removes the LDS
        # transpose round-trip *and* cuts VGPR pressure enough that a 128x128 tile reaches
        # occ=4 waves/SIMD (vs occ=2 for the old 256x128 no-trans default). ~8-13% faster
        # across DSV3/DSV2/Qwen down-projection wgrad shapes.
        "TRANS_FREE": True,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    }

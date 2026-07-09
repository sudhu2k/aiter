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

__all__ = ["flydsl_moe_wgrad", "flydsl_moe_wgrad_autotuned", "WGRAD_BLOCK_M"]


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


# Tile configs the autotuner sweeps (block_n, block_k, warps_n, warps_k). Mirrors the
# set in op_tests/flydsl_tests/bench_moe_wgrad_flydsl_v2.py; configs that violate the
# compile-time shape constraints for a given problem are skipped by the tuner.
_AUTOTUNE_TILES = [
    (64, 64, 1, 1),
    (128, 64, 1, 1),
    (128, 128, 2, 2),
    (128, 128, 4, 2),
    (256, 128, 4, 2),
    (256, 256, 4, 4),
    (128, 256, 2, 4),
    (256, 128, 2, 2),
]

_wgrad_autotuner = None


def _wgrad_run(
    dw,
    x,
    grad,
    topk_w,
    sorted_token_ids,
    block_start,
    blocks_per_expert,
    N,
    K,
    num_valid_tokens,
    top_k,
    num_experts,
    mul_routed_weight,
    block_n=128,
    block_k=128,
    warps_n=2,
    warps_k=2,
):
    """Dispatch target for the FlyDSL autotuner: compile (lru-cached) + launch one tile."""
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


def _get_autotuner(warmup=10, rep=30):
    """Build the shape-keyed Autotuner lazily (one instance, disk-cached results)."""
    global _wgrad_autotuner
    if _wgrad_autotuner is None:
        from flydsl.autotune import Autotuner, Config

        configs = [
            Config(block_n=bn, block_k=bk, warps_n=wn, warps_k=wk)
            for (bn, bk, wn, wk) in _AUTOTUNE_TILES
        ]
        # Key on the GEMM problem: x.shape=(M,K), grad-feature N, top_k, and the
        # mul_routed_weight flag (it selects a different compiled kernel). dtypes of
        # tensor args are folded in automatically by the tuner.
        _wgrad_autotuner = Autotuner(
            _wgrad_run,
            configs,
            key=["x", "N", "top_k", "mul_routed_weight"],
            warmup=warmup,
            rep=rep,
        )
    return _wgrad_autotuner


def flydsl_moe_wgrad_autotuned(
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
) -> None:
    """Shape-autotuned variant of :func:`flydsl_moe_wgrad`.

    First call for a given ``(x.shape, N, top_k, mul_routed_weight)`` benchmarks every
    tile in ``_AUTOTUNE_TILES`` and caches the fastest (in-memory + on disk under
    ``~/.flydsl/autotune/``); later calls reuse it with no benchmarking overhead. The
    wgrad launch is idempotent for a fixed input, so no ``reset_to_zero`` is needed and
    the benchmark timing stays free of memset noise.
    """
    num_experts, N, K = dw.shape
    num_valid_tokens = topk_ids.numel()

    assert x.dtype == grad.dtype == dw.dtype == torch.bfloat16
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()

    topk_w = topk_weights.reshape(-1).to(torch.float32)
    if not topk_w.is_contiguous():
        topk_w = topk_w.contiguous()

    _get_autotuner()(
        dw,
        x,
        grad,
        topk_w,
        sorted_token_ids,
        block_start,
        blocks_per_expert,
        int(N),
        int(K),
        int(num_valid_tokens),
        int(top_k),
        int(num_experts),
        bool(mul_routed_weight),
    )

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE weight-gradient (wgrad) op wrapper.

Mirrors the route-list Triton ``fused_route_list_moe_wgrad`` contract (see
``transformer_engine/pytorch/triton_kernels/route_list_moe_wgrad.py``) so the same
routing metadata (``sorted_slot_ids`` holding the received-token row per slot, plus
``block_start`` / ``blocks_per_expert`` / ``route_start``) can be reused verbatim.
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
    sorted_slot_ids: torch.Tensor,
    block_start: torch.Tensor,
    blocks_per_expert: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
    block_n: int = 128,
    block_k: int = 128,
    warps_n: int = 2,
    warps_k: int = 2,
) -> None:
    """Compute ``dw[e] += grad[route]^T @ x[token(route)]`` grouped by expert, in place.

    See ``fused_route_list_moe_wgrad`` for the argument contract: ``x`` is
    ``[num_recv_tokens, K]`` (gathered by received-token row), ``grad`` is the compact
    ``[num_routes, N]`` per-route gradient, ``sorted_slot_ids`` maps each block-padded
    route slot to its received-token row (sentinel ``num_recv_tokens`` for padding), and
    ``route_start[e]`` is the compact first-route index of expert ``e``.

    ``block_n``/``block_k`` and ``warps_n``/``warps_k`` select the workgroup tile; the
    defaults (``128x128`` over ``2x2`` warps) are a strong general config on CDNA4.
    """
    num_experts, N, K = dw.shape

    assert x.dtype == grad.dtype == dw.dtype == torch.bfloat16
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()

    exe = compile_moe_wgrad_v2(
        dtype="bf16",
        block_n=int(block_n),
        block_k=int(block_k),
        warps_n=int(warps_n),
        warps_k=int(warps_k),
    )

    _run_compiled(
        exe,
        ptr_arg(dw),
        ptr_arg(x),
        ptr_arg(grad),
        ptr_arg(sorted_slot_ids),
        ptr_arg(block_start),
        ptr_arg(blocks_per_expert),
        ptr_arg(route_start),
        int(N),
        int(K),
        int(num_recv_tokens),
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
    sorted_slot_ids,
    block_start,
    blocks_per_expert,
    route_start,
    N,
    K,
    num_recv_tokens,
    num_experts,
    block_n=128,
    block_k=128,
    warps_n=2,
    warps_k=2,
):
    """Dispatch target for the FlyDSL autotuner: compile (lru-cached) + launch one tile."""
    exe = compile_moe_wgrad_v2(
        dtype="bf16",
        block_n=int(block_n),
        block_k=int(block_k),
        warps_n=int(warps_n),
        warps_k=int(warps_k),
    )
    _run_compiled(
        exe,
        ptr_arg(dw),
        ptr_arg(x),
        ptr_arg(grad),
        ptr_arg(sorted_slot_ids),
        ptr_arg(block_start),
        ptr_arg(blocks_per_expert),
        ptr_arg(route_start),
        int(N),
        int(K),
        int(num_recv_tokens),
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
        # Key on the GEMM problem: x.shape=(num_recv_tokens, K), grad-feature N, and the
        # expert count. dtypes of tensor args are folded in automatically by the tuner.
        _wgrad_autotuner = Autotuner(
            _wgrad_run,
            configs,
            key=["x", "N", "num_experts"],
            warmup=warmup,
            rep=rep,
        )
    return _wgrad_autotuner


def flydsl_moe_wgrad_autotuned(
    x: torch.Tensor,
    grad: torch.Tensor,
    dw: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    block_start: torch.Tensor,
    blocks_per_expert: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
) -> None:
    """Shape-autotuned variant of :func:`flydsl_moe_wgrad`.

    First call for a given ``(x.shape, N, num_experts)`` benchmarks every tile in
    ``_AUTOTUNE_TILES`` and caches the fastest (in-memory + on disk under
    ``~/.flydsl/autotune/``); later calls reuse it with no benchmarking overhead. The
    wgrad launch is idempotent for a fixed input, so no ``reset_to_zero`` is needed and
    the benchmark timing stays free of memset noise.
    """
    num_experts, N, K = dw.shape

    assert x.dtype == grad.dtype == dw.dtype == torch.bfloat16
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()

    _get_autotuner()(
        dw,
        x,
        grad,
        sorted_slot_ids,
        block_start,
        blocks_per_expert,
        route_start,
        int(N),
        int(K),
        int(num_recv_tokens),
        int(num_experts),
    )

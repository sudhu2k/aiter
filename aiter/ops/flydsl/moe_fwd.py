# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE route-list forward gather-GEMM op wrapper.

Mirrors the Triton ``fused_route_list_moe`` forward contract (see
``transformer_engine/pytorch/triton_kernels/route_list_moe_gemm.py``) so the same routing
metadata (``sorted_slot_ids`` / ``expert_ids`` / ``block_start`` / ``route_start``) can be
reused verbatim. Writes the compact ``[em_max, WIDTH_N]`` route output in place.

Supports the forward gather paths (FC1 fwd with fused gated silu/gelu + optional route-prob +
optional pre-activation save, and FC2 fwd route-read). The transposed-weight dgrad path
(non-unit ``stride_bk``) is not handled here and should use the Triton kernel.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .kernels.moe_fwd_flydsl import compile_moe_fwd, ACT_SILU, ACT_GELU
from .kernels.tensor_shim import ptr_arg, _run_compiled

__all__ = ["flydsl_moe_fwd", "flydsl_moe_fwd_autotuned", "flydsl_moe_fwd_supported"]

_ACT_IDS = {"silu": ACT_SILU, "gelu": ACT_GELU}

# MFMA + fill constants (must match kernels/moe_fwd_flydsl.py).
_WMMA = 16
_FILL_V = 8
_WARP = 64
_LDS_PAD = 8
_LDS_LIMIT = 163840  # gfx950 per-workgroup LDS (160 KB)

_WARP_CHOICES = [1, 2, 4, 8, 16]


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _warp_valid(block_m, block_n, block_k, wm, wn):
    n_threads = wm * wn * _WARP
    if block_m % (wm * _WMMA) or block_n % (wn * _WMMA):
        return False
    if (block_m * block_k) % (n_threads * _FILL_V):
        return False
    if (block_n * block_k) % (n_threads * _FILL_V):
        return False
    return True


def _pick_warps(block_m: int, block_n: int, block_k: int):
    """Pick (warps_m, warps_n) balancing per-warp MFMA tile (M_STEPS x N_STEPS) vs occupancy.

    Prefers keeping the per-warp atom counts moderate (fewer accumulators -> more waves) while
    landing a 256-512 thread workgroup, which measured fastest across the Qwen MoE shapes.
    """
    best = None
    for wm in _WARP_CHOICES:
        for wn in _WARP_CHOICES:
            if not _warp_valid(block_m, block_n, block_k, wm, wn):
                continue
            n_threads = wm * wn * _WARP
            if n_threads > 512:
                continue
            m_steps = block_m // (wm * _WMMA)
            n_steps = block_n // (wn * _WMMA)
            # Favor a small, balanced per-warp atom footprint (fewer accumulators -> more
            # waves), then a 256-512 thread workgroup. Ties broken toward |m_steps-n_steps|
            # small (balanced reuse of A and B fragments).
            score = (
                m_steps * n_steps,
                abs(m_steps - n_steps),
                0 if 256 <= n_threads <= 512 else 1,
                n_threads,
            )
            if best is None or score < best[0]:
                best = (score, (wm, wn))
    return best[1] if best is not None else None


def _fwd_buffering():
    """(num_buffers, lds_pad) for the production DMA+swizzle fill path.

    Both default on (opt out with ``MOE_FWD_DMA=0`` / ``MOE_FWD_SWZ=0``). The DMA path runs
    a distance-2, 3-buffer ring; the register fallback keeps 2-buffer ping/pong.
    """
    use_dma = _env_flag("MOE_FWD_DMA", True)
    swz = _env_flag("MOE_FWD_SWZ", True)
    pad = 0 if (use_dma or swz) else _LDS_PAD
    return (3 if use_dma else 2), pad


def _lds_bytes(block_m, block_n, block_k, gated, transpose_b=False, gated_a=False):
    n_bt = 2 if gated else 1
    nbuf, pad = _fwd_buffering()
    a_tile = block_m * (block_k + pad)
    # Hybrid gated_a DMA path stages the raw ``up`` half in a parallel A tile (2x A LDS).
    hybrid_a = gated_a and _env_flag("MOE_FWD_DMA", True) and _env_flag("MOE_FWD_GATEDA_DMA", False)
    a_tiles = 2 if hybrid_a else 1
    # dgrad stages B as [k, n] (row stride = block_n+pad); fwd as [n, k] (block_k+pad).
    b_tile = block_k * (block_n + pad) if transpose_b else block_n * (block_k + pad)
    return (a_tiles * a_tile + n_bt * b_tile) * nbuf * 2


def _default_block_n(block_m, block_k, gated, transpose_b=False):
    """Widest N tile (128 then 64) that fits LDS -- wider N raises arithmetic intensity."""
    for bn in (128, 64):
        if _lds_bytes(block_m, bn, block_k, gated, transpose_b) <= _LDS_LIMIT and bn % _WMMA == 0:
            return bn
    return 64


def flydsl_moe_fwd_supported(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    block_m: int,
    block_n: int = 64,
    block_k: int = 64,
) -> bool:
    """Whether the FlyDSL fwd kernel can handle these operands (else use the Triton kernel)."""
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        return False
    if A.stride(1) != 1:  # A must be contiguous along the contraction (K)
        return False
    # B contiguous along K (fwd) or along N (dgrad transposed-weight view).
    if B.stride(2) != 1 and B.stride(1) != 1:
        return False
    K = B.shape[2]
    if K % block_k != 0:
        return False
    return _pick_warps(block_m, block_n, block_k) is not None


def flydsl_moe_fwd(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_start: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
    block_m: int,
    block_n: Optional[int] = None,
    block_k: int = 64,
    warps_m: Optional[int] = None,
    warps_n: Optional[int] = None,
    index_a_by_route_pos: bool = False,
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    preact_out: Optional[torch.Tensor] = None,
    gated_a: bool = False,
) -> None:
    """Route-list gather-GEMM forward, writing the compact ``C[em_max, WIDTH_N]`` in place.

    ``A`` is ``[*, K]`` (received-token acts, gathered by ``sorted_slot_ids`` when
    ``index_a_by_route_pos=False``, else read at the compact route row). ``B`` is
    ``[num_experts, N_OUT, K]`` (contiguous inner ``K``). With ``activation`` set the fused
    **gated** epilogue (FC1, ``gated_a=False``) applies ``act(gate) * up`` over ``N_OUT = 2F``
    into the ``F``-wide ``C``; ``dispatched_probs`` multiplies the per-route prob after the
    activation, and ``preact_out`` (``[em_max, 2F]``) saves the raw ``[gate | up]``
    pre-activation. With ``gated_a=True`` (FC2) ``A`` is the raw ``[gate | up]`` pre-activation
    (width ``2F``), the prologue applies ``act(gate) * up [* prob]`` into an ``F``-wide tile,
    and the GEMM contracts over ``K = F`` against ``B[e, H, F]``.
    """
    assert A.dtype == B.dtype == C.dtype == torch.bfloat16
    assert A.stride(1) == 1, "A must be contiguous along the contraction (K)"
    # B is [E, N_OUT, K]: contiguous along K (fwd) or along N (dgrad transposed-weight view).
    transpose_b = B.stride(2) != 1
    if transpose_b:
        assert B.stride(1) == 1, "transposed B must be contiguous along N (dgrad view)"
        assert activation is None, "dgrad (transposed B) does not support fused activation"

    gated = activation is not None and not gated_a
    if gated_a:
        if activation is None:
            raise ValueError("gated_a requires activation ('silu' or 'gelu')")
        if not index_a_by_route_pos:
            raise ValueError("gated_a requires index_a_by_route_pos=True")
    if block_n is None:
        block_n = _default_block_n(block_m, block_k, gated or gated_a, transpose_b)

    if warps_m is not None and warps_n is not None:
        warps_m, warps_n = int(warps_m), int(warps_n)
    else:
        warps = _pick_warps(block_m, block_n, block_k)
        if warps is None:
            raise ValueError(f"no valid warp layout for block_m={block_m} block_n={block_n}")
        warps_m, warps_n = warps

    K = int(B.shape[2])
    N_OUT = int(B.shape[1])
    width_n = int(C.shape[1])
    if gated and N_OUT != 2 * width_n:
        raise ValueError(f"gated fwd expects N_OUT == 2*C.width, got {N_OUT} vs {width_n}")
    if gated_a and A.shape[1] != 2 * K:
        raise ValueError(
            f"gated_a expects A width == 2*B.K (2F preact), got A={A.shape[1]} vs 2*{K}"
        )

    mul_prob = dispatched_probs is not None
    save_preact = preact_out is not None
    if save_preact and not gated:
        raise ValueError("preact_out requires the FC1 gated epilogue (not gated_a)")
    act_id = _ACT_IDS[activation] if (gated or gated_a) else ACT_SILU

    probs = dispatched_probs
    if mul_prob:
        if probs.dtype != torch.float32:
            probs = probs.to(torch.float32)
        stride_pm, stride_pe = int(probs.stride(0)), int(probs.stride(1))
    else:
        probs = C  # dummy (unread)
        stride_pm = stride_pe = 0

    preact = preact_out if save_preact else C  # dummy (unread) when not saving

    num_m_blocks = int(sorted_slot_ids.shape[0]) // block_m

    # The direct-B v2 path (contiguous-weight LDS staging, DMA+swizzle) is the default.
    # Set ``AITER_MOE_FWD_KERNEL=v1`` to fall back to the legacy route-list implementation.
    if os.getenv("AITER_MOE_FWD_KERNEL", "").lower() in ("v1", "legacy", "og"):
        compiler = compile_moe_fwd
    else:
        from .kernels.moe_fwd_flydsl_v2 import compile_moe_fwd_v2

        compiler = compile_moe_fwd_v2

    exe = compiler(
        dtype="bf16",
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        warps_m=int(warps_m),
        warps_n=int(warps_n),
        gated=bool(gated),
        gated_a=bool(gated_a),
        activation=int(act_id),
        mul_prob=bool(mul_prob),
        save_preact=bool(save_preact),
        index_a_by_route_pos=bool(index_a_by_route_pos),
        transpose_b=bool(transpose_b),
    )

    _run_compiled(
        exe,
        ptr_arg(A),
        ptr_arg(B),
        ptr_arg(C),
        ptr_arg(sorted_slot_ids),
        ptr_arg(expert_ids),
        ptr_arg(block_start),
        ptr_arg(route_start),
        ptr_arg(probs),
        ptr_arg(preact),
        int(K),
        int(N_OUT),
        int(width_n),
        int(num_recv_tokens),
        int(A.stride(0)),
        int(B.stride(0)),
        int(B.stride(1)),
        int(B.stride(2)),
        int(stride_pm),
        int(stride_pe),
        int(num_m_blocks),
        torch.cuda.current_stream(),
    )


# Tile/warp configs the autotuner sweeps: (block_n, block_k, warps_m, warps_n). Fill path
# (DMA+swizzle, 3-buffer) is fixed at the env defaults above -- only tile geometry is tuned.
#
# Two shapes matter for the wide-M (block_m=256) MoE cases, where every block_k=128 and
# 256x64 entry below is filtered out by the LDS limit, leaving only 128x64 candidates:
#   * a square-ish w4x4 warp grid, which spreads the cooperative A-fill (2 fills) / B-fill
#     (1 fill) DMA and the LDS fragment reads more evenly than the tall w8x2 layout;
#   * block_n=256 with block_k=32, which fits LDS at 3 buffers and raises arithmetic
#     intensity to block_m*block_n/(2*(block_m+block_n)) = 64 MAC/byte vs 42.7 at block_n=128
#     (the ratio is independent of block_k, so widening N is what pays).
# Measured on FC1 no-act (qwen235b, block_m=256), idle machine, alias scopes on:
# 256x32 w4x4 ~1628us, 128x64 w4x4 ~1694us, 128x64 w8x2 ~1726us.
#
# An exhaustive sweep of the valid space (all block_n in 64..512 x block_k in 32..256 x warp
# grids >=4 waves; benchmarks/microbenchmarks/sweep_fc1_configs.py) found nothing better, so
# the list below is not missing a winner. Two directions are dead ends and are deliberately
# absent: block_n>=384 costs 4-22x (per-wave accumulator spill plus 120-144KB LDS pinning
# occupancy to 1 workgroup), and trading block_m down to reach block_k=128 -- 4x fewer
# barriers, which the ATT trace makes look attractive -- costs 59% (2581us at block_m=128
# bk=128) because arithmetic intensity falls faster than barrier count. Note block_k>=128 does
# not fit LDS at all once NUM_BUF>=3, which the kernel enforces.
#
# Do not add 256x384x32 w1x4 or 512x32 w1x4 / w2x2: they abort the backend outright
# ("Bad machine code: Virtual register defs don't dominate all uses"), which would take the
# autotuner down with them rather than being skipped. Pre-existing, unrelated to alias scopes.
_FWD_TUNE_CONFIGS = [
    (64, 64, 2, 2),
    (128, 64, 2, 2),
    (128, 64, 4, 2),
    (128, 64, 2, 4),
    (128, 64, 4, 4),
    (128, 64, 8, 2),
    (256, 32, 4, 4),
    (256, 64, 4, 2),
    (128, 128, 2, 2),
    (128, 128, 4, 2),
    (64, 128, 2, 2),
    (256, 64, 2, 4),
]

# Winner cache: {shape/mode key -> (block_n, block_k, warps_m, warps_n)}.
_FWD_CACHE: dict = {}


def _valid_config(block_m, bn, bk, wm, wn, K, gated, gated_a=False):
    if K % bk != 0 or bn % _WMMA != 0:
        return False
    if not _warp_valid(block_m, bn, bk, wm, wn):
        return False
    return _lds_bytes(block_m, bn, bk, gated, gated_a=gated_a) <= _LDS_LIMIT


def flydsl_moe_fwd_autotuned(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_start: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
    block_m: int,
    block_k: int = 64,
    index_a_by_route_pos: bool = False,
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    preact_out: Optional[torch.Tensor] = None,
    gated_a: bool = False,
    warmup: int = 3,
    iters: int = 10,
) -> None:
    """Shape-autotuned :func:`flydsl_moe_fwd`.

    On the first call for a given (block_m, GEMM shape, epilogue mode) the valid subset of
    ``_FWD_TUNE_CONFIGS`` is benchmarked and the fastest ``(block_n, block_k, warps_m, warps_n)``
    is cached. The production DMA+swizzle fill path is always used; only tile geometry is swept.
    """
    gated = activation is not None and not gated_a
    N_OUT, K = int(B.shape[1]), int(B.shape[2])
    width_n = int(C.shape[1])
    key = (
        int(block_m), N_OUT, K, width_n, bool(gated), bool(gated_a),
        activation, dispatched_probs is not None, preact_out is not None,
        bool(index_a_by_route_pos),
    )

    def _launch(bn, bk, wm, wn):
        flydsl_moe_fwd(
            A, B, C, sorted_slot_ids, expert_ids, block_start, route_start,
            num_recv_tokens=num_recv_tokens, block_m=block_m, block_n=bn, block_k=bk,
            warps_m=wm, warps_n=wn, index_a_by_route_pos=index_a_by_route_pos,
            activation=activation, dispatched_probs=dispatched_probs, preact_out=preact_out,
            gated_a=gated_a,
        )

    best = _FWD_CACHE.get(key)
    if best is None:
        candidates = [
            (bn, bk, wm, wn)
            for (bn, bk, wm, wn) in _FWD_TUNE_CONFIGS
            if _valid_config(block_m, bn, bk, wm, wn, K, gated, gated_a)
        ]
        if not candidates:
            _launch(None, block_k, None, None)  # heuristic fallback
            return
        best_t = None
        for cfg in candidates:
            try:
                for _ in range(warmup):
                    _launch(*cfg)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                for _ in range(iters):
                    _launch(*cfg)
                end.record()
                torch.cuda.synchronize()
                t = start.elapsed_time(end) / iters
            except Exception:  # noqa: BLE001 -- skip configs that fail to compile/run
                continue
            if best_t is None or t < best_t:
                best_t, best = t, cfg
        if best is None:
            _launch(None, block_k, None, None)
            return
        _FWD_CACHE[key] = best

    _launch(*best)

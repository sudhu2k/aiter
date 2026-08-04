# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE weight-gradient (wgrad) op wrapper.

Mirrors the route-list Triton ``fused_route_list_moe_wgrad`` contract (see
``transformer_engine/pytorch/triton_kernels/route_list_moe_wgrad.py``) so the same
routing metadata (``sorted_slot_ids`` holding the received-token row per slot, plus
``block_start`` / ``blocks_per_expert`` / ``route_start``) can be reused verbatim.
"""

from __future__ import annotations

import os

import torch

from .kernels.moe_wgrad_flydsl_v2 import compile_moe_wgrad_v2, WGRAD_BLOCK_M
from .kernels.tensor_shim import ptr_arg, _run_compiled

__all__ = ["flydsl_moe_wgrad", "flydsl_moe_wgrad_autotuned", "WGRAD_BLOCK_M"]


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v.strip() != "" else default


# Best-performing wgrad fill/pipeline defaults on CDNA4 (per the DMA+swizzle 3-stage
# sweep: ~+7-12% over the padded baseline on the Qwen shapes). These are the fill/
# pipeline knobs only -- the workgroup *tile* is still autotuned per shape. Override
# via env:
#   AITER_WGRAD_DMA_SWIZZLE  global->LDS DMA + XOR swizzle fill      (default on)
#   AITER_WGRAD_PIPE_STAGES  LDS pipeline depth (3 = triple buffer)  (default 3)
_WGRAD_DMA_SWIZZLE = _env_flag("AITER_WGRAD_DMA_SWIZZLE", True)
_WGRAD_PIPE_STAGES = _env_int("AITER_WGRAD_PIPE_STAGES", 3)


def _resolve_dma_opts(swap_gather: bool):
    """Resolve the (dma_swizzle, pipe_stages) pair from env defaults.

    The DMA+swizzle fill supports both FC1 and FC2 (``swap_gather``): the token-gathered
    operand simply moves from ``x`` to ``grad``. ``pipe_stages`` > 2 collapses to 2 whenever
    DMA is off so the kernel never hits an invalid combo.
    """
    dma = _WGRAD_DMA_SWIZZLE
    stages = _WGRAD_PIPE_STAGES if dma else 2
    return dma, stages


def _v3_wgrad_active() -> bool:
    """Whether the MegaMOE-port v3 wgrad kernel is selected.

    Gated behind its *own* ``AITER_MOE_FLYDSL_V3_WGRAD`` flag (NOT the shared
    ``AITER_MOE_FLYDSL_V3`` that drives fwd/dgrad): the wgrad port is numerically exact but
    ~0.5-0.65x of the tuned v2 kernel because its gather is on the contraction (token) axis --
    ``SORTED`` indices reload every K-step and the X fetch is non-coalesced, unlike the
    K-invariant fwd/dgrad row gather. So v3 fwd/dgrad stay a win while wgrad keeps v2 by default.
    """
    return _env_flag("AITER_MOE_FLYDSL_V3_WGRAD", False)


def _try_run_v3_wgrad(
    x, grad, dw, sorted_slot_ids, block_start, blocks_per_expert, route_start,
    num_recv_tokens, accumulate, out_dtype, swap_gather,
) -> bool:
    """Dispatch to the v3 (MegaMOE variable-K) wgrad kernel when it applies; else ``False``.

    The v3 kernel is the FC1 wgrad path only (``X`` gathered on the contraction, compact grad),
    bf16 overwrite. Unsupported requests (FC2 ``swap_gather``, fp32/accumulate epilogue, or a dW
    whose feature dims don't tile the fixed 256x256 geometry) fall through to the v2 kernel. The
    routing metadata must be 256-aligned (TE's ``_wgrad_contract_m`` handles this under the flag).
    """
    if not _v3_wgrad_active():
        return False
    if swap_gather or accumulate or out_dtype != "bf16":
        return False
    E, N, K = dw.shape
    from .kernels.moe_wgrad_flydsl_v3 import grouped_gemm_wgrad_gather_bf16, WGRAD_ALIGN

    if K % 256 != 0 or N % 256 != 0:
        return False
    grouped_gemm_wgrad_gather_bf16(
        x,
        grad,
        sorted_slot_ids,
        block_start,
        blocks_per_expert,
        route_start,
        num_recv_tokens=int(num_recv_tokens),
        out=dw,
        out_dtype=torch.bfloat16,
    )
    return True


def _resolve_out_dtype(dw: torch.Tensor, out_dtype):
    """Pick the kernel ``out_dtype`` ('bf16'/'fp32') and validate it against ``dw``."""
    if out_dtype is None:
        out_dtype = "fp32" if dw.dtype == torch.float32 else "bf16"
    if out_dtype == "fp32":
        assert dw.dtype == torch.float32, "out_dtype='fp32' requires a float32 dw buffer"
    elif out_dtype == "bf16":
        assert dw.dtype == torch.bfloat16, "out_dtype='bf16' requires a bfloat16 dw buffer"
    else:
        raise ValueError(f"out_dtype must be 'bf16' or 'fp32', got {out_dtype!r}")
    return out_dtype


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
    accumulate: bool = False,
    out_dtype: str | None = None,
    swap_gather: bool = False,
    dma_swizzle: bool | None = None,
    pipe_stages: int | None = None,
) -> None:
    """Compute the grouped wgrad ``grad[route]^T @ x[token(route)]`` into ``dw``, per expert.

    See ``fused_route_list_moe_wgrad`` for the argument contract: ``x`` is
    ``[num_recv_tokens, K]`` (gathered by received-token row), ``grad`` is the compact
    ``[num_routes, N]`` per-route gradient, ``sorted_slot_ids`` maps each block-padded
    route slot to its received-token row (sentinel ``num_recv_tokens`` for padding), and
    ``route_start[e]`` is the compact first-route index of expert ``e``.

    ``block_n``/``block_k`` and ``warps_n``/``warps_k`` select the workgroup tile; the
    defaults (``128x128`` over ``2x2`` warps) are a strong general config on CDNA4.

    Output modes (each dW element is written by exactly one workgroup, so accumulation is a
    race-free read-modify-write -- no atomics):
      * ``accumulate=False`` (default): overwrite ``dw`` (``dw[e] = grad^T @ x``).
      * ``accumulate=True``: add into ``dw`` (``dw[e] += grad^T @ x``) -- lets the caller
        fold wgrad straight into a param's ``main_grad`` / ``.grad`` without a separate add.
    ``out_dtype`` ('bf16' or 'fp32', inferred from ``dw`` when ``None``) picks the store
    precision, so an fp32 ``main_grad`` accumulator can be targeted directly.

    ``swap_gather`` moves the ``SORTED`` token-gather from the ``x`` (K) operand to the ``grad``
    (N) operand: ``x`` is then read by route position from a compact ``[num_routes, K]`` buffer
    and ``grad`` is token-gathered from a ``[num_recv, N]`` buffer. This is the FC2 wgrad case --
    passing ``x=fc2_input`` (route-ordered) and ``grad=grad_output`` (token-space) makes the
    kernel emit ``dW2`` in ``[E, out, in]`` directly, with no transpose post-pass.
    """
    num_experts, N, K = dw.shape

    assert x.dtype == grad.dtype == torch.bfloat16
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()
    out_dtype = _resolve_out_dtype(dw, out_dtype)

    if _try_run_v3_wgrad(
        x, grad, dw, sorted_slot_ids, block_start, blocks_per_expert, route_start,
        num_recv_tokens, accumulate, out_dtype, swap_gather,
    ):
        return

    # Fill/pipeline knobs default to the env-configured best config; explicit args win.
    _dma, _stages = _resolve_dma_opts(bool(swap_gather))
    if dma_swizzle is not None:
        _dma = bool(dma_swizzle)
        _stages = _stages if _dma else 2
    if pipe_stages is not None:
        _stages = int(pipe_stages)

    exe = compile_moe_wgrad_v2(
        dtype="bf16",
        block_n=int(block_n),
        block_k=int(block_k),
        warps_n=int(warps_n),
        warps_k=int(warps_k),
        accumulate=bool(accumulate),
        out_dtype=out_dtype,
        swap_gather=bool(swap_gather),
        dma_swizzle=_dma,
        pipe_stages=_stages,
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


# Tile configs the autotuner sweeps (block_n, block_k, warps_n, warps_k). Fill path
# (DMA+swizzle, 3-stage) is fixed at the env defaults above -- only tile geometry is tuned.
_AUTOTUNE_TILES = [
    (64, 64, 1, 1),
    (128, 64, 1, 1),
    (128, 128, 2, 2),
    (128, 128, 4, 2),
    (256, 128, 4, 2),
    (256, 256, 4, 4),
    (128, 256, 2, 4),
    (256, 128, 2, 2),
    # 256x256 at 8 waves (512 threads): the Mega variable-K geometry. On DeepSeek-V3 FC1
    # wgrad this hits ~999 TF (88.5% of Mega's ~1131) vs the 128x128/w2x2 config's ~911 TF --
    # the 1024-thread (256,256,4,4) above instead collapses (occupancy). Added so the tuner can
    # pick the 8-wave 256x256 on the large FC shapes where it dominates.
    (256, 256, 2, 4),
    (256, 256, 4, 2),
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
    swap_gather=False,
    block_n=128,
    block_k=128,
    warps_n=2,
    warps_k=2,
):
    """Dispatch target for the FlyDSL autotuner: compile (lru-cached) + launch one tile.

    Uses the env-configured fill/pipeline defaults so the tile sweep is benchmarked in
    the same mode the production launch will run (the best tile depends on it -- the DMA
    3-stage path favors the large 256x256 tile). ``swap_gather`` must match the production
    launch: it flips which operand carries the token-gather, so tuning it in the wrong mode
    route-walks the compact operand out of bounds (FC2 has ``routes`` >> ``num_recv_tokens``).
    """
    _dma, _stages = _resolve_dma_opts(bool(swap_gather))
    exe = compile_moe_wgrad_v2(
        dtype="bf16",
        block_n=int(block_n),
        block_k=int(block_k),
        warps_n=int(warps_n),
        warps_k=int(warps_k),
        swap_gather=bool(swap_gather),
        dma_swizzle=_dma,
        pipe_stages=_stages,
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
        # Key on the GEMM problem: x.shape=(num_recv_tokens, K), grad-feature N, the
        # expert count, and the gather mode (FC1 vs FC2 route/token operand swap). dtypes
        # of tensor args are folded in automatically by the tuner.
        _wgrad_autotuner = Autotuner(
            _wgrad_run,
            configs,
            key=["x", "N", "num_experts", "swap_gather"],
            warmup=warmup,
            rep=rep,
        )
    return _wgrad_autotuner


def _select_wgrad_config(
    x, grad, sorted_slot_ids, block_start, blocks_per_expert, route_start,
    N, K, num_recv_tokens, num_experts, swap_gather=False,
):
    """Return the autotuned ``(block_n, block_k, warps_n, warps_k)`` for this problem.

    Tuning benchmarks the plain *overwrite* bf16 kernel into a throwaway scratch buffer, so
    it never touches (and never zeroes) a real accumulation target. The tile choice is
    independent of the epilogue's accumulate/out_dtype, so the same cached config drives the
    accumulate/fp32 launch. Reuses the shared on-disk cache keyed on ``(x.shape, N, E)``.
    """
    tuner = _get_autotuner()
    scratch = torch.empty(num_experts, N, K, device=x.device, dtype=torch.bfloat16)
    args = (
        scratch, x, grad, sorted_slot_ids, block_start, blocks_per_expert, route_start,
        int(N), int(K), int(num_recv_tokens), int(num_experts), bool(swap_gather),
    )
    key = tuner._make_key(args, {})
    if key not in tuner.cache:
        tuner(*args)  # one-time benchmark into scratch, populates the cache
    cfg = tuner.cache[key].kwargs
    return cfg["block_n"], cfg["block_k"], cfg["warps_n"], cfg["warps_k"]


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
    accumulate: bool = False,
    out_dtype: str | None = None,
    swap_gather: bool = False,
) -> None:
    """Shape-autotuned variant of :func:`flydsl_moe_wgrad`.

    First call for a given ``(x.shape, N, num_experts)`` benchmarks every tile in
    ``_AUTOTUNE_TILES`` and caches the fastest (in-memory + on disk under
    ``~/.flydsl/autotune/``). The production DMA+swizzle fill path is always used; only
    tile geometry is swept.
    """
    num_experts, N, K = dw.shape

    assert x.dtype == grad.dtype == torch.bfloat16
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()
    out_dtype = _resolve_out_dtype(dw, out_dtype)

    if _try_run_v3_wgrad(
        x, grad, dw, sorted_slot_ids, block_start, blocks_per_expert, route_start,
        num_recv_tokens, accumulate, out_dtype, swap_gather,
    ):
        return

    if not accumulate and out_dtype == "bf16" and not swap_gather:
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
        return

    block_n, block_k, warps_n, warps_k = _select_wgrad_config(
        x, grad, sorted_slot_ids, block_start, blocks_per_expert, route_start,
        N, K, num_recv_tokens, num_experts, swap_gather=swap_gather,
    )
    flydsl_moe_wgrad(
        x,
        grad,
        dw,
        sorted_slot_ids,
        block_start,
        blocks_per_expert,
        route_start,
        num_recv_tokens=int(num_recv_tokens),
        block_n=block_n,
        block_k=block_k,
        warps_n=warps_n,
        warps_k=warps_k,
        accumulate=accumulate,
        out_dtype=out_dtype,
        swap_gather=swap_gather,
    )

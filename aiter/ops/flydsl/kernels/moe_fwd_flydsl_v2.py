# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE route-list forward gather-GEMM in FlyDSL (bf16), v2 entry point.

``compile_moe_gemm1`` here is the route-list-contract two-stage MoE GEMM: it takes the exact
metadata/layout that ``route_list_moe_gemm.py`` and ``aiter.ops.flydsl.moe_fwd`` use
(``sorted_slot_ids`` / ``expert_ids`` / ``block_start`` / ``route_start``, plain ``[E, N_OUT, K]``
bf16 weights, compact ``[num_routes, WIDTH_N]`` output). Unlike the generic vLLM-style
preshuffled-weight ``compile_moe_gemm1`` in ``moe_gemm_2stage.py`` (which requires physically
preshuffled weight bytes and is only wired for the quantized/inference dispatch), this copy
reads the plain contiguous weight layout and stages **both** operands through double-buffered
LDS -- the preshuffle register-resident B trick does not transfer to weights that change every
optimizer step, so B uses the same coalesced-fill + XOR16-swizzle LDS pipeline as A.

``compile_moe_fwd_v2`` and ``compile_moe_fwd`` are aliases of ``compile_moe_gemm1`` so the
``aiter.ops.flydsl.moe_fwd`` wrapper can select it via ``AITER_MOE_FWD_KERNEL=v2`` with no
signature change.

Mirrors the Triton ``fused_route_list_moe`` forward kernel
(``transformer_engine/pytorch/triton_kernels/route_list_moe_gemm.py``): a gather-in GEMM
that computes the compact per-route output

    C[out_row(pos), :] = A[a_row(pos), :] @ B[e]^T           (contraction over K)

for every block-padded route slot ``pos`` of expert ``e = expert_ids[pid_m]``, with

    out_row(pos) = route_start[e] + (pos - block_start[e] * BLOCK_M)
    a_row(pos)   = sorted_slot_ids[pos]          (fwd)   -- gather the received-token row
                 = out_row(pos)                  (dgrad) -- read the compact route row

Padding slots carry the sentinel ``sorted_slot_ids[pos] == num_recv_tokens`` and are masked
out of the store. Tail blocks (``expert_ids[pid_m] < 0``) exit early.

Unlike the wgrad kernel, the contraction axis (``K`` = ``in_features``) is the *contiguous*
inner dim of both ``A`` and ``B``, so it maps directly onto the MFMA per-lane K-fragment: the
LDS tiles are staged as ``[row, k]`` and read back with a plain ``ds_read`` (no transpose).

Fused gated-activation epilogue (FC1, ``GATED``)
------------------------------------------------
When ``GATED`` the GEMM output width is the gate+up width ``N_OUT = 2F``; the kernel computes
two ``[BLOCK_M, BLOCK_N]`` accumulators from a shared ``A`` tile -- one over the gate columns
``[0, F)`` and one over the up columns ``[F, 2F)`` -- so the matching (gate, up) pair for an
output feature lands in the *same lane* (no cross-lane shuffle). The epilogue emits
``act(gate) * up`` (``act`` = silu/gelu) into the ``F``-wide ``C``; ``MUL_PROB`` multiplies the
per-route gating prob in *after* the activation; ``SAVE_PREACT`` also stores the raw ``2F``
``[gate | up]`` pre-activation for the backward.

Fused gated-activation prologue (FC2, ``GATED_A``)
--------------------------------------------------
When ``GATED_A`` the ``A`` operand is the raw ``2F`` pre-activation ``[gate | up]`` (route
layout, ``index_a_by_route_pos=True``). Before the MFMA the gather stage loads matching
``(gate[k], up[k])`` pairs, applies ``act(gate) * up`` (and ``MUL_PROB`` when set), and stages
the resulting ``F``-wide tile into LDS. ``K`` is the contracted ``F`` width; ``B`` is
``[E, H, F]``. This lets FC1 emit raw ``2F`` only and moves activation (+ route prob) to FC2.

Supported: bf16, contiguous-inner ``A``/``B`` (``stride_ak == stride_bk == 1``), both
``INDEX_A_BY_ROUTE_POS`` modes (FC1 fwd gather, FC2 fwd route-read). The transposed-weight
dgrad path (non-unit ``stride_bk``) is left to the Triton kernel.
"""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import builtin as _builtin
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import ptr_rsrc

__all__ = ["compile_moe_gemm1", "compile_moe_fwd_v2", "compile_moe_fwd", "ACT_SILU", "ACT_GELU"]

# MFMA atom dims (16x16x32 bf16 on gfx950/gfx942).
WMMA_M = 16   # output rows -> route (m)
WMMA_N = 16   # output cols -> out feature (n)
WMMA_K = 32   # contraction -> hidden (k)
C_FRAG = 4    # f32 values per lane for the C fragment (4 rows)
WARP_SIZE = 64

# Coalesced fill vector width (bf16 elems). 8 bf16 = 16 B = one global_load_dwordx4.
FILL_V = 8
# LDS row-stride padding (bf16 elems) to break bank conflicts on the fragment read.
LDS_PAD = int(os.environ.get("MOE_FWD_LDS_PAD", "8"))


def _swz_elem(row, col, S):
    """XOR bank-swizzle of a [row, col] element in a row-major, unpadded (stride ``S``) tile.

    Permutes the ``FILL_V``-wide column-chunk index by the low bits of the row so successive
    rows of the same column map to different LDS banks -- a padding-free alternative to
    ``LDS_PAD`` that keeps a power-of-two row stride (needed for async DMA fills). The map is:

        chunk = col // FILL_V ;  chunk' = chunk XOR (row & (S//FILL_V - 1))
        phys  = row*S + chunk'*FILL_V + (col % FILL_V)

    It is bijective and preserves ``FILL_V``-element groups, so vec8 stores/loads and the
    4-wide ``ds_read_tr16_b64`` unit stay contiguous+aligned. Because ``ds_read_tr`` applies a
    fixed cross-lane transpose over the bytes each lane loads, applying the *same* map on the
    store and read sides leaves the transposed fragment unchanged. Requires ``S % FILL_V == 0``
    and ``S // FILL_V`` a power of two. ``row``/``col`` are ``fx.Int32``; ``S`` is a Python int.
    """
    mask = (S // FILL_V) - 1
    chunk_swz = (col >> 3) ^ (row & mask)          # FILL_V == 8 -> col // 8
    return row * fx.Int32(S) + chunk_swz * fx.Int32(FILL_V) + (col & 7)

def _swz_col(row, col, S):
    """Within-row part of :func:`_swz_elem`, i.e. ``_swz_elem(row, col, S) - row*S``.

    A register store applies the swizzle to the LDS *destination* address. An async DMA
    fill cannot address LDS per element -- it writes each lane's ``FILL_V`` chunk to a
    fixed lane-contiguous slot ``row*S + col`` -- so instead we bake the swizzle into the
    *global source column* the lane gathers: contiguous slot ``(row, col)`` pulls global
    column ``_swz_col(row, col, S)``. This makes the DMA-filled tile physically identical
    to the register-stored one, so the shared ``_swz_elem`` read path is correct for both.
    ``row``/``col`` are ``fx.Int32``; ``S`` is a Python int. Returns ``fx.Int32``.
    """
    mask = (S // FILL_V) - 1
    chunk_swz = (col >> 3) ^ (row & mask)
    return chunk_swz * fx.Int32(FILL_V) + (col & 7)

ACT_SILU = 0
ACT_GELU = 1

_LOG2E = 1.4426950408889634


def _exp2_f32(x):
    return _llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [x], [], [])


def _rcp_f32(x):
    return _llvm.call_intrinsic(T.f32, "llvm.amdgcn.rcp.f32", [x], [], [])


def _act_f32(g, act: int):
    """Scalar gated-activation nonlinearity in f32 (exp2/rcp-based, matches the Triton epilogue)."""
    f32 = T.f32
    one = arith.constant(1.0, type=f32)
    log2e = arith.constant(_LOG2E, type=f32)
    if act == ACT_SILU:
        # silu(g) = g * sigmoid(g) = g / (1 + exp2(-g*log2e)).
        t = arith.mulf(g, arith.constant(-_LOG2E, type=f32))
        sig = _rcp_f32(arith.addf(one, _exp2_f32(t)))
        return arith.mulf(g, sig)
    # gelu tanh approx: 0.5*g*(1 + tanh(inner)), inner = c*(g + 0.044715 g^3),
    # tanh(x) = 2*sigmoid(2x) - 1 = 2/(1 + exp2(-2x*log2e)) - 1.
    c = arith.constant(0.7978845608028654, type=f32)
    a = arith.constant(0.044715, type=f32)
    g2 = arith.mulf(g, g)
    g3 = arith.mulf(g2, g)
    inner = arith.mulf(c, arith.addf(g, arith.mulf(a, g3)))
    t2 = arith.mulf(arith.mulf(inner, arith.constant(-2.0, type=f32)), log2e)
    s = _rcp_f32(arith.addf(one, _exp2_f32(t2)))
    tanh = arith.subf(arith.mulf(arith.constant(2.0, type=f32), s), one)
    half_g = arith.mulf(arith.constant(0.5, type=f32), g)
    return arith.mulf(half_g, arith.addf(one, tanh))


def _gated_a_bf16_vec8(gate_v, up_v, prob, act: int, *, mul_prob: bool):
    """``act(gate) * up [* prob]`` elementwise over an 8-wide bf16 vector (FC2 A prologue)."""
    bf16 = T.bf16
    f32 = T.f32
    out_elems = []
    for i in range_constexpr(FILL_V):
        g = vector.extract(gate_v, static_position=[i], dynamic_position=[])
        u = vector.extract(up_v, static_position=[i], dynamic_position=[])
        gf = arith.extf(f32, g)
        uf = arith.extf(f32, u)
        val = arith.mulf(_act_f32(gf, act), uf)
        if mul_prob:
            val = arith.mulf(val, prob)
        out_elems.append(arith.truncf(bf16, val))
    return vector.from_elements(T.vec(FILL_V, bf16), out_elems)


@functools.lru_cache(maxsize=None)
def compile_moe_gemm1(
    *,
    dtype: str = "bf16",
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 64,
    warps_m: int = 2,
    warps_n: int = 2,
    gated: bool = False,
    gated_a: bool = False,
    activation: int = ACT_SILU,
    mul_prob: bool = False,
    save_preact: bool = False,
    index_a_by_route_pos: bool = False,
    transpose_b: bool = False,
):
    if dtype != "bf16":
        raise ValueError(f"moe_fwd flydsl kernel only supports bf16, got {dtype!r}")
    if block_m % (warps_m * WMMA_M) != 0 or block_n % (warps_n * WMMA_N) != 0:
        raise ValueError("block_m/block_n must be multiples of warps_*16")
    if block_k % WMMA_K != 0:
        raise ValueError(f"block_k must be a multiple of {WMMA_K}")
    if save_preact and not gated:
        raise ValueError("save_preact requires gated activation")
    if gated and gated_a:
        raise ValueError("gated (FC1 epilogue) and gated_a (FC2 prologue) are mutually exclusive")
    if transpose_b and (gated or gated_a):
        # dgrad (transposed weights) is never gated; keep the epilogues decoupled.
        raise ValueError("transpose_b (dgrad) does not support fused activation")
    if gated_a and not index_a_by_route_pos:
        raise ValueError("gated_a (FC2 prologue) requires index_a_by_route_pos=True")

    n_threads = warps_m * warps_n * WARP_SIZE
    if (block_m * block_k) % (n_threads * FILL_V) != 0:
        raise ValueError("block_m*block_k must be a multiple of n_threads*FILL_V")
    if (block_n * block_k) % (n_threads * FILL_V) != 0:
        raise ValueError("block_n*block_k must be a multiple of n_threads*FILL_V")

    gpu_arch = get_rocm_arch()

    # Async global->LDS fill (buffer_load ... lds DMA): collapses the
    # global_load -> s_waitcnt(vmcnt) -> ds_write chain into one op that writes
    # LDS directly, and lets the fill overlap the MFMA stream (waited on vmcnt at
    # the workgroup barrier) instead of exposing a post-compute ds_write. The DMA
    # writes LDS contiguously in lane order, which requires unpadded rows, so the
    # LDS tiles drop their bank-conflict padding in this mode.
    def _env_on(name: str, default: bool = True) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        return v.strip().lower() not in ("0", "off", "false", "no")

    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        return int(v) if v is not None and v.strip() != "" else default

    use_dma = _env_on("MOE_FWD_DMA", True)
    # FC2 gated_a prologue has two fill strategies:
    #  * register path (default): gather (gate, up) to VGPR, apply act(gate)*up, ds_write to LDS.
    #  * hybrid DMA path (MOE_FWD_GATEDA_DMA=1): keep the fast global->LDS DMA by staging the raw
    #    gate and up halves into two parallel LDS tiles, then fuse act(gate)*up on the LDS->VGPR
    #    read into the MFMA. Route prob is a per-route (M) scalar, so it factors out of the GEMM
    #    and is applied in the epilogue for *both* paths (out[m,n] *= prob[m]).
    hybrid_a = gated_a and use_dma and _env_on("MOE_FWD_GATEDA_DMA", False)
    # Experiment: emit the DMA barrier as ROCDL s_waitcnt/s_barrier intrinsics rather than an
    # inline-asm blob, so SIInsertWaitcnts can see the wait instead of treating it as opaque.
    _WAITCNT_INTRINSIC = _env_on("MOE_FWD_WAITCNT_INTRINSIC", False)
    # Experiment: build the global->LDS DMA destination as a GEP off the @smem object instead
    # of inttoptr(readfirstlane(byte_off)), giving it provenance the aliasing analysis can use.
    _DMA_GEP = _env_on("MOE_FWD_DMA_GEP", False)
    # Per-ring-slot LLVM alias scopes on both the LDS-DMA writes and the LDS operand reads.
    # ``SIInsertWaitcnts`` only takes its alias-aware LDS-DMA path for a read whose memory
    # operand carries AA info (``if (Ptr && Memop->getAAInfo())``); otherwise it waits on the
    # generic "any LDS DMA" slot, which is the blunt ``s_waitcnt vmcnt(0)`` in front of every
    # operand read. Provenance alone does not supply AA info -- the metadata does. Tagging the
    # fill of ring slot ``r`` and the read of ring slot ``r`` with the same scope (and noalias
    # against every other slot) keeps the real same-slot RAW dependency while dropping the
    # false cross-slot ones, so the pass can emit a graduated vmcnt. The read side must be an
    # ``llvm.load`` because ``vector.load`` has no alias-metadata operand. Recipe mirrors the
    # mxfp8 4-wave HK kernels (``flydsl_4wave_hk.py`` / ``kernel_grouped_4wave.py``).
    # ``transpose_b`` (dgrad) reads B through ``ds_read_tr16_b64``: it is scoped via the same
    # recipe as the wgrad v2 kernel -- the read is GEP-rooted off ``_lds_root`` for provenance
    # and the ``ds_read`` carries the ring-slot scope (see ``_tr_read_frag``). ``hybrid_a`` reads
    # a second parallel up-tile that would need scopes of its own, so it still falls back to the
    # unscoped reads rather than silently mixing a scoped write with an unscoped read (which is
    # still MayAlias, i.e. no gain, and would make the flag break that config).
    #
    # On by default (opt out with ``MOE_FWD_DMA_ALIAS=0``). ATT on FC1 no-act (qwen235b,
    # 256x256x32 w4x4) shows the ``s_waitcnt vmcnt`` stall bucket dropping 29.2% -> 8.0% of
    # stall cycles; the freed cycles are largely reabsorbed by the ds_read/DMA they were
    # hiding, so the net is ~1.9% (1660us -> 1628us, non-overlapping over 5 interleaved reps).
    # The gain scales with the number of K iterations, so it is a win at block_k=32 (-2.2%)
    # and roughly a wash to slightly negative at block_k=64 (+1%), where the two extra VGPRs
    # the scoped loads cost are not repaid.
    _DMA_ALIAS = (
        use_dma and not hybrid_a and _env_on("MOE_FWD_DMA_ALIAS", True)
    )
    if _DMA_ALIAS:
        # The DMA destination must be a GEP off the LDS object for the fill and the read to
        # share a base the aliasing analysis can compare.
        _DMA_GEP = True
    # Register-path deep VGPR load-ahead: issue the global gather two K-tiles ahead (carried in
    # VGPRs across the loop) instead of one, so the ``buffer_load`` latency overlaps a full
    # MFMA+store+barrier cycle rather than only the (short) MFMA burst. Keeps NUM_BUF==2 LDS
    # buffers (occupancy unchanged); costs one extra tile of gathered VGPRs. For ``gated_a`` the
    # *raw* gate/up halves are carried and the ``silu(gate)*up`` fuse is deferred to store time,
    # so the load is not immediately waited on by the activation's vector.extract.
    reg_pf2 = _env_on("MOE_FWD_REG_PF2", True)
    # Pipeline the gated_a activation: instead of applying silu(gate)*up as a serial burst in
    # front of the MFMA (store_a_vecs before compute), emit the per-descriptor silu+ds_write as
    # thunks interleaved across the MFMA groups inside ``compute`` so the (expensive) v_exp
    # transcendental co-issues on the VALU while the matrix pipe runs. The activated tile targets
    # the ``nxt`` LDS buffer (disjoint from the ``cur`` buffer the MFMA reads), so the store is
    # hazard-free and still lands before the end-of-iteration barrier.
    #
    # DEFAULT OFF: measured a net regression (large ~7.5%, qwen235b ~38% slower). ATT confirms the
    # v_exp *does* overlap (per-wave VALU stall ~halves), but interleaving keeps the carried raw
    # gate/up VGPRs live across the whole MFMA burst (maxVGPR v57 -> v67), bumping the allocation
    # bucket and costing ~1 wave/SIMD of occupancy. This kernel is occupancy-bound, so the lost
    # cross-wave latency hiding outweighs the per-wave gain. Kept as a flag for experimentation.
    silu_pipe = _env_on("MOE_FWD_SILU_PIPE", False)
    if gated_a and not hybrid_a:
        # Paired (gate, up) gathers from the ``[gate | up]`` layout; the plain DMA path assumes a
        # single contiguous-K staging tile.
        use_dma = False
    # XOR bank-swizzle replaces LDS_PAD: keeps a power-of-two (unpadded) row stride while
    # staying bank-conflict-free on the transpose read. Implies the DMA path's unpadded layout.
    # Default-on to match the tuned GEMM's unconditional XOR16 swizzle (frees the LDS_PAD row
    # padding for a power-of-two stride); opt out with MOE_FWD_SWZ=0 to compare against padding.
    use_swz = _env_on("MOE_FWD_SWZ", True)
    lds_pad = 0 if (use_dma or use_swz) else LDS_PAD

    WM = block_m // warps_m       # per-warp route span
    WN = block_n // warps_n       # per-warp out-feature span
    M_STEPS = WM // WMMA_M
    N_STEPS = WN // WMMA_N
    KK = block_k // WMMA_K        # contraction atoms per fill step
    N_BT = 2 if gated else 1      # B tiles: gate + up when gated
    NACC = N_BT * M_STEPS * N_STEPS

    SA = block_k + lds_pad
    A_TILE = block_m * SA
    A_FILLS = (block_m * block_k) // (n_threads * FILL_V)
    CPR_A = block_k // FILL_V
    # B LDS tile: forward stages [n, k] (coalesce/read along contiguous K); dgrad stages
    # [k, n] (coalesce along contiguous N, transpose-read the K-fragment).
    if transpose_b:
        SB = block_n + lds_pad          # [k(row), n(col)] tile row stride
        B_TILE = block_k * SB
        CPR_B = block_n // FILL_V        # chunks per k-row (coalesce along n)
    else:
        SB = block_k + lds_pad          # [n(row), k(col)] tile row stride
        B_TILE = block_n * SB
        CPR_B = block_k // FILL_V        # chunks per n-row (coalesce along k)
    B_FILLS = (block_n * block_k) // (n_threads * FILL_V)
    LANE_FILL = WARP_SIZE * FILL_V       # LDS elems one wave fills per DMA step
    DMA_BYTES = FILL_V * 2               # 16 B = one buffer_load_dwordx4 -> lds

    # Both paths ping-pong 2 LDS buffers. The DMA path unrolls the K-loop by 2 so the
    # per-sub-tile read buffer (0/1) and DMA-write buffer (1/0) are *compile-time-constant*
    # offsets into the LDS tile. That lets the backend prove read(buf_cur) does not alias
    # the in-flight DMA write(buf_other) and drop the conservative ``s_waitcnt vmcnt(0)``
    # it otherwise plants in front of the operand reads (the dynamic ``it % NUM_BUF`` ring
    # defeated that analysis and serialized the DMA -- see the ATT waitcnt-bound trace).
    # The DMA path runs a distance-(NUM_BUF-1), NUM_BUF-buffer ring (K-loop unrolled by NUM_BUF):
    # sub-tile g reads buf g and prefetches tile g+(NUM_BUF-1) into buf (g+NUM_BUF-1)%NUM_BUF, so
    # each tile's DMA streams under (NUM_BUF-1) MFMA bursts. Deeper rings (MOE_FWD_DMA_BUF=4) hide
    # more vmcnt(0) latency, but cost +1 full LDS tile: a compute-efficient tile (e.g. FC1's
    # 256x128x64) already fills LDS at NUM_BUF==3, so a 4th buffer only fits by shrinking block_k
    # / block_n, which loses more MFMA throughput than the deeper prefetch recovers. Worth it only
    # for shapes with LDS headroom at the good tile. The register path keeps the 2-buffer ping/pong.
    NUM_BUF = max(3, _env_int("MOE_FWD_DMA_BUF", 3)) if use_dma else 2

    # XCD->pid remap for L2 reuse (MI300 has 8 XCDs, each with a private L2). The default
    # round-robin workgroup->XCD dispatch scatters same-expert m-blocks (which reuse the same
    # B weight tile) across all 8 XCDs, so each XCD's L2 misses the weight independently -> up
    # to 8x redundant HBM weight traffic and a cold-miss on first touch. ``xcd_swizzle`` (the
    # M-major group width) relinearizes (pid_m, pid_n) so a contiguous run of workgroups stays
    # on one XCD, warming its L2 for the shared weight tile. Mirrors the grouped-MoE
    # ``mixed_moe_gemm_2stage`` / ``xcd_remap_bx_by`` scheme.
    #
    # DEFAULT OFF: measured a consistent regression on FC1 no-act (~9-13% at widths 1..16, worse
    # at 32) across both the qwen235b and large shapes. The reason is that the *default* dispatch
    # is already L2-friendly here: HIP launches grid=(gx, gy) x-fastest, so consecutive
    # workgroups form a full row of n-tiles for one m-block -- they share the gathered A rows and
    # a single expert's weights (streamed contiguously along N). The M-major XCD regrouping
    # scatters that natural reuse instead of concentrating it (even width 1 regresses, isolating
    # the reorder itself as the cost). Kept as a flag for experimentation on shapes where the
    # default block->tile order is *not* already locality-friendly (opt in with MOE_FWD_XCD=N).
    xcd_swizzle = max(0, _env_int("MOE_FWD_XCD", 0))

    if use_swz:
        for _S in (SA, SB):
            _nch = _S // FILL_V
            assert _S % FILL_V == 0 and (_nch & (_nch - 1)) == 0, (
                f"MOE_FWD_SWZ needs a power-of-two chunk count per row; got stride {_S} "
                f"({_nch} chunks of {FILL_V})."
            )

    KERNEL_NAME = (
        f"moe_fwd_routelist_{dtype}_{block_m}x{block_n}x{block_k}"
        f"_w{warps_m}x{warps_n}"
        f"{'_gated' if gated else ''}{'_gateda' if gated_a else ''}{'_prob' if mul_prob else ''}"
        f"{'_pre' if save_preact else ''}{'_dg' if index_a_by_route_pos else ''}"
        f"{'_tb' if transpose_b else ''}{'_dma' if use_dma else ''}"
        f"{'_swz' if use_swz else ''}"
        f"{f'_xcd{xcd_swizzle}' if xcd_swizzle > 0 else ''}"
    )

    # LDS: NUM_BUF-buffered A tile + (N_BT) NUM_BUF-buffered B tiles (NUM_BUF==2 register).
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    a_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = a_lds_off + A_TILE * NUM_BUF * 2
    # Hybrid gated_a: a parallel A tile stages the raw ``up`` half alongside ``gate``.
    a_up_lds_off = None
    if hybrid_a:
        a_up_lds_off = allocator._align(allocator.ptr, 16)
        allocator.ptr = a_up_lds_off + A_TILE * NUM_BUF * 2
    b_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = b_lds_off + N_BT * B_TILE * NUM_BUF * 2

    def _acc_idx(t, mi, nj):
        return (t * M_STEPS + mi) * N_STEPS + nj

    @flyc.kernel(known_block_size=[n_threads, 1, 1])
    def fwd_kernel(
        A: fx.Pointer,           # [*, K] bf16 gather source (received-token acts / compact grad)
        B: fx.Pointer,           # [E, N_OUT, K] bf16 weights (contiguous inner K)
        C: fx.Pointer,           # [em_max, WIDTH_N] bf16 compact route output
        SORTED: fx.Pointer,      # [em_max] i32 received-token row per slot (sentinel num_recv)
        EXPERT_IDS: fx.Pointer,  # [num_m_blocks] i32 expert per block (-1 past end)
        BLOCK_START: fx.Pointer, # [E] i32 first block index of each expert (block units)
        ROUTE_START: fx.Pointer, # [E] i32 first compact route index of each expert
        PROBS: fx.Pointer,       # [num_recv, E] f32 gating probs (or dummy)
        PREACT: fx.Pointer,      # [em_max, N_OUT] bf16 raw pre-activation (or dummy)
        K: fx.Int32,
        N_OUT: fx.Int32,
        WIDTH_N: fx.Int32,       # n-axis / C width (F when gated, N_OUT otherwise)
        num_recv_tokens: fx.Int32,
        stride_am: fx.Int32,
        stride_be: fx.Int32,
        stride_bn: fx.Int32,
        stride_bk: fx.Int32,       # B contraction stride (1 fwd; in_features for dgrad view)
        stride_pm: fx.Int32,
        stride_pe: fx.Int32,
    ):
        bf16 = T.bf16
        f32 = T.f32
        c0 = arith.constant(0, index=True)

        a_rsrc = ptr_rsrc(A)
        b_rsrc = ptr_rsrc(B)
        c_rsrc = ptr_rsrc(C)
        sorted_rsrc = ptr_rsrc(SORTED)
        eid_rsrc = ptr_rsrc(EXPERT_IDS)
        bstart_rsrc = ptr_rsrc(BLOCK_START)
        rstart_rsrc = ptr_rsrc(ROUTE_START)
        probs_rsrc = ptr_rsrc(PROBS)
        pre_rsrc = ptr_rsrc(PREACT)

        base_ptr = allocator.get_base()
        a_lds = SmemPtr(base_ptr, a_lds_off, bf16, shape=(NUM_BUF * A_TILE,)).get()
        if const_expr(hybrid_a):
            a_up_lds = SmemPtr(base_ptr, a_up_lds_off, bf16, shape=(NUM_BUF * A_TILE,)).get()
        else:
            a_up_lds = None
        b_lds = SmemPtr(base_ptr, b_lds_off, bf16, shape=(N_BT * NUM_BUF * B_TILE,)).get()

        tid = fx.Int32(gpu.thread_id("x"))
        pid_n = fx.Int32(gpu.block_id("x"))
        pid_m = fx.Int32(gpu.block_id("y"))

        # XCD->pid remap (L2 reuse). Relinearize (pid_m, pid_n) so a contiguous run of
        # workgroups lands on the same XCD, keeping the shared B weight tile warm in that
        # XCD's L2. gx = grid.x (n-tiles), gy = grid.y (m-blocks). The XCD linearization is a
        # remainder-safe bijection (matches xcd_remap_bx_by); the second stage re-tiles it
        # M-major in groups of ``xcd_swizzle``, folding the ragged tail group. Must precede the
        # ``expert = expert_ids[pid_m]`` load below so the remapped block picks its expert.
        if const_expr(xcd_swizzle > 0):
            NUM_XCDS = 8
            gx_i = arith.index_cast(T.index, gpu.grid_dim.x)
            gy_i = arith.index_cast(T.index, gpu.grid_dim.y)
            m_i = arith.index_cast(T.index, pid_m)
            n_i = arith.index_cast(T.index, pid_n)
            linear_id = m_i * gx_i + n_i
            num_wgs = gx_i * gy_i
            c_xcds = arith.index(NUM_XCDS)
            _q = num_wgs // c_xcds
            _r = num_wgs % c_xcds
            _xcd = linear_id % c_xcds
            _in_xcd = linear_id // c_xcds
            _clip = arith.cmpi(arith.CmpIPredicate.ult, _xcd, _r).select(_xcd, _r)
            wgid = _xcd * _q + _clip + _in_xcd

            c_wgm = arith.index(xcd_swizzle)
            num_in_group = c_wgm * gx_i
            group_id = wgid // num_in_group
            first_m = group_id * c_wgm
            remaining_m = gy_i - first_m
            group_size_m = arith.cmpi(
                arith.CmpIPredicate.ult, remaining_m, c_wgm
            ).select(remaining_m, c_wgm)
            in_group = wgid % num_in_group
            pid_m = arith.index_cast(T.i32, first_m + (in_group % group_size_m))
            pid_n = arith.index_cast(T.i32, in_group // group_size_m)

        # Early exit for tail blocks past the padded route extent.
        expert = buffer_load_i32(eid_rsrc, pid_m)
        e_ok = arith.cmpi(arith.CmpIPredicate.sge, expert, arith.constant(0, type=T.i32))
        if_valid = scf.IfOp(e_ok, results_=[], has_else=False)
        with ir.InsertionPoint(if_valid.then_block):
            wid = tid // WARP_SIZE
            lane = tid % WARP_SIZE
            lane_row = lane % 16          # MFMA m (A) / n (B) index
            lane_kg = lane // 16          # contraction group (0..3), 8 k each
            warp_m_base = (wid // warps_n) * fx.Int32(WM)
            warp_n_base = (wid % warps_n) * fx.Int32(WN)
            # ds_read_tr16 lane addressing (dgrad B transpose-read); mirrors wgrad v2.
            tr_k_group = (lane % 16) // 4  # 0..3
            tr_col_sub = lane % 4          # 0..3
            stride_bk_idx = arith.index_cast(T.index, stride_bk)

            nrecv_idx = arith.index_cast(T.index, num_recv_tokens)
            width_idx = arith.index_cast(T.index, WIDTH_N)
            n_out_idx = arith.index_cast(T.index, N_OUT)
            stride_am_idx = arith.index_cast(T.index, stride_am)
            stride_be_idx = arith.index_cast(T.index, stride_be)
            stride_bn_idx = arith.index_cast(T.index, stride_bn)
            # Route-prob addressing (needed by the FC2 gated_a prologue in ``gather_a`` and by the
            # FC1 gated epilogue); define once up front so both closures can capture it.
            e_pm = arith.index_cast(T.index, expert) * arith.index_cast(T.index, stride_pe)
            stride_pm_idx = arith.index_cast(T.index, stride_pm)

            bstart_e = buffer_load_i32(bstart_rsrc, expert)
            rstart_e = buffer_load_i32(rstart_rsrc, expert)
            route_start_e = arith.index_cast(T.index, rstart_e)
            # first pos of this expert's blocks = block_start[e] * BLOCK_M
            block_pos0 = arith.index_cast(T.index, bstart_e) * arith.index(block_m)
            pid_m_pos = arith.index_cast(T.index, pid_m) * arith.index(block_m)
            expert_be = arith.index_cast(T.index, expert) * stride_be_idx

            zero_v = arith.constant_vector(0.0, T.vec(FILL_V, bf16))

            # ---- A-fill descriptors (loop-invariant: rows/tokens fixed across K) ----
            a_desc = []
            for i in range_constexpr(A_FILLS):
                chunk = tid + fx.Int32(i * n_threads)
                row = chunk // fx.Int32(CPR_A)
                feat = (chunk % fx.Int32(CPR_A)) * fx.Int32(FILL_V)
                row_idx = arith.index_cast(T.index, row)
                feat_idx = arith.index_cast(T.index, feat)
                pos = pid_m_pos + row_idx
                slot = buffer_load_i32_idx(sorted_rsrc, pos)
                slot_idx = arith.index_cast(T.index, slot)
                token_ok = arith.cmpi(arith.CmpIPredicate.ult, slot_idx, nrecv_idx)
                if const_expr(index_a_by_route_pos):
                    a_row = route_start_e + (pos - block_pos0)
                else:
                    a_row = slot_idx
                arow_base = token_ok.select(a_row, c0) * stride_am_idx
                a_desc.append((arow_base, token_ok, row_idx, feat_idx, row, feat))

            # ---- B-fill descriptors (per gate/up tile) ----
            # fwd  : LDS [n(row), k(col)], coalesce+read along contiguous K (straight read).
            # dgrad: LDS [k(row), n(col)], coalesce along contiguous N, transpose-read K.
            n_base = arith.index_cast(T.index, pid_n) * arith.index(block_n)
            b_desc = [[] for _ in range_constexpr(N_BT)]
            for t in range_constexpr(N_BT):
                col_off = width_idx if const_expr(t == 1) else c0
                for i in range_constexpr(B_FILLS):
                    chunk = tid + fx.Int32(i * n_threads)
                    r = chunk // fx.Int32(CPR_B)
                    feat = (chunk % fx.Int32(CPR_B)) * fx.Int32(FILL_V)
                    r_idx = arith.index_cast(T.index, r)
                    feat_idx = arith.index_cast(T.index, feat)
                    if const_expr(transpose_b):
                        # r = k-row (LDS row), feat = n-col (contiguous global n).
                        n_global = n_base + feat_idx + col_off
                        base = expert_be + n_global * stride_bn_idx
                    else:
                        # r = n-row (LDS row), feat = k-col (contiguous global k).
                        n_global = n_base + r_idx + col_off
                        base = expert_be + n_global * stride_bn_idx
                    b_desc[t].append((base, r_idx, feat_idx, r, feat))

            def gather_a(k_idx):
                if const_expr(gated_a):
                    k_idx_i = arith.index_cast(T.index, k_idx)
                    k_width = arith.index_cast(T.index, K)
                    out = []
                    for (arow_base, ok, rw, feat_idx, row_i32, feat_i32) in a_desc:
                        # Global A load is always unswizzled (swizzle is applied only when
                        # storing to LDS via ``_a_lds_elem``), matching the non-gated path.
                        # Route prob is applied in the epilogue (per-M scalar), not here.
                        gate_off = arow_base + k_idx_i + feat_idx
                        up_off = gate_off + k_width
                        gate_vec = buffer_load_bf16_vec(a_rsrc, gate_off, FILL_V)
                        up_vec = buffer_load_bf16_vec(a_rsrc, up_off, FILL_V)
                        act_vec = _gated_a_bf16_vec8(
                            gate_vec, up_vec, None, activation, mul_prob=False
                        )
                        out.append((act_vec, ok, rw, feat_idx))
                    return out
                return [
                    (buffer_load_bf16_vec(a_rsrc, ab + k_idx + ft, FILL_V), ok, rw, ft)
                    for (ab, ok, rw, ft, _ri, _fi) in a_desc
                ]

            def _a_lds_elem(rw, ft):
                if const_expr(use_swz):
                    e = _swz_elem(
                        arith.index_cast(T.i32, rw), arith.index_cast(T.i32, ft), SA
                    )
                    return arith.index_cast(T.index, e)
                return rw * arith.index(SA) + ft

            def store_a(regs, buf_elem):
                for vec, ok, rw, ft in regs:
                    vector.store(
                        ok.select(vec, zero_v), a_lds,
                        [buf_elem + _a_lds_elem(rw, ft)], alignment=16,
                    )

            # ---- split gather/store for the deep VGPR load-ahead (reg_pf2). ``gather_a_vecs``
            # only issues the global loads and returns the raw bf16 vectors to carry across the
            # loop; ``store_a_vecs`` consumes them (fusing silu(gate)*up for gated_a) into LDS.
            # gated_a carries 2 vecs/desc (gate, up); the plain path carries 1.
            A_VECS_PER = (2 if gated_a else 1) * len(a_desc)

            def gather_a_vecs(k_idx):
                k_i = arith.index_cast(T.index, k_idx)
                vecs = []
                if const_expr(gated_a):
                    k_width = arith.index_cast(T.index, K)
                    for (arow_base, ok, rw, ft, ri, fi) in a_desc:
                        g = arow_base + k_i + ft
                        vecs.append(buffer_load_bf16_vec(a_rsrc, g, FILL_V))
                        vecs.append(buffer_load_bf16_vec(a_rsrc, g + k_width, FILL_V))
                else:
                    for (ab, ok, rw, ft, ri, fi) in a_desc:
                        vecs.append(buffer_load_bf16_vec(a_rsrc, ab + k_i + ft, FILL_V))
                return vecs

            def store_a_vecs(vecs, buf_elem):
                for j, (base_or_ab, ok, rw, ft, ri, fi) in enumerate(a_desc):
                    if const_expr(gated_a):
                        act = _gated_a_bf16_vec8(
                            vecs[2 * j], vecs[2 * j + 1], None, activation, mul_prob=False
                        )
                        val = ok.select(act, zero_v)
                    else:
                        val = ok.select(vecs[j], zero_v)
                    vector.store(val, a_lds, [buf_elem + _a_lds_elem(rw, ft)], alignment=16)

            def store_a_thunks(vecs, buf_elem):
                # One thunk per descriptor doing the (optionally gated) activation *and* its
                # ds_write into ``buf_elem``. Handed to ``compute(act_fill=...)`` so the v_exp and
                # the LDS store interleave with the MFMA burst instead of preceding it.
                def make(j):
                    base_or_ab, ok, rw, ft, ri, fi = a_desc[j]

                    def thunk():
                        if const_expr(gated_a):
                            act = _gated_a_bf16_vec8(
                                vecs[2 * j], vecs[2 * j + 1], None, activation, mul_prob=False
                            )
                            val = ok.select(act, zero_v)
                        else:
                            val = ok.select(vecs[j], zero_v)
                        vector.store(
                            val, a_lds, [buf_elem + _a_lds_elem(rw, ft)], alignment=16
                        )

                    return thunk

                return [make(j) for j in range_constexpr(len(a_desc))]

            def gather_b(t, k_idx):
                out = []
                for base, r_idx, feat_idx, _ri, _fi in b_desc[t]:
                    if const_expr(transpose_b):
                        off = base + (k_idx + r_idx) * stride_bk_idx
                    else:
                        off = base + k_idx + feat_idx
                    out.append((buffer_load_bf16_vec(b_rsrc, off, FILL_V), r_idx, feat_idx))
                return out

            def _b_lds_elem(r_idx, feat_idx):
                if const_expr(use_swz):
                    e = _swz_elem(
                        arith.index_cast(T.i32, r_idx),
                        arith.index_cast(T.i32, feat_idx), SB,
                    )
                    return arith.index_cast(T.index, e)
                return r_idx * arith.index(SB) + feat_idx

            def store_b(t, regs, buf_elem):
                for vec, r_idx, feat_idx in regs:
                    vector.store(
                        vec, b_lds,
                        [buf_elem + _b_lds_elem(r_idx, feat_idx)], alignment=16,
                    )

            B_VECS_PER = [len(b_desc[t]) for t in range(N_BT)]

            def gather_b_vecs(t, k_idx):
                vecs = []
                for base, r_idx, feat_idx, _ri, _fi in b_desc[t]:
                    if const_expr(transpose_b):
                        off = base + (k_idx + r_idx) * stride_bk_idx
                    else:
                        off = base + k_idx + feat_idx
                    vecs.append(buffer_load_bf16_vec(b_rsrc, off, FILL_V))
                return vecs

            def store_b_vecs(t, vecs, buf_elem):
                for j, (base, r_idx, feat_idx, _ri, _fi) in enumerate(b_desc[t]):
                    vector.store(
                        vecs[j], b_lds,
                        [buf_elem + _b_lds_elem(r_idx, feat_idx)], alignment=16,
                    )

            # ---- async global->LDS DMA fill (buffer_load ... lds). Requires the
            # unpadded (lane-contiguous) LDS layout: with lds_pad==0 the register
            # store address row*S + feat collapses to FILL_V*(tid + i*n_threads),
            # exactly the contiguous lane-order destination the DMA produces. ----
            if const_expr(_DMA_GEP):
                # One shared LDS base pointer, rooted in the real @smem allocation, carrying
                # the only dynamic (per-wave) displacement. Every DMA destination is then a
                # GEP off *this* value with a compile-time-constant byte offset, so the fills
                # are provably disjoint from each other and from the ds_read slices, instead
                # of each being an anonymous inttoptr the aliasing analysis must assume may
                # overlap. Mirrors the construction in preshuffle_gemm.py.
                from flydsl._mlir.dialects import memref as _memref_dialect

                _lds_root = buffer_ops.create_llvm_ptr(
                    _memref_dialect.extract_aligned_pointer_as_index(base_ptr),
                    address_space=3,
                )
                _wave_byte = wid * fx.Int32(LANE_FILL) * fx.Int32(2)
                _lds_wave_base = buffer_ops.get_element_ptr(
                    _lds_root,
                    rocdl.readfirstlane(
                        T.i64, arith.index_cast(T.i64, arith.index_cast(T.index, _wave_byte))
                    ),
                )

            if const_expr(_DMA_ALIAS):
                # One alias scope per (operand, ring slot). A fill of slot r and a read of
                # slot r share a scope, so their RAW dependency survives; every other pair is
                # marked noalias, which is what lets the waitcnt pass keep the older fills in
                # flight instead of draining them.
                _ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "moe_fwd_lds">'
                _SCOPE_IDS = tuple(
                    [f"a{r}" for r in range(NUM_BUF)]
                    + [f"b{t}_{r}" for t in range(N_BT) for r in range(NUM_BUF)]
                )

                def _scope_attr(ids):
                    inner = ", ".join(
                        f'#llvm.alias_scope<id = "{sid}", domain = {_ALIAS_DOMAIN}>'
                        for sid in ids
                    )
                    return ir.Attribute.parse(f"[{inner}]")

                _MY_SCOPE = {sid: _scope_attr((sid,)) for sid in _SCOPE_IDS}
                _NOALIAS_SCOPE = {
                    sid: _scope_attr(tuple(o for o in _SCOPE_IDS if o != sid))
                    for sid in _SCOPE_IDS
                }

            def _a_sid(slot):
                return f"a{slot % NUM_BUF}"

            def _b_sid(t, slot):
                return f"b{t}_{slot % NUM_BUF}"

            def _scope_kw(sid):
                """alias/noalias metadata kwargs for one (operand, ring slot), or {}."""
                if const_expr(not _DMA_ALIAS or sid is None):
                    return {}
                return {
                    "alias_scopes": _MY_SCOPE[sid],
                    "noalias_scopes": _NOALIAS_SCOPE[sid],
                }

            def _lds_scoped_load(lds_off, elem_idx, sid):
                """vec8 bf16 LDS read as an ``llvm.load`` so it can carry alias scopes.

                ``vector.load`` has no alias-metadata operand, and a scoped write against an
                unscoped read is still MayAlias, so the read has to be lowered by hand here.
                """
                byte = arith.index_cast(T.i32, elem_idx) * fx.Int32(2)
                ptr = buffer_ops.get_element_ptr(
                    _lds_root, byte_offset=byte, static_byte_offset=lds_off
                )
                return _llvm.LoadOp(
                    T.vec(8, bf16), ptr, alignment=16, **_scope_kw(sid)
                ).result

            def _dma_wave_ptr(lds_off, i, buf_i32, buf_elem_py=None):
                if const_expr(_DMA_GEP):
                    return buffer_ops.get_element_ptr(
                        _lds_wave_base,
                        static_byte_offset=lds_off
                        + (buf_elem_py + i * n_threads * FILL_V) * 2,
                    )
                base_elem = (
                    buf_i32 + fx.Int32(i * n_threads * FILL_V)
                    + wid * fx.Int32(LANE_FILL)
                )
                byte = base_elem * fx.Int32(2) + fx.Int32(lds_off)
                byte_i64 = arith.index_cast(T.i64, arith.index_cast(T.index, byte))
                scal = rocdl.readfirstlane(T.i64, byte_i64)
                return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), scal).result

            def _dma(rsrc, lds_ptr, voff_b, sid=None):
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, lds_ptr, arith.constant(DMA_BYTES, type=T.i32),
                    voff_b, arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32), arith.constant(1, type=T.i32),
                    **_scope_kw(sid),
                )

            def _dma_barrier(keep=0):
                # Drain global->LDS DMA down to ``keep`` in-flight vmem ops before the
                # workgroup barrier (so all waves see the staged tile), retiring this wave's
                # ds_reads (lgkmcnt) before the buffer is recycled. keep>0 leaves the newest
                # tiles' DMA streaming across the barrier to overlap the next step's MFMA.
                if const_expr(_WAITCNT_INTRINSIC):
                    # Same wait, but as ROCDL ops instead of an opaque asm blob, so the
                    # waitcnt pass can see it. gfx9 simm16 layout:
                    #   vmcnt[3:0] | expcnt << 4 | lgkmcnt << 8 | vmcnt[5:4] << 14
                    # expcnt=7 means "do not wait"; lgkmcnt=0 waits for all LDS ops.
                    imm = (keep & 0xF) | (7 << 4) | (((keep >> 4) & 0x3) << 14)
                    rocdl.s_waitcnt(imm)
                    rocdl.s_barrier()
                else:
                    asm = f"s_waitcnt vmcnt({keep}) lgkmcnt(0)\ns_barrier"
                    _llvm.InlineAsmOp(
                        res=None, operands_=[], asm_string=asm,
                        constraints="", has_side_effects=True, is_align_stack=False,
                    )

            def dma_a(k_idx, buf_i32, buf_py=None):
                # Unconditional DMA: `arow_base` already clamps invalid (padding) tokens to
                # row 0, so every lane issues exactly one `buffer_load...lds` from a valid
                # address. Padding rows load garbage, but the epilogue store is gated by
                # `token_ok` (store_ok = token_ok & col_ok) so that garbage is never written
                # to C, and MFMA is row-independent so it can't leak into valid outputs.
                # Keeping the op count data-independent (A_FILLS per tile) is what lets the
                # distance-2 pipeline use a *compile-time-constant* graduated vmcnt(keep).
                for i, (arow_base, token_ok, row_idx, feat_idx, row_i32, feat_i32) in enumerate(a_desc):
                    # Bake the LDS swizzle into the global K-column so the lane-contiguous
                    # DMA fill physically matches the swizzled `_swz_elem` read (A: row=m, col=k).
                    if const_expr(use_swz):
                        gcol = arith.index_cast(T.index, _swz_col(row_i32, feat_i32, SA))
                    else:
                        gcol = feat_idx
                    voff_b = arith.index_cast(T.i32, arow_base + k_idx + gcol) * fx.Int32(2)
                    a_elem_py = None if buf_py is None else buf_py * A_TILE
                    lds_ptr = _dma_wave_ptr(a_lds_off, i, buf_i32, a_elem_py)
                    _dma(a_rsrc, lds_ptr, voff_b,
                         None if buf_py is None else _a_sid(buf_py))
                    if const_expr(hybrid_a):
                        # Stage the ``up`` half (column + K) into the parallel tile at the same
                        # swizzled LDS slot, so read_a_frag can fuse act(gate)*up on the read.
                        up_off = arow_base + k_idx + gcol + arith.index_cast(T.index, K)
                        voff_up = arith.index_cast(T.i32, up_off) * fx.Int32(2)
                        up_ptr = _dma_wave_ptr(a_up_lds_off, i, buf_i32, a_elem_py)
                        _dma(a_rsrc, up_ptr, voff_up)

            def dma_b(t, k_idx, buf_i32, buf_py=None):
                for i, (base, r_idx, feat_idx, r_i32, feat_i32) in enumerate(b_desc[t]):
                    # Bake the LDS swizzle into the global source column of the DMA gather.
                    if const_expr(transpose_b):
                        # LDS [k(row=r), n(col=feat)]; swizzle the global N-column, which
                        # lives inside `base` via n_global*stride_bn -> shift base by delta.
                        if const_expr(use_swz):
                            swz = arith.index_cast(T.index, _swz_col(r_i32, feat_i32, SB))
                            base_eff = base + (swz - feat_idx) * stride_bn_idx
                        else:
                            base_eff = base
                        off = base_eff + (k_idx + r_idx) * stride_bk_idx
                    else:
                        # LDS [n(row=r), k(col=feat)]; swizzle the contiguous global K-column.
                        if const_expr(use_swz):
                            gcol = arith.index_cast(T.index, _swz_col(r_i32, feat_i32, SB))
                        else:
                            gcol = feat_idx
                        off = base + k_idx + gcol
                    voff_b = arith.index_cast(T.i32, off) * fx.Int32(2)
                    b_elem_py = (
                        None if buf_py is None
                        else t * NUM_BUF * B_TILE + buf_py * B_TILE
                    )
                    lds_ptr = _dma_wave_ptr(b_lds_off, i, buf_i32, b_elem_py)
                    _dma(b_rsrc, lds_ptr, voff_b,
                         None if buf_py is None else _b_sid(t, buf_py))

            def read_a_frag(buf_elem, atom_off, kk, slot=None):
                row = warp_m_base + atom_off + lane_row
                col = fx.Int32(kk * WMMA_K) + lane_kg * fx.Int32(8)
                off = _swz_elem(row, col, SA) if const_expr(use_swz) else row * fx.Int32(SA) + col
                elem = arith.index_cast(T.index, off)
                if const_expr(_DMA_ALIAS and slot is not None):
                    raw = _lds_scoped_load(a_lds_off, buf_elem + elem, _a_sid(slot))
                else:
                    raw = vector.load_op(T.vec(8, bf16), a_lds, [buf_elem + elem])
                if const_expr(hybrid_a):
                    # Fuse act(gate)*up on the LDS->VGPR read (prob deferred to the epilogue).
                    raw_up = vector.load_op(T.vec(8, bf16), a_up_lds, [buf_elem + elem])
                    fused = _gated_a_bf16_vec8(raw, raw_up, None, activation, mul_prob=False)
                    return fx.Vector(fused, (8,), fx.BFloat16)
                return fx.Vector(raw, (8,), fx.BFloat16)

            def read_b_frag(buf_elem, atom_off, kk, slot=None, t=0):
                if const_expr(transpose_b):
                    buf_byte = arith.index_cast(T.i32, buf_elem) * fx.Int32(2)
                    # dgrad transpose read: GEP off the LDS root (provenance) and carry the
                    # ring-slot alias scope so SIInsertWaitcnts keeps older fills in flight
                    # instead of draining vmcnt to 0 before the ds_read (see wgrad v2 recipe).
                    if const_expr(_DMA_ALIAS):
                        _tr_root = _lds_root
                        _tr_alias = (
                            _scope_kw(_b_sid(t, slot)) if slot is not None else None
                        )
                    else:
                        _tr_root = None
                        _tr_alias = None
                    return _tr_read_frag(
                        b_lds_off, SB, warp_n_base, atom_off, kk,
                        lane_kg, tr_k_group, tr_col_sub, buf_byte, use_swz,
                        lds_root=_tr_root, alias_kw=_tr_alias,
                    )
                row = warp_n_base + atom_off + lane_row
                col = fx.Int32(kk * WMMA_K) + lane_kg * fx.Int32(8)
                off = _swz_elem(row, col, SB) if const_expr(use_swz) else row * fx.Int32(SB) + col
                elem = arith.index_cast(T.index, off)
                if const_expr(_DMA_ALIAS and slot is not None):
                    raw = _lds_scoped_load(b_lds_off, buf_elem + elem, _b_sid(t, slot))
                else:
                    raw = vector.load_op(T.vec(8, bf16), b_lds, [buf_elem + elem])
                return fx.Vector(raw, (8,), fx.BFloat16)

            def compute(accs, a_buf, b_bufs, dma_prefetch=None, act_fill=None, slot=None):
                # Software-pipelined operand reads (mirrors the wgrad v2 kernel):
                # prefetch the next A fragment while MFMA-ing the current one, and
                # fetch each mi-invariant B fragment exactly once but interleaved
                # with the mi==0 MFMAs so its ds_read latency overlaps compute
                # instead of stalling as a serial burst. The MFMA region runs at
                # raised priority so the matrix pipe stays fed while reads are in
                # flight. sched_barrier(0) after each mi's MFMA group keeps the
                # reads from sinking past the matrix ops.
                #
                # B fragments span (t, nj); flatten that space for the mi==0
                # look-ahead prefetch. A fragments span mi. Both are per-kk.
                N_BFRAG = N_BT * N_STEPS
                new = list(accs)

                if const_expr(dma_prefetch is not None):
                    # Reads-first DMA schedule (mixed-MoE / wgrad v2): burst *every* operand
                    # ds_read for this tile BEFORE issuing the next tile's global->LDS DMA, so the
                    # reads consume the tile staged last iteration (already waited at the previous
                    # barrier) and the freshly-issued DMA streams under the MFMA burst. The
                    # reads-before-DMA *source order* is the load-bearing invariant: because
                    # ``buffer_load...lds`` is opaque to memory-SSA, scheduling the DMA in front of
                    # the reads makes the backend's waitcnt pass plant a blunt ``s_waitcnt vmcnt(0)``
                    # (measured +30-45% on FC1 no-act). The explicit sched_barrier fences that used
                    # to bracket the DMA were redundant belt-and-suspenders once the K-loop was
                    # unrolled to compile-time-constant buffers -- dropping them is perf-neutral to
                    # marginally faster -- but the ordering itself must stay: do NOT hoist the DMA.
                    a_frags = [
                        [
                            read_a_frag(a_buf, mi * WMMA_M, kk, slot)
                            for mi in range_constexpr(M_STEPS)
                        ]
                        for kk in range_constexpr(KK)
                    ]
                    b_frags = [
                        [
                            read_b_frag(
                                b_bufs[s // N_STEPS], (s % N_STEPS) * WMMA_N, kk,
                                slot, s // N_STEPS,
                            )
                            for s in range_constexpr(N_BFRAG)
                        ]
                        for kk in range_constexpr(KK)
                    ]
                    if const_expr(_DMA_GEP):
                        # With GEP-rooted LDS destinations the DMA is no longer opaque, so the
                        # scheduler happily hoists it above the operand reads -- which is the
                        # one order this loop must not have. Fence it back down.
                        rocdl.sched_barrier(0)
                        dma_prefetch()
                        rocdl.sched_barrier(0)
                    else:
                        dma_prefetch()
                    rocdl.s_setprio(1)
                    for kk in range_constexpr(KK):
                        for mi in range_constexpr(M_STEPS):
                            s = 0
                            for t in range_constexpr(N_BT):
                                for nj in range_constexpr(N_STEPS):
                                    idx = _acc_idx(t, mi, nj)
                                    new[idx] = rocdl.mfma_f32_16x16x32_bf16(
                                        T.vec(C_FRAG, f32),
                                        [a_frags[kk][mi], b_frags[kk][s], new[idx], 0, 0, 0],
                                    )
                                    s += 1
                            rocdl.sched_barrier(0)
                    rocdl.s_setprio(0)
                    return new

                # Spread the (optional) activation thunks across the KK MFMA groups so each
                # chunk of silu+ds_write co-issues on the VALU with a matrix burst. The thunks
                # land inside the raised-priority region, bounded by the per-kk sched_barrier(0),
                # so they overlap the MFMAs of their group instead of forming a serial prologue.
                fills = list(act_fill) if const_expr(act_fill is not None) else []
                n_fill = len(fills)
                fill_i = 0
                per_kk = (n_fill + KK - 1) // KK if n_fill else 0
                rocdl.s_setprio(1)
                for kk in range_constexpr(KK):
                    def read_b(s):
                        t, nj = s // N_STEPS, s % N_STEPS
                        return read_b_frag(b_bufs[t], nj * WMMA_N, kk, slot, t)

                    b_frags = [None] * N_BFRAG
                    a_next = read_a_frag(a_buf, 0, kk, slot)
                    b_frags[0] = read_b(0)  # first B needed for the very first MFMA
                    for mi in range_constexpr(M_STEPS):
                        a_cur = a_next
                        if const_expr(mi + 1 < M_STEPS):
                            a_next = read_a_frag(a_buf, (mi + 1) * WMMA_M, kk, slot)
                        s = 0
                        for t in range_constexpr(N_BT):
                            for nj in range_constexpr(N_STEPS):
                                # Prefetch the next B fragment during the first mi
                                # only; later mi reuse the resident frags. No extra
                                # VGPR: every B frag is live from mi==0 onward anyway.
                                if const_expr(mi == 0 and s + 1 < N_BFRAG):
                                    b_frags[s + 1] = read_b(s + 1)
                                idx = _acc_idx(t, mi, nj)
                                new[idx] = rocdl.mfma_f32_16x16x32_bf16(
                                    T.vec(C_FRAG, f32),
                                    [a_cur, b_frags[s], new[idx], 0, 0, 0],
                                )
                                s += 1
                        rocdl.sched_barrier(0)
                    for _ in range_constexpr(per_kk):
                        if const_expr(fill_i < n_fill):
                            fills[fill_i]()
                            fill_i += 1
                    if const_expr(n_fill):
                        rocdl.sched_barrier(0)
                # Drain any remainder (n_fill not divisible by KK).
                for _ in range_constexpr(n_fill - fill_i):
                    fills[fill_i]()
                    fill_i += 1
                rocdl.s_setprio(0)
                return new

            acc_init = [
                arith.constant_vector(0.0, T.vec(C_FRAG, f32)) for _ in range(NACC)
            ]
            step = arith.index(block_k)
            K_idx = arith.index_cast(T.index, K)

            def a_buf_elem(cur):
                return arith.index_cast(T.index, cur * fx.Int32(A_TILE))

            def b_buf_elem(t, cur):
                return arith.index_cast(
                    T.index, fx.Int32(t * NUM_BUF * B_TILE) + cur * fx.Int32(B_TILE)
                )

            def a_buf_i32(cur):
                return cur * fx.Int32(A_TILE)

            def b_buf_i32(t, cur):
                return fx.Int32(t * NUM_BUF * B_TILE) + cur * fx.Int32(B_TILE)

            acc_ty = T.vec(C_FRAG, f32)

            if const_expr(use_dma):
                # ---- DMA path: static NUM_BUF-buffer ring, distance-(NUM_BUF-1), K-loop
                # unrolled by NUM_BUF ----
                # Each physical iteration processes NUM_BUF contraction sub-tiles. Unrolling by
                # NUM_BUF makes every sub-tile's read/write buffer a *Python constant* (sub-tile
                # j always reads buffer j), so every LDS address is a compile-time offset and the
                # backend can prove read(buf j) never aliases the in-flight DMA write(buf
                # (j+DIST)%NUM_BUF). With distance-DIST prefetch, sub-tile g issues the DMA for
                # tile g+DIST (consumed DIST sub-tiles later), so each tile's global->LDS DMA
                # streams under *DIST* MFMA bursts before it is read.
                #
                # The graduated ``s_waitcnt vmcnt((DIST-1)*PER_TILE_DMA)`` at every sub-tile keeps
                # the DIST-1 most-recently-issued tiles' DMA in flight and drains the oldest (the
                # tile the *next* sub-tile reads). PER_TILE_DMA is a compile-time constant only
                # because dma_a/dma_b issue a data-independent op count per tile.
                RING = NUM_BUF
                DIST = RING - 1
                PER_TILE_DMA = A_FILLS * (2 if hybrid_a else 1) + N_BT * B_FILLS
                # Tiles left in flight after each graduated wait. MOE_FWD_DMA_KEEP overrides it
                # for A/B experiments; values above DIST-1 only stay correct because the backend
                # plants its own vmcnt(0) ahead of the operand reads.
                KEEP = _env_int("MOE_FWD_DMA_KEEP", DIST - 1)
                blk = arith.index(block_k)
                blkN = arith.index(RING * block_k)
                k_i32 = fx.Int32(arith.index_cast(T.i32, K_idx))
                ntiles = k_i32 // fx.Int32(block_k)
                # Main-loop end rounded down to a whole number of RING-tile groups; a trailing
                # 0..RING-1 tiles (if any) are handled by the runtime tail below.
                trip_end = arith.index_cast(
                    T.index, (ntiles // fx.Int32(RING)) * fx.Int32(RING * block_k)
                )

                def _clamp_k(kv):
                    return arith.cmpi(arith.CmpIPredicate.ult, kv, K_idx).select(kv, c0)

                def _dma_tile(k_tile, wbuf):
                    # Issue the global->LDS DMA of one contraction tile into constant buffer
                    # ``wbuf``; clamp past-K offsets to 0 (raw-address resources are not
                    # bounds-checked -- the clamped tile lands in a buffer never read).
                    kc = _clamp_k(k_tile)
                    dma_a(kc, a_buf_i32(fx.Int32(wbuf)), wbuf)
                    for t in range_constexpr(N_BT):
                        dma_b(t, kc, b_buf_i32(t, fx.Int32(wbuf)), wbuf)

                def _sub_tile(rbuf, wbuf, k_pf, accs, do_pf):
                    # Read constant buffer ``rbuf``; the tile-DIST-ahead DMA into constant buffer
                    # ``wbuf`` is issued *inside* compute, after the operand reads (reads-first
                    # schedule), so it overlaps the MFMA burst without forcing a vmcnt(0) drain
                    # in front of the reads. The graduated barrier then drains everything except
                    # the KEEP newest tiles before the next sub-tile reads its buffer.
                    a_cur = a_buf_elem(fx.Int32(rbuf))
                    b_cur = [b_buf_elem(t, fx.Int32(rbuf)) for t in range_constexpr(N_BT)]
                    pf = (lambda: _dma_tile(k_pf, wbuf)) if const_expr(do_pf) else None
                    new = compute(accs, a_cur, b_cur, dma_prefetch=pf, slot=rbuf)
                    _dma_barrier(KEEP * PER_TILE_DMA if const_expr(do_pf) else 0)
                    return new

                # Prologue: stage the first DIST tiles (distance-DIST) into buf0..buf(DIST-1).
                # The first sub-tile reads only buf0 (tile 0), so drain just that one and keep
                # the KEEP=DIST-1 newest tiles streaming under the first MFMA burst (the graduated
                # drain matches the steady-state loop). The s_barrier still publishes tile 0's LDS
                # to the whole workgroup before the first read.
                for j in range_constexpr(DIST):
                    _dma_tile(arith.index(j * block_k), j)
                _dma_barrier(KEEP * PER_TILE_DMA)

                loop = scf.ForOp(c0, trip_end, blkN, iter_args=acc_init)
                with ir.InsertionPoint(loop.body):
                    k_base = loop.induction_variable
                    accs = [loop.body.arguments[1 + i] for i in range(NACC)]
                    # sub-tile j reads buf j (tile RING*p+j), prefetches tile RING*p+j+DIST into
                    # buf (j+DIST)%RING.
                    for j in range_constexpr(RING):
                        k_pf = k_base + arith.index((j + DIST) * block_k)
                        accs = _sub_tile(j, (j + DIST) % RING, k_pf, accs, True)
                    scf.YieldOp(accs)
                accs = [loop.results[i] for i in range(NACC)]

                # Tail: 0..RING-1 leftover tiles. The last group's prefetches left tile
                # (G+m) staged in buf m for m in 0..DIST-1 (G = trip_end tile index), so the
                # leftover tiles are read straight from buf 0,1,.. with no further prefetch.
                # Drain any DMA still in flight first. K is workgroup-uniform, so every branch
                # is uniform. Build the nested "has>=m" chain generically for arbitrary RING.
                _dma_barrier()
                grp = (ntiles // fx.Int32(RING)) * fx.Int32(RING)
                rem = ntiles - grp

                def _tail(m, accs_in):
                    # Consume leftover tile index m (0-based) if rem >= m+1, then recurse.
                    if const_expr(m >= RING):
                        return accs_in
                    has = arith.cmpi(arith.CmpIPredicate.uge, rem, fx.Int32(m + 1))
                    iff = scf.IfOp(has, results_=[acc_ty] * NACC, has_else=True)
                    with ir.InsertionPoint(iff.then_block):
                        acc_m = _sub_tile(m, 0, c0, accs_in, False)
                        scf.YieldOp(_tail(m + 1, acc_m))
                    with ir.InsertionPoint(iff.else_block):
                        scf.YieldOp(accs_in)
                    return [iff.results[i] for i in range(NACC)]

                accs = _tail(0, accs)
            elif const_expr(reg_pf2):
                # ---- Register path, deep VGPR load-ahead: 2 LDS buffers, but the global gather
                # runs 2 K-tiles ahead (carried in VGPRs via loop iter_args) so each tile's
                # buffer_load overlaps a whole MFMA+store+barrier cycle. ----
                def _clamp_k(kv):
                    return arith.cmpi(arith.CmpIPredicate.ult, kv, K_idx).select(kv, c0)

                # Prologue: stage tile 0 into buf0, then issue tile 1's gather into the carry.
                store_a_vecs(gather_a_vecs(c0), a_buf_elem(fx.Int32(0)))
                for t in range_constexpr(N_BT):
                    store_b_vecs(t, gather_b_vecs(t, c0), b_buf_elem(t, fx.Int32(0)))
                gpu.barrier()

                k1 = _clamp_k(c0 + step)
                a_carry0 = gather_a_vecs(k1)
                b_carry0 = [gather_b_vecs(t, k1) for t in range_constexpr(N_BT)]
                b_carry0_flat = [v for t in range_constexpr(N_BT) for v in b_carry0[t]]
                NB = len(b_carry0_flat)

                loop = scf.ForOp(
                    c0, K_idx, step, iter_args=acc_init + a_carry0 + b_carry0_flat
                )
                with ir.InsertionPoint(loop.body):
                    k_base = loop.induction_variable
                    args = loop.body.arguments
                    accs = [args[1 + i] for i in range(NACC)]
                    a_carry = [args[1 + NACC + i] for i in range(A_VECS_PER)]
                    b_carry_flat = [args[1 + NACC + A_VECS_PER + i] for i in range(NB)]
                    b_carry = []
                    off = 0
                    for t in range_constexpr(N_BT):
                        b_carry.append(b_carry_flat[off:off + B_VECS_PER[t]])
                        off += B_VECS_PER[t]

                    it = fx.Int32(arith.index_cast(T.i32, k_base)) // fx.Int32(block_k)
                    cur = it % fx.Int32(2)
                    nxt = fx.Int32(1) - cur
                    a_cur = a_buf_elem(cur)
                    b_cur = [b_buf_elem(t, cur) for t in range_constexpr(N_BT)]

                    # Store the already-loaded tile (it+1) carried from last iteration into nxt.
                    # With silu_pipe the A activation+store is deferred into ``compute`` (see
                    # act_fill) so the v_exp overlaps the MFMA; the B store (no activation) stays
                    # ahead of compute either way.
                    a_fill = None
                    if const_expr(silu_pipe):
                        a_fill = store_a_thunks(a_carry, a_buf_elem(nxt))
                    else:
                        store_a_vecs(a_carry, a_buf_elem(nxt))
                    for t in range_constexpr(N_BT):
                        store_b_vecs(t, b_carry[t], b_buf_elem(t, nxt))

                    # Issue tile (it+2)'s global gather now (streams under this tile's MFMA).
                    k2 = _clamp_k(k_base + step + step)
                    a_next = gather_a_vecs(k2)
                    b_next = [gather_b_vecs(t, k2) for t in range_constexpr(N_BT)]
                    b_next_flat = [v for t in range_constexpr(N_BT) for v in b_next[t]]

                    new_accs = compute(accs, a_cur, b_cur, act_fill=a_fill)
                    rocdl.sched_barrier(0)
                    gpu.barrier()
                    scf.YieldOp(new_accs + a_next + b_next_flat)
                accs = [loop.results[i] for i in range(NACC)]
            else:
                # ---- Register path: 2-buffer ping/pong (global->VGPR->ds_write) ----
                store_a(gather_a(c0), a_buf_elem(fx.Int32(0)))
                for t in range_constexpr(N_BT):
                    store_b(t, gather_b(t, c0), b_buf_elem(t, fx.Int32(0)))
                gpu.barrier()

                loop = scf.ForOp(c0, K_idx, step, iter_args=acc_init)
                with ir.InsertionPoint(loop.body):
                    k_base = loop.induction_variable
                    accs = [loop.body.arguments[1 + i] for i in range(NACC)]

                    it = fx.Int32(arith.index_cast(T.i32, k_base)) // fx.Int32(block_k)
                    cur = it % fx.Int32(2)
                    nxt = fx.Int32(1) - cur
                    a_cur = a_buf_elem(cur)
                    b_cur = [b_buf_elem(t, cur) for t in range_constexpr(N_BT)]
                    # Prefetch next tile into regs; clamp final-iteration overrun to 0.
                    k_next_raw = k_base + step
                    k_next = arith.cmpi(
                        arith.CmpIPredicate.ult, k_next_raw, K_idx
                    ).select(k_next_raw, c0)
                    a_regs = gather_a(k_next)
                    b_regs = [gather_b(t, k_next) for t in range_constexpr(N_BT)]

                    new_accs = compute(accs, a_cur, b_cur)

                    rocdl.sched_barrier(0)
                    store_a(a_regs, a_buf_elem(nxt))
                    for t in range_constexpr(N_BT):
                        store_b(t, b_regs[t], b_buf_elem(t, nxt))
                    gpu.barrier()
                    scf.YieldOp(new_accs)
                accs = [loop.results[i] for i in range(NACC)]

            # ---- Epilogue: gated act(+prob)(+preact) or plain store ----
            # (e_pm / stride_pm_idx were hoisted above so the FC2 gated_a prologue can use them.)
            for mi in range_constexpr(M_STEPS):
                for nj in range_constexpr(N_STEPS):
                    ncol = warp_n_base + fx.Int32(nj * WMMA_N) + lane_row
                    ncol_g = n_base + arith.index_cast(T.index, ncol)
                    col_ok = arith.cmpi(arith.CmpIPredicate.ult, ncol_g, width_idx)
                    if const_expr(gated):
                        ag = accs[_acc_idx(0, mi, nj)]
                        au = accs[_acc_idx(1, mi, nj)]
                    else:
                        ac = accs[_acc_idx(0, mi, nj)]
                    for ii in range_constexpr(C_FRAG):
                        row_t = warp_m_base + fx.Int32(mi * WMMA_M) + lane_kg * fx.Int32(4) + fx.Int32(ii)
                        pos = pid_m_pos + arith.index_cast(T.index, row_t)
                        slot = buffer_load_i32_idx(sorted_rsrc, pos)
                        slot_idx = arith.index_cast(T.index, slot)
                        token_ok = arith.cmpi(arith.CmpIPredicate.ult, slot_idx, nrecv_idx)
                        out_row = route_start_e + (pos - block_pos0)
                        store_ok = arith.andi(token_ok, col_ok)
                        c_off = out_row * width_idx + ncol_g

                        if const_expr(gated):
                            g = vector.extract(ag, static_position=[ii], dynamic_position=[])
                            u = vector.extract(au, static_position=[ii], dynamic_position=[])
                            val = arith.mulf(_act_f32(g, activation), u)
                            if const_expr(mul_prob):
                                p_off = token_ok.select(slot_idx, c0) * stride_pm_idx + e_pm
                                prob = buffer_load_f32(probs_rsrc, p_off)
                                val = arith.mulf(val, prob)
                            cval = arith.truncf(bf16, val)
                            store_if = scf.IfOp(store_ok, results_=[], has_else=False)
                            with ir.InsertionPoint(store_if.then_block):
                                buffer_store_bf16(c_rsrc, c_off, cval)
                                if const_expr(save_preact):
                                    pre_g = out_row * n_out_idx + ncol_g
                                    buffer_store_bf16(pre_rsrc, pre_g, arith.truncf(bf16, g))
                                    buffer_store_bf16(
                                        pre_rsrc, pre_g + width_idx, arith.truncf(bf16, u)
                                    )
                                scf.YieldOp([])
                        else:
                            v = vector.extract(ac, static_position=[ii], dynamic_position=[])
                            if const_expr(gated_a and mul_prob):
                                # FC2 fused prologue: A already holds act(gate)*up; the per-route
                                # prob factors out of the GEMM and is applied here (out *= prob[m]).
                                p_off = token_ok.select(slot_idx, c0) * stride_pm_idx + e_pm
                                prob = buffer_load_f32(probs_rsrc, p_off)
                                v = arith.mulf(v, prob)
                            cval = arith.truncf(bf16, v)
                            store_if = scf.IfOp(store_ok, results_=[], has_else=False)
                            with ir.InsertionPoint(store_if.then_block):
                                buffer_store_bf16(c_rsrc, c_off, cval)
                                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_fwd(
        A: fx.Pointer,
        B: fx.Pointer,
        C: fx.Pointer,
        SORTED: fx.Pointer,
        EXPERT_IDS: fx.Pointer,
        BLOCK_START: fx.Pointer,
        ROUTE_START: fx.Pointer,
        PROBS: fx.Pointer,
        PREACT: fx.Pointer,
        K: fx.Int32,
        N_OUT: fx.Int32,
        WIDTH_N: fx.Int32,
        num_recv_tokens: fx.Int32,
        stride_am: fx.Int32,
        stride_be: fx.Int32,
        stride_bn: fx.Int32,
        stride_bk: fx.Int32,
        stride_pm: fx.Int32,
        stride_pe: fx.Int32,
        num_m_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gx = (WIDTH_N + block_n - 1) // block_n
        gy = num_m_blocks
        fwd_kernel._func.__name__ = KERNEL_NAME
        fwd_kernel(
            A, B, C, SORTED, EXPERT_IDS, BLOCK_START, ROUTE_START, PROBS, PREACT,
            K, N_OUT, WIDTH_N, num_recv_tokens,
            stride_am, stride_be, stride_bn, stride_bk, stride_pm, stride_pe,
        ).launch(grid=(gx, gy, 1), block=(n_threads, 1, 1), stream=stream)

    return launch_fwd


# Aliases: the route-list ``compile_moe_gemm1`` is the v2 forward entry point. ``moe_fwd.py``
# selects it via ``AITER_MOE_FWD_KERNEL=v2`` (imports ``compile_moe_fwd_v2``); ``compile_moe_fwd``
# keeps the historical name available for any direct importer.
compile_moe_fwd_v2 = compile_moe_gemm1
compile_moe_fwd = compile_moe_gemm1


def _tr_lds_ptr(lds_off, byte_elem, buf_byte, lds_root=None):
    byte = byte_elem * fx.Int32(2) + fx.Int32(lds_off) + buf_byte
    if lds_root is not None:
        # GEP off the real LDS object so the transpose read shares a base with the DMA fill;
        # the aliasing analysis needs this provenance (an inttoptr address is opaque to it).
        return buffer_ops.get_element_ptr(lds_root, byte_offset=byte)
    byte_i64 = arith.index_cast(T.i64, arith.index_cast(T.index, byte))
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), byte_i64).result


def _tr_read_frag(
    lds_off, stride, warp_col_base, col_const, kk,
    lane_m_base, tr_k_group, tr_col_sub, buf_byte, use_swz=False,
    lds_root=None, alias_kw=None,
):
    """ds_read_tr16_b64 B fragment for a 16-wide feature (n) column from a [k, n] LDS tile.

    Mirrors the wgrad v2 recipe: the tile holds contraction (k) along rows and feature (n)
    along columns, and the transpose-read reconstructs the canonical MFMA K-fragment (lane%16
    = feature, 8 contraction values per lane). ``kk`` selects the 32-wide contraction substep
    (row base ``kk*32``); ``buf_byte`` picks the ping-pong buffer (runtime byte offset).

    The two ds_read_tr halves read logical rows ``row_lo`` and ``row_lo+4`` (a fixed 4-row
    byte stride when unswizzled). Under ``use_swz`` those rows land at different swizzled
    columns, so each half's address is recomputed via :func:`_swz_elem` on the same logical
    ``(row, col)`` the unswizzled path would read -- keeping the transpose output identical.
    """
    col_run = warp_col_base + tr_col_sub * fx.Int32(4)
    row_lo = fx.Int32(kk * WMMA_K) + lane_m_base * fx.Int32(8) + tr_k_group  # contraction row

    if use_swz:
        n_read = col_run + fx.Int32(col_const)     # actual feature column this lane reads
        lo_ptr = _tr_lds_ptr(lds_off, _swz_elem(row_lo, n_read, stride), buf_byte, lds_root)
        hi_ptr = _tr_lds_ptr(
            lds_off, _swz_elem(row_lo + fx.Int32(4), n_read, stride), buf_byte, lds_root
        )
        lo = _ds_read_tr_bf16x4(lo_ptr, alias_kw=alias_kw)
        hi = _ds_read_tr_bf16x4(hi_ptr, alias_kw=alias_kw)
        return lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7])

    base_elem = row_lo * fx.Int32(stride) + col_run
    base_ptr = _tr_lds_ptr(lds_off, base_elem, buf_byte, lds_root)

    col_byte = 2 * col_const       # compile-time feature-column shift (bytes)
    hi_byte = 2 * 4 * stride        # compile-time lo->hi row shift (4 rows apart)
    lo = _ds_read_tr_bf16x4(base_ptr, col_byte, alias_kw=alias_kw)
    hi = _ds_read_tr_bf16x4(base_ptr, col_byte + hi_byte, alias_kw=alias_kw)
    return lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7])


def _ds_read_tr_bf16x4(base_ptr, const_byte=0, alias_kw=None):
    if const_byte:
        ptr = _llvm.GEPOp(
            ir.Type.parse("!llvm.ptr<3>"),
            base_ptr,
            [],
            [const_byte],
            ir.IntegerType.get_signless(8),
            _llvm.GEPNoWrapFlags.inboundsFlag,
        ).result
    else:
        ptr = base_ptr
    raw = rocdl.ds_read_tr16_b64(T.vec(4, T.bf16), ptr, **(alias_kw or {})).result
    return fx.Vector(raw, (4,), fx.BFloat16)


def buffer_load_i32(rsrc, off_i32):
    return buffer_ops.buffer_load(rsrc, off_i32, vec_width=1, dtype=T.i32)


def buffer_load_i32_idx(rsrc, off_idx):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.i32)


def buffer_load_f32(rsrc, off_idx):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.f32)


def buffer_load_bf16_vec(rsrc, off_idx, v):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=v, dtype=T.bf16)


def buffer_store_bf16(rsrc, off_idx, val):
    buffer_ops.buffer_store(val, rsrc, off_idx)

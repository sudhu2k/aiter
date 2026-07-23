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

Fused gated-activation epilogue (matches the Triton contract)
------------------------------------------------------------
When ``GATED`` the GEMM output width is the gate+up width ``N_OUT = 2F``; the kernel computes
two ``[BLOCK_M, BLOCK_N]`` accumulators from a shared ``A`` tile -- one over the gate columns
``[0, F)`` and one over the up columns ``[F, 2F)`` -- so the matching (gate, up) pair for an
output feature lands in the *same lane* (no cross-lane shuffle). The epilogue emits
``act(gate) * up`` (``act`` = silu/gelu) into the ``F``-wide ``C``; ``MUL_PROB`` multiplies the
per-route gating prob in *after* the activation; ``SAVE_PREACT`` also stores the raw ``2F``
``[gate | up]`` pre-activation for the backward.

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
    if transpose_b and gated:
        # dgrad (transposed weights) is never gated; keep the two epilogues decoupled.
        raise ValueError("transpose_b (dgrad) does not support the gated epilogue")

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

    use_dma = _env_on("MOE_FWD_DMA", True)
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
    # The DMA path runs a distance-2, 3-buffer ring (K-loop unrolled by 3): sub-tile g reads
    # buf g and prefetches tile g+2 into buf (g+2)%3, so each tile's DMA streams under two
    # MFMA bursts. The register path keeps the classic 2-buffer ping/pong.
    NUM_BUF = 3 if use_dma else 2

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
        f"{'_gated' if gated else ''}{'_prob' if mul_prob else ''}"
        f"{'_pre' if save_preact else ''}{'_dg' if index_a_by_route_pos else ''}"
        f"{'_tb' if transpose_b else ''}{'_dma' if use_dma else ''}"
        f"{'_swz' if use_swz else ''}"
    )

    # LDS: NUM_BUF-buffered A tile + (N_BT) NUM_BUF-buffered B tiles (NUM_BUF==2 register).
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    a_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = a_lds_off + A_TILE * NUM_BUF * 2
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
        b_lds = SmemPtr(base_ptr, b_lds_off, bf16, shape=(N_BT * NUM_BUF * B_TILE,)).get()

        tid = fx.Int32(gpu.thread_id("x"))
        pid_n = fx.Int32(gpu.block_id("x"))
        pid_m = fx.Int32(gpu.block_id("y"))

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

            # ---- async global->LDS DMA fill (buffer_load ... lds). Requires the
            # unpadded (lane-contiguous) LDS layout: with lds_pad==0 the register
            # store address row*S + feat collapses to FILL_V*(tid + i*n_threads),
            # exactly the contiguous lane-order destination the DMA produces. ----
            def _dma_wave_ptr(lds_off, i, buf_i32):
                base_elem = (
                    buf_i32 + fx.Int32(i * n_threads * FILL_V)
                    + wid * fx.Int32(LANE_FILL)
                )
                byte = base_elem * fx.Int32(2) + fx.Int32(lds_off)
                byte_i64 = arith.index_cast(T.i64, arith.index_cast(T.index, byte))
                scal = rocdl.readfirstlane(T.i64, byte_i64)
                return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), scal).result

            def _dma(rsrc, lds_ptr, voff_b):
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, lds_ptr, arith.constant(DMA_BYTES, type=T.i32),
                    voff_b, arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32), arith.constant(1, type=T.i32),
                )

            def _dma_barrier(keep=0):
                # Drain global->LDS DMA down to ``keep`` in-flight vmem ops before the
                # workgroup barrier (so all waves see the staged tile), retiring this wave's
                # ds_reads (lgkmcnt) before the buffer is recycled. keep>0 leaves the newest
                # tiles' DMA streaming across the barrier to overlap the next step's MFMA.
                asm = f"s_waitcnt vmcnt({keep}) lgkmcnt(0)\ns_barrier"
                _llvm.InlineAsmOp(
                    res=None, operands_=[], asm_string=asm,
                    constraints="", has_side_effects=True, is_align_stack=False,
                )

            def dma_a(k_idx, buf_i32):
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
                    lds_ptr = _dma_wave_ptr(a_lds_off, i, buf_i32)
                    _dma(a_rsrc, lds_ptr, voff_b)

            def dma_b(t, k_idx, buf_i32):
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
                    lds_ptr = _dma_wave_ptr(b_lds_off, i, buf_i32)
                    _dma(b_rsrc, lds_ptr, voff_b)

            def read_a_frag(buf_elem, atom_off, kk):
                row = warp_m_base + atom_off + lane_row
                col = fx.Int32(kk * WMMA_K) + lane_kg * fx.Int32(8)
                off = _swz_elem(row, col, SA) if const_expr(use_swz) else row * fx.Int32(SA) + col
                elem = arith.index_cast(T.index, off)
                raw = vector.load_op(T.vec(8, bf16), a_lds, [buf_elem + elem])
                return fx.Vector(raw, (8,), fx.BFloat16)

            def read_b_frag(buf_elem, atom_off, kk):
                if const_expr(transpose_b):
                    buf_byte = arith.index_cast(T.i32, buf_elem) * fx.Int32(2)
                    return _tr_read_frag(
                        b_lds_off, SB, warp_n_base, atom_off, kk,
                        lane_kg, tr_k_group, tr_col_sub, buf_byte, use_swz,
                    )
                row = warp_n_base + atom_off + lane_row
                col = fx.Int32(kk * WMMA_K) + lane_kg * fx.Int32(8)
                off = _swz_elem(row, col, SB) if const_expr(use_swz) else row * fx.Int32(SB) + col
                elem = arith.index_cast(T.index, off)
                raw = vector.load_op(T.vec(8, bf16), b_lds, [buf_elem + elem])
                return fx.Vector(raw, (8,), fx.BFloat16)

            def compute(accs, a_buf, b_bufs, dma_prefetch=None):
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
                    # ds_read for this tile BEFORE issuing the next tile's global->LDS DMA.
                    # The async buffer_load...lds write is opaque to memory-SSA, so a fresh
                    # DMA placed immediately in front of the reads makes the backend plant a
                    # blunt ``s_waitcnt vmcnt(0)`` (the 2.3M-cycle stall the ATT trace shows).
                    # With the reads ahead of the DMA issue, they instead consume the tile
                    # staged last iteration (already waited at the previous barrier), and the
                    # freshly-issued DMA streams under the MFMA burst. sched_barrier fences
                    # keep the read burst from sinking below the DMA issue.
                    a_frags = [
                        [read_a_frag(a_buf, mi * WMMA_M, kk) for mi in range_constexpr(M_STEPS)]
                        for kk in range_constexpr(KK)
                    ]
                    b_frags = [
                        [
                            read_b_frag(b_bufs[s // N_STEPS], (s % N_STEPS) * WMMA_N, kk)
                            for s in range_constexpr(N_BFRAG)
                        ]
                        for kk in range_constexpr(KK)
                    ]
                    rocdl.sched_barrier(0)
                    dma_prefetch()
                    rocdl.sched_barrier(0)
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

                rocdl.s_setprio(1)
                for kk in range_constexpr(KK):
                    def read_b(s):
                        t, nj = s // N_STEPS, s % N_STEPS
                        return read_b_frag(b_bufs[t], nj * WMMA_N, kk)

                    b_frags = [None] * N_BFRAG
                    a_next = read_a_frag(a_buf, 0, kk)
                    b_frags[0] = read_b(0)  # first B needed for the very first MFMA
                    for mi in range_constexpr(M_STEPS):
                        a_cur = a_next
                        if const_expr(mi + 1 < M_STEPS):
                            a_next = read_a_frag(a_buf, (mi + 1) * WMMA_M, kk)
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
                # ---- DMA path: static 3-buffer ring, distance-2, K-loop unrolled by 3 ----
                # Each physical iteration processes three contraction sub-tiles. Unrolling by
                # NUM_BUF (=3) makes every sub-tile's read/write buffer a *Python constant*
                # (sub-tile j always reads buffer j), so every LDS address is a compile-time
                # offset and the backend can prove read(buf j) never aliases the in-flight DMA
                # write(buf j+2). With distance-2 prefetch, sub-tile g issues the DMA for tile
                # g+2 (consumed two sub-tiles later), so each tile's global->LDS DMA streams
                # under *two* MFMA bursts before it is read.
                #
                # The graduated ``s_waitcnt vmcnt(PER_TILE_DMA)`` at every sub-tile keeps only
                # the just-issued tile's DMA in flight and drains the older one (the tile the
                # *next* sub-tile reads). PER_TILE_DMA is a compile-time constant only because
                # dma_a/dma_b issue a data-independent op count per tile -- hence the
                # unconditional dma_a above.
                PER_TILE_DMA = A_FILLS + N_BT * B_FILLS
                blk = arith.index(block_k)
                blk2 = arith.index(2 * block_k)
                blk3 = arith.index(3 * block_k)
                k_i32 = fx.Int32(arith.index_cast(T.i32, K_idx))
                ntiles = k_i32 // fx.Int32(block_k)
                # Main-loop end rounded down to a whole number of 3-tile groups; a trailing
                # 1 or 2 tiles (if any) are handled by the runtime tail below.
                trip_end = arith.index_cast(
                    T.index, (ntiles // fx.Int32(3)) * fx.Int32(3 * block_k)
                )

                def _clamp_k(kv):
                    return arith.cmpi(arith.CmpIPredicate.ult, kv, K_idx).select(kv, c0)

                def _dma_tile(k_tile, wbuf):
                    # Issue the global->LDS DMA of one contraction tile into constant buffer
                    # ``wbuf``; clamp past-K offsets to 0 (raw-address resources are not
                    # bounds-checked -- the clamped tile lands in a buffer never read).
                    kc = _clamp_k(k_tile)
                    dma_a(kc, a_buf_i32(fx.Int32(wbuf)))
                    for t in range_constexpr(N_BT):
                        dma_b(t, kc, b_buf_i32(t, fx.Int32(wbuf)))

                def _sub_tile(rbuf, wbuf, k_pf, accs, do_pf):
                    # Read constant buffer ``rbuf``; the tile-two-ahead DMA into constant buffer
                    # ``wbuf`` is issued *inside* compute, after the operand reads (reads-first
                    # schedule), so it overlaps the MFMA burst without forcing a vmcnt(0) drain
                    # in front of the reads. The graduated barrier then drains everything except
                    # the just-issued tile before the next sub-tile reads its buffer.
                    a_cur = a_buf_elem(fx.Int32(rbuf))
                    b_cur = [b_buf_elem(t, fx.Int32(rbuf)) for t in range_constexpr(N_BT)]
                    pf = (lambda: _dma_tile(k_pf, wbuf)) if const_expr(do_pf) else None
                    new = compute(accs, a_cur, b_cur, dma_prefetch=pf)
                    _dma_barrier(PER_TILE_DMA if const_expr(do_pf) else 0)
                    return new

                # Prologue: stage the first two tiles (distance-2) into buf0, buf1; full drain.
                _dma_tile(c0, 0)
                _dma_tile(blk, 1)
                _dma_barrier()

                loop = scf.ForOp(c0, trip_end, blk3, iter_args=acc_init)
                with ir.InsertionPoint(loop.body):
                    k_base = loop.induction_variable
                    accs = [loop.body.arguments[1 + i] for i in range(NACC)]
                    # sub-tile j reads buf j (tile 3p+j), prefetches tile 3p+j+2 -> buf (j+2)%3
                    accs = _sub_tile(0, 2, k_base + blk2, accs, True)
                    accs = _sub_tile(1, 0, k_base + blk3, accs, True)
                    accs = _sub_tile(2, 1, k_base + blk3 + blk, accs, True)
                    scf.YieldOp(accs)
                accs = [loop.results[i] for i in range(NACC)]

                # Tail: 0, 1 or 2 leftover tiles, already prefetched into buf0 (and buf1) by the
                # last group. Drain any DMA still in flight first, then consume without further
                # prefetch. K is workgroup-uniform, so both branches are uniform.
                _dma_barrier()
                rem = ntiles - (ntiles // fx.Int32(3)) * fx.Int32(3)
                has1 = arith.cmpi(arith.CmpIPredicate.uge, rem, fx.Int32(1))
                has2 = arith.cmpi(arith.CmpIPredicate.uge, rem, fx.Int32(2))
                t1 = scf.IfOp(has1, results_=[acc_ty] * NACC, has_else=True)
                with ir.InsertionPoint(t1.then_block):
                    a1 = _sub_tile(0, 0, c0, accs, False)
                    t2 = scf.IfOp(has2, results_=[acc_ty] * NACC, has_else=True)
                    with ir.InsertionPoint(t2.then_block):
                        scf.YieldOp(_sub_tile(1, 0, c0, a1, False))
                    with ir.InsertionPoint(t2.else_block):
                        scf.YieldOp(a1)
                    scf.YieldOp([t2.results[i] for i in range(NACC)])
                with ir.InsertionPoint(t1.else_block):
                    scf.YieldOp(accs)
                accs = [t1.results[i] for i in range(NACC)]
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
            e_pm = arith.index_cast(T.index, expert) * arith.index_cast(T.index, stride_pe)
            stride_pm_idx = arith.index_cast(T.index, stride_pm)
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


def _tr_lds_ptr(lds_off, byte_elem, buf_byte):
    byte = byte_elem * fx.Int32(2) + fx.Int32(lds_off) + buf_byte
    byte_i64 = arith.index_cast(T.i64, arith.index_cast(T.index, byte))
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), byte_i64).result


def _tr_read_frag(
    lds_off, stride, warp_col_base, col_const, kk,
    lane_m_base, tr_k_group, tr_col_sub, buf_byte, use_swz=False,
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
        lo_ptr = _tr_lds_ptr(lds_off, _swz_elem(row_lo, n_read, stride), buf_byte)
        hi_ptr = _tr_lds_ptr(
            lds_off, _swz_elem(row_lo + fx.Int32(4), n_read, stride), buf_byte
        )
        lo = _ds_read_tr_bf16x4(lo_ptr)
        hi = _ds_read_tr_bf16x4(hi_ptr)
        return lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7])

    base_elem = row_lo * fx.Int32(stride) + col_run
    base_ptr = _tr_lds_ptr(lds_off, base_elem, buf_byte)

    col_byte = 2 * col_const       # compile-time feature-column shift (bytes)
    hi_byte = 2 * 4 * stride        # compile-time lo->hi row shift (4 rows apart)
    lo = _ds_read_tr_bf16x4(base_ptr, col_byte)
    hi = _ds_read_tr_bf16x4(base_ptr, col_byte + hi_byte)
    return lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7])


def _ds_read_tr_bf16x4(base_ptr, const_byte=0):
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
    raw = rocdl.ds_read_tr16_b64(T.vec(4, T.bf16), ptr).result
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

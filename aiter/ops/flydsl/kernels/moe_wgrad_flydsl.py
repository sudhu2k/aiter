# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE weight-gradient (wgrad) grouped GEMM in FlyDSL.

Computes, per expert ``e``::

    dW[e][n, k] = sum_{routed slot s of e} grad[s, n] * x[token(s), k]

where the reduction runs over the expert's routed token-slots, gathered through the
same block-padded ``sorted_token_ids`` buffer used by the forward gather-GEMM (so no
activation/grad permutation is materialized). ``token(s) = slot_id(s) // top_k``.

v0 design notes
---------------
This is the correctness-first variant. Each program owns a single MFMA ``16x16x32``
output atom ``dW[e][n0:n0+16, k0:k0+16]`` and is one wavefront (64 lanes). The
contraction axis is the token-slot dimension, tiled in steps of ``WMMA_K = 32``.

Crucially, it feeds the matrix core **directly from VGPRs**: each lane gathers its own
8-element A (grad) and B (x) MFMA fragments straight from global memory and calls the
MFMA -- there is no LDS staging and no workgroup barrier. This deliberately sidesteps
the ``ds_write -> s_barrier -> ds_read_tr`` LDS transpose round-trip that dominates the
Triton wgrad kernel, at the cost of strided (uncoalesced) global gathers. Tiling /
coalescing / multi-warp blocking are follow-ups.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from .splitk_hgemm import WmmaHalf_m16n16k32
from .tensor_shim import ptr_rsrc

__all__ = ["compile_moe_wgrad", "WGRAD_BLOCK_M"]

# MFMA atom dims (16x16x32 bf16 on gfx950/gfx942).
WMMA_M = 16  # output rows  -> N (grad feature) tile
WMMA_N = 16  # output cols  -> K (x feature) tile
WMMA_K = 32  # contraction  -> token-slot tile
A_FRAG = 8   # bf16 values per lane for the A (grad) fragment
B_FRAG = 8   # bf16 values per lane for the B (x) fragment
C_FRAG = 4   # f32 values per lane for the C (dW) fragment
WARP_SIZE = 64

# ``block_size`` used to build the align/offset buffers (must match the wrapper).
WGRAD_BLOCK_M = 32


@functools.lru_cache(maxsize=None)
def compile_moe_wgrad(
    *,
    dtype: str = "bf16",
    mul_routed_weight: bool = False,
    block_n: int = 64,
    block_k: int = 64,
):
    """Register-blocked permute-free wgrad (v1).

    One wavefront computes a ``block_n x block_k`` output tile of ``dW[e]`` as an
    ``M_STEPS x N_STEPS`` grid of ``16x16x32`` MFMA atoms held in registers. Per
    contraction step (32 slots) each lane decodes its slot-ids/tokens once and reuses
    them: each gathered grad fragment feeds all ``N_STEPS`` x-atoms and each x fragment
    feeds all ``M_STEPS`` grad-atoms, amortizing the strided gathers over the atom grid.
    """
    if dtype != "bf16":
        raise ValueError(f"moe_wgrad flydsl kernel only supports bf16, got {dtype!r}")
    if block_n % WMMA_M != 0 or block_k % WMMA_N != 0:
        raise ValueError("block_n/block_k must be multiples of 16")

    gpu_arch = get_rocm_arch()
    wmma = WmmaHalf_m16n16k32(dtype)
    MUL_W = mul_routed_weight
    M_STEPS = block_n // WMMA_M   # grad-feature (MFMA-M) atoms
    N_STEPS = block_k // WMMA_N   # x-feature   (MFMA-N) atoms
    NACC = M_STEPS * N_STEPS
    KERNEL_NAME = (
        f"moe_wgrad_{dtype}{'_mw' if MUL_W else ''}_{block_n}x{block_k}_v1"
    )

    @flyc.kernel
    def wgrad_kernel(
        dW: fx.Pointer,          # [E, N, K] bf16 output
        X: fx.Pointer,           # [num_tokens, K] bf16
        GRAD: fx.Pointer,        # [num_tokens * top_k, N] bf16
        TOPK_W: fx.Pointer,      # [num_tokens * top_k] f32 (routed-slot indexed)
        SORTED: fx.Pointer,      # [padded] i32 routed-slot ids grouped by expert
        BLOCK_START: fx.Pointer,      # [E] i32 first sorted-block per expert
        BLOCKS_PER_EXPERT: fx.Pointer,  # [E] i32 blocks per expert
        N: fx.Int32,
        K: fx.Int32,
        num_valid_tokens: fx.Int32,
        top_k: fx.Int32,
    ):
        bf16 = T.bf16
        c0 = arith.constant(0, index=True)
        zero_bf = arith.constant(0.0, type=bf16)
        zero_f32 = arith.constant(0.0, type=T.f32)

        dW_rsrc = ptr_rsrc(dW)
        x_rsrc = ptr_rsrc(X)
        grad_rsrc = ptr_rsrc(GRAD)
        sorted_rsrc = ptr_rsrc(SORTED)
        bstart_rsrc = ptr_rsrc(BLOCK_START)
        bpe_rsrc = ptr_rsrc(BLOCKS_PER_EXPERT)
        if const_expr(MUL_W):
            w_rsrc = ptr_rsrc(TOPK_W)

        tid = fx.Int32(gpu.thread_id("x"))
        n_tile = fx.Int32(gpu.block_id("x"))   # along N (grad feature)
        k_tile = fx.Int32(gpu.block_id("y"))   # along K (x feature)
        expert = fx.Int32(gpu.block_id("z"))   # expert id

        lane = tid % WARP_SIZE
        lane_mod16 = lane % 16
        lane_div16 = lane // 16  # 0..3 -> which group of 8 slot values

        # first slot offset within the 32-tile handled by this lane (runtime, per-lane)
        k_slot_lane_idx = arith.index_cast(T.index, lane_div16 * A_FRAG)

        N_idx = arith.index_cast(T.index, N)
        K_idx = arith.index_cast(T.index, K)
        nvalid_idx = arith.index_cast(T.index, num_valid_tokens)
        topk_idx = arith.index_cast(T.index, top_k)

        n_block_base = n_tile * block_n
        k_block_base = k_tile * block_k

        # Per-atom feature columns (loop-invariant) + OOB guards.
        n_feat_idx = []   # grad feature column for A-atom mi, lane offset folded in
        n_feat_ok = []
        for mi in range_constexpr(M_STEPS):
            nf = arith.index_cast(T.index, n_block_base + mi * WMMA_M + lane_mod16)
            n_feat_idx.append(nf)
            n_feat_ok.append(arith.cmpi(arith.CmpIPredicate.ult, nf, N_idx))
        k_feat_idx = []   # x feature column for B-atom nj
        k_feat_ok = []
        for nj in range_constexpr(N_STEPS):
            kf = arith.index_cast(T.index, k_block_base + nj * WMMA_N + lane_mod16)
            k_feat_idx.append(kf)
            k_feat_ok.append(arith.cmpi(arith.CmpIPredicate.ult, kf, K_idx))

        # Per-expert routed-slot range.
        bstart = buffer_load_i32(bstart_rsrc, expert)
        nblocks = buffer_load_i32(bpe_rsrc, expert)
        base_slot = arith.index_cast(T.index, bstart) * arith.index(WGRAD_BLOCK_M)
        num_slots = arith.index_cast(T.index, nblocks) * arith.index(WGRAD_BLOCK_M)

        acc_init = [arith.constant_vector(0.0, T.vec(C_FRAG, T.f32)) for _ in range(NACC)]
        step = arith.index(WMMA_K)

        loop = scf.ForOp(c0, num_slots, step, iter_args=acc_init)
        with ir.InsertionPoint(loop.body):
            s_base = loop.induction_variable
            accs = [loop.body.arguments[1 + i] for i in range(NACC)]

            # Decode this lane's 8 slots once; reuse across all atoms.
            slot_safe = []
            token = []
            valid = []
            wv = []
            for j in range_constexpr(A_FRAG):
                slot_pos = base_slot + s_base + k_slot_lane_idx + arith.index(j)
                slot_id = buffer_load_i32_idx(sorted_rsrc, slot_pos)
                sidx = arith.index_cast(T.index, slot_id)
                vj = arith.cmpi(arith.CmpIPredicate.ult, sidx, nvalid_idx)
                ssafe = vj.select(sidx, c0)
                slot_safe.append(ssafe)
                token.append(arith.divui(ssafe, topk_idx))
                valid.append(vj)
                if const_expr(MUL_W):
                    w = buffer_load_f32(w_rsrc, ssafe)
                    wv.append(vj.select(w, zero_f32))

            # A (grad) fragments: reused across all N_STEPS x-atoms.
            a_frag = []
            for mi in range_constexpr(M_STEPS):
                elems = []
                for j in range_constexpr(A_FRAG):
                    off = valid[j].select(slot_safe[j] * N_idx + n_feat_idx[mi], c0)
                    g = buffer_load_bf16(grad_rsrc, off)
                    g = valid[j].select(g, zero_bf)
                    if const_expr(MUL_W):
                        g = arith.truncf(bf16, arith.mulf(arith.extf(T.f32, g), wv[j]))
                    elems.append(g)
                a_frag.append(vector.from_elements(T.vec(A_FRAG, bf16), elems))

            # B (x) fragments: reused across all M_STEPS grad-atoms.
            b_frag = []
            for nj in range_constexpr(N_STEPS):
                elems = []
                for j in range_constexpr(A_FRAG):
                    off = valid[j].select(token[j] * K_idx + k_feat_idx[nj], c0)
                    xv = buffer_load_bf16(x_rsrc, off)
                    elems.append(valid[j].select(xv, zero_bf))
                b_frag.append(vector.from_elements(T.vec(B_FRAG, bf16), elems))

            new_accs = []
            for mi in range_constexpr(M_STEPS):
                for nj in range_constexpr(N_STEPS):
                    new_accs.append(
                        wmma(a_frag[mi], b_frag[nj], accs[mi * N_STEPS + nj])
                    )
            scf.YieldOp(new_accs)

        accs = [loop.results[i] for i in range(NACC)]

        # Store: atom (mi,nj), lane holds c_n = k col (lane%16),
        # c_m rows = lane_div16*4 + ii (n rows) for ii in 0..3.
        E_NK_row = arith.index_cast(T.index, expert) * N_idx * K_idx
        for mi in range_constexpr(M_STEPS):
            for nj in range_constexpr(N_STEPS):
                acc = accs[mi * N_STEPS + nj]
                c_n_idx = arith.index_cast(
                    T.index, k_block_base + nj * WMMA_N + lane_mod16
                )
                k_ok = arith.cmpi(arith.CmpIPredicate.ult, c_n_idx, K_idx)
                for ii in range_constexpr(C_FRAG):
                    n_out = n_block_base + mi * WMMA_M + lane_div16 * C_FRAG + ii
                    n_out_idx = arith.index_cast(T.index, n_out)
                    in_bounds = arith.andi(
                        arith.cmpi(arith.CmpIPredicate.ult, n_out_idx, N_idx), k_ok
                    )
                    store_if = scf.IfOp(in_bounds, results_=[], has_else=False)
                    with ir.InsertionPoint(store_if.then_block):
                        val = vector.extract(
                            acc, static_position=[ii], dynamic_position=[]
                        )
                        out_off = E_NK_row + n_out_idx * K_idx + c_n_idx
                        buffer_store_bf16(dW_rsrc, out_off, arith.truncf(bf16, val))
                        scf.YieldOp([])

    @flyc.jit
    def launch_wgrad(
        dW: fx.Pointer,
        X: fx.Pointer,
        GRAD: fx.Pointer,
        TOPK_W: fx.Pointer,
        SORTED: fx.Pointer,
        BLOCK_START: fx.Pointer,
        BLOCKS_PER_EXPERT: fx.Pointer,
        N: fx.Int32,
        K: fx.Int32,
        num_valid_tokens: fx.Int32,
        top_k: fx.Int32,
        num_experts: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        gx = (N + block_n - 1) // block_n
        gy = (K + block_k - 1) // block_k
        gz = num_experts
        wgrad_kernel._func.__name__ = KERNEL_NAME
        wgrad_kernel(
            dW, X, GRAD, TOPK_W, SORTED, BLOCK_START, BLOCKS_PER_EXPERT,
            N, K, num_valid_tokens, top_k,
        ).launch(grid=(gx, gy, gz), block=(WARP_SIZE, 1, 1), stream=stream)

    return launch_wgrad


def buffer_load_i32(rsrc, off_i32):
    from flydsl.expr import buffer_ops
    return buffer_ops.buffer_load(rsrc, off_i32, vec_width=1, dtype=T.i32)


def buffer_load_i32_idx(rsrc, off_idx):
    from flydsl.expr import buffer_ops
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.i32)


def buffer_load_bf16(rsrc, off_idx):
    from flydsl.expr import buffer_ops
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.bf16)


def buffer_load_f32(rsrc, off_idx):
    from flydsl.expr import buffer_ops
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.f32)


def buffer_store_bf16(rsrc, off_idx, val):
    from flydsl.expr import buffer_ops
    buffer_ops.buffer_store(val, rsrc, off_idx)

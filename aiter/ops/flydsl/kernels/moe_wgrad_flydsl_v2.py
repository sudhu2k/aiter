# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE weight-gradient (wgrad) grouped GEMM in FlyDSL -- v2 (LDS transpose).

Same math as v0/v1::

    dW[e][n, k] = sum_{routed slot s of e} grad[s, n] * x[token(s), k]

but the contraction tile is staged through LDS and transposed on-read:

  1. Coalesced fill: each 32-slot contraction step loads ``grad[slot, n_feat]`` and
     ``x[token(slot), k_feat]`` into LDS as ``[slot(row), feature(col)]`` tiles with
     wide vector loads along the contiguous feature axis.
  2. Hardware transpose-read: ``ds_read_tr16_b64`` reads the ``[slot, feature]`` tile
     transposed into the MFMA A/B fragment layout ``[feature, slot]`` -- so the token
     slot becomes the matrix-core contraction axis with no VGPR shuffle and no strided
     global gather.

v2.1 -- multi-warp workgroup
----------------------------
A workgroup of ``warps_n x warps_k`` warps computes one ``block_n x block_k`` output
tile. All warps **cooperatively fill** the shared LDS contraction tile once per step
(amortizing the global gather), then each warp owns a ``(block_n/warps_n) x
(block_k/warps_k)`` sub-tile of 16x16 MFMA atoms, transpose-reading its own feature
columns out of the shared tile. ``warps_n = warps_k = 1`` reduces to the single-warp v2.

The transpose-read lane addressing mirrors the CK recipe in ``chunk_gated_delta_h.py``:
an 8-wide bf16 K-fragment is two ``ds_read_tr16_b64`` (lo / hi, offset by 4 rows)
shuffled together.
"""

from __future__ import annotations

import functools

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

__all__ = ["compile_moe_wgrad_v2", "WGRAD_BLOCK_M"]

# MFMA atom dims (16x16x32 bf16 on gfx950/gfx942).
WMMA_M = 16  # output rows  -> N (grad feature)
WMMA_N = 16  # output cols  -> K (x feature)
WMMA_K = 32  # contraction  -> token-slot tile
C_FRAG = 4   # f32 values per lane for the C (dW) fragment
WARP_SIZE = 64

import os

# Coalesced fill vector width (bf16 elems). 8 bf16 = 16 B = one global_load_dwordx4.
FILL_V = 8
# LDS stride padding (bf16 elems) to break bank conflicts on the transpose read.
# Must be a multiple of 4 (ds_read_tr16_b64 is an 8-byte read -> 4-bf16-aligned stride).
# Overridable via env for sweeps; default chosen from the LDS_PAD ATT sweep.
LDS_PAD = int(os.environ.get("MOE_WGRAD_LDS_PAD", "8"))

WGRAD_BLOCK_M = 32  # contraction (slot) step; matches the align block_size


@functools.lru_cache(maxsize=None)
def compile_moe_wgrad_v2(
    *,
    dtype: str = "bf16",
    mul_routed_weight: bool = False,
    block_n: int = 64,
    block_k: int = 64,
    warps_n: int = 1,
    warps_k: int = 1,
    topk: int | None = None,
):
    if dtype != "bf16":
        raise ValueError(f"moe_wgrad v2 flydsl kernel only supports bf16, got {dtype!r}")
    if block_n % (warps_n * WMMA_M) != 0 or block_k % (warps_k * WMMA_N) != 0:
        raise ValueError("block_n/block_k must be multiples of warps_*16")

    n_threads = warps_n * warps_k * WARP_SIZE
    if (WGRAD_BLOCK_M * block_n) % (n_threads * FILL_V) != 0:
        raise ValueError("32*block_n must be a multiple of n_threads*FILL_V")
    if (WGRAD_BLOCK_M * block_k) % (n_threads * FILL_V) != 0:
        raise ValueError("32*block_k must be a multiple of n_threads*FILL_V")

    gpu_arch = get_rocm_arch()
    MUL_W = mul_routed_weight
    WN = block_n // warps_n          # per-warp grad-feature span
    WK = block_k // warps_k          # per-warp x-feature span
    M_STEPS = WN // WMMA_M           # grad-feature atoms per warp (MFMA-M)
    N_STEPS = WK // WMMA_N           # x-feature atoms   per warp (MFMA-N)
    NACC = M_STEPS * N_STEPS

    SG = block_n + LDS_PAD           # grad LDS row stride (bf16 elems)
    SX = block_k + LDS_PAD           # x LDS row stride
    
    G_TILE_ELEMS = WGRAD_BLOCK_M * SG
    X_TILE_ELEMS = WGRAD_BLOCK_M * SX
    G_FILLS = (WGRAD_BLOCK_M * block_n) // (n_threads * FILL_V)
    X_FILLS = (WGRAD_BLOCK_M * block_k) // (n_threads * FILL_V)

    KERNEL_NAME = (
        f"moe_wgrad_{dtype}{'_mw' if MUL_W else ''}"
        f"_{block_n}x{block_k}_w{warps_n}x{warps_k}"
        f"{f'_tk{topk}' if topk is not None else ''}"
        f"_p{LDS_PAD}_v2"
    )

    # LDS allocation: double-buffered grad tile + x tile (2 bytes/bf16, 2 buffers each).
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    g_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = g_lds_off + G_TILE_ELEMS * 2 * 2
    x_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = x_lds_off + X_TILE_ELEMS * 2 * 2

    @flyc.kernel(known_block_size=[n_threads, 1, 1])
    def wgrad_kernel(
        dW: fx.Pointer,          # [E, N, K] bf16 output
        X: fx.Pointer,           # [num_tokens, K] bf16
        GRAD: fx.Pointer,        # [num_tokens * top_k, N] bf16
        TOPK_W: fx.Pointer,      # [num_tokens * top_k] f32
        SORTED: fx.Pointer,      # [padded] i32 routed-slot ids grouped by expert
        BLOCK_START: fx.Pointer,      # [E] i32
        BLOCKS_PER_EXPERT: fx.Pointer,  # [E] i32
        N: fx.Int32,
        K: fx.Int32,
        num_valid_tokens: fx.Int32,
        top_k: fx.Int32,
    ):
        bf16 = T.bf16
        c0 = arith.constant(0, index=True)

        dW_rsrc = ptr_rsrc(dW)
        x_rsrc = ptr_rsrc(X)
        grad_rsrc = ptr_rsrc(GRAD)
        sorted_rsrc = ptr_rsrc(SORTED)
        bstart_rsrc = ptr_rsrc(BLOCK_START)
        bpe_rsrc = ptr_rsrc(BLOCKS_PER_EXPERT)
        if const_expr(MUL_W):
            w_rsrc = ptr_rsrc(TOPK_W)

        base_ptr = allocator.get_base()
        g_lds_ptr = SmemPtr(base_ptr, g_lds_off, bf16, shape=(2 * G_TILE_ELEMS,))
        x_lds_ptr = SmemPtr(base_ptr, x_lds_off, bf16, shape=(2 * X_TILE_ELEMS,))
        g_lds = g_lds_ptr.get()
        x_lds = x_lds_ptr.get()

        tid = fx.Int32(gpu.thread_id("x"))
        n_tile = fx.Int32(gpu.block_id("x"))   # along N (grad feature)
        k_tile = fx.Int32(gpu.block_id("y"))   # along K (x feature)
        expert = fx.Int32(gpu.block_id("z"))   # expert id

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wn_id = wid // warps_k                 # warp row (grad feature)
        wk_id = wid % warps_k                  # warp col (x feature)
        lane_n = lane % 16                     # MFMA C col (k_feat)
        lane_m_base = lane // 16               # 0..3
        tr_k_group = (lane % 16) // 4          # 0..3
        tr_col_sub = lane % 4                  # 0..3

        warp_n_base = wn_id * fx.Int32(WN)     # grad-feature col base of this warp
        warp_k_base = wk_id * fx.Int32(WK)     # x-feature col base of this warp

        N_idx = arith.index_cast(T.index, N)
        K_idx = arith.index_cast(T.index, K)
        nvalid_idx = arith.index_cast(T.index, num_valid_tokens)
        # `token = slot // topk` is a per-element divide. With a compile-time topk the
        # divisor is constant so LLVM lowers it to a shift (pow2) or magic-number mul,
        # instead of the v_rcp/v_cvt software-division sequence used for a runtime arg.
        topk_idx = arith.index(topk) if topk is not None else arith.index_cast(T.index, top_k)

        n_block_base = n_tile * block_n
        k_block_base = k_tile * block_k
        n_base_idx = arith.index_cast(T.index, n_block_base)
        k_base_idx = arith.index_cast(T.index, k_block_base)

        # Per-expert routed-slot range.
        bstart = buffer_load_i32(bstart_rsrc, expert)
        nblocks = buffer_load_i32(bpe_rsrc, expert)
        base_slot = arith.index_cast(T.index, bstart) * arith.index(WGRAD_BLOCK_M)
        num_slots = arith.index_cast(T.index, nblocks) * arith.index(WGRAD_BLOCK_M)

        zero_v = arith.constant_vector(0.0, T.vec(FILL_V, bf16))
        CPR_G = block_n // FILL_V
        CPR_X = block_k // FILL_V

        # ---- gather (global -> registers) and store (registers -> LDS) closures ----
        def _tile_idx(i, cpr):
            chunk = tid + fx.Int32(i * n_threads)
            slot_i32 = chunk // fx.Int32(cpr)
            feat_i32 = (chunk % fx.Int32(cpr)) * fx.Int32(FILL_V)
            slot_idx = arith.index_cast(T.index, slot_i32)
            feat_idx = arith.index_cast(T.index, feat_i32)
            # slot_i32/feat_i32 are exposed for the write-side swizzle (bf16-elem
            # coords); existing callers keep using the index [0]/[1] entries.
            return slot_idx, feat_idx, slot_i32, feat_i32

        # Fill is split three ways so each latency is overlapped by independent work:
        #  * ``load_slot_ids`` issues the (indirect) ``sorted`` index loads. These are
        #    prefetched *two* steps ahead and carried across the loop as iter_args, so the
        #    ~500cyc index latency overlaps a full MFMA step instead of stalling in front
        #    of the data-address math (the dominant ``vmcnt`` before this change).
        #  * ``gather_*`` consume the already-resident ids and only *issue* the data loads.
        #  * ``store_*`` do the mask/scale + LDS write -- emitted after the MFMA behind a
        #    ``sched_barrier`` so their data-load ``vmcnt`` lands past the matrix ops.
        def load_slot_ids(s_base_idx):
            g_ids = [
                buffer_load_i32_idx(sorted_rsrc, base_slot + s_base_idx + _tile_idx(i, CPR_G)[0])
                for i in range_constexpr(G_FILLS)
            ]

            x_ids = [
                buffer_load_i32_idx(sorted_rsrc, base_slot + s_base_idx + _tile_idx(i, CPR_X)[0])
                for i in range_constexpr(X_FILLS)
            ]
            return g_ids, x_ids

        def gather_grad(g_ids):
            raw = []
            for i in range_constexpr(G_FILLS):
                feat_idx = _tile_idx(i, CPR_G)[1]
                sidx = arith.index_cast(T.index, g_ids[i])
                valid = arith.cmpi(arith.CmpIPredicate.ult, sidx, nvalid_idx)
                goff = valid.select(sidx, c0) * N_idx + n_base_idx + feat_idx
                gvec = buffer_load_bf16_vec(grad_rsrc, goff, FILL_V)
                w = buffer_load_f32(w_rsrc, valid.select(sidx, c0)) if const_expr(MUL_W) else None
                raw.append((gvec, valid, w))
            return raw

        def gather_x(x_ids):
            raw = []
            for i in range_constexpr(X_FILLS):
                feat_idx = _tile_idx(i, CPR_X)[1]
                sidx = arith.index_cast(T.index, x_ids[i])
                valid = arith.cmpi(arith.CmpIPredicate.ult, sidx, nvalid_idx)
                token = arith.divui(valid.select(sidx, c0), topk_idx)
                xoff = token * K_idx + k_base_idx + feat_idx
                xvec = buffer_load_bf16_vec(x_rsrc, xoff, FILL_V)
                raw.append((xvec, valid, None))
            return raw

        def store_grad(raw, buf_elem):
            for i in range_constexpr(G_FILLS):
                gvec, valid, w = raw[i]
                if const_expr(MUL_W):
                    gvec = _scale_bf16_vec(gvec, w)
                slot_idx, feat_idx, slot_i32, feat_i32 = _tile_idx(i, CPR_G)
                vector.store(
                    valid.select(gvec, zero_v), g_lds,
                    [buf_elem + slot_idx * arith.index(SG) + feat_idx], alignment=16,
                )

        def store_x(raw, buf_elem):
            for i in range_constexpr(X_FILLS):
                xvec, valid, _ = raw[i]
                slot_idx, feat_idx, slot_i32, feat_i32 = _tile_idx(i, CPR_X)
                vector.store(
                    valid.select(xvec, zero_v), x_lds,
                    [buf_elem + slot_idx * arith.index(SX) + feat_idx], alignment=16,
                )

        def compute(accs, g_buf_byte, x_buf_byte):
            # A fragments are read one-per-mi to keep VGPR pressure low (hoisting all of
            # them costs occupancy, which the large tile-count shapes depend on). B
            # fragments are loop-invariant across mi, so they are fetched exactly once --
            # but interleaved with the mi==0 MFMAs (see below) rather than as a serial
            # burst up front, so their ds_read latency overlaps compute. The MFMA region
            # runs at raised priority so the matrix pipe stays fed while reads are in
            # flight instead of the scheduler round-robining to a stalled wave.
            def read_a(mi):
                return _tr_read_frag(
                    g_lds_off, SG, warp_n_base, mi * WMMA_M,
                    lane_m_base, tr_k_group, tr_col_sub, g_buf_byte,
                )

            def read_b(nj):
                return _tr_read_frag(
                    x_lds_off, SX, warp_k_base, nj * WMMA_N,
                    lane_m_base, tr_k_group, tr_col_sub, x_buf_byte,
                )

            new_accs = [None] * NACC
            b_frags = [None] * N_STEPS
            a_next = read_a(0)
            b_frags[0] = read_b(0)  # first B needed for the very first MFMA
            rocdl.s_setprio(1)
            for mi in range_constexpr(M_STEPS):
                a_cur = a_next
                if const_expr(mi + 1 < M_STEPS):
                    a_next = read_a(mi + 1)  # prefetch next A while MFMA-ing current
                for nj in range_constexpr(N_STEPS):
                    # Prefetch the next B fragment during the first mi only; its ds_read
                    # then overlaps this step's MFMA and all later mi reuse the resident
                    # frag. No extra VGPR: every B frag is live from mi==0 onward anyway.
                    if const_expr(mi == 0 and nj + 1 < N_STEPS):
                        b_frags[nj + 1] = read_b(nj + 1)
                    idx = mi * N_STEPS + nj
                    new_accs[idx] = rocdl.mfma_f32_16x16x32_bf16(
                        T.vec(C_FRAG, T.f32), [a_cur, b_frags[nj], accs[idx], 0, 0, 0]
                    )
                rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            return new_accs

        def _barrier():
            # Per-step ping-pong sync
            gpu.barrier()

        acc_init = [arith.constant_vector(0.0, T.vec(C_FRAG, T.f32)) for _ in range(NACC)]
        step = arith.index(WMMA_K)

        # When sharing slot ids (block_n == block_k), only the grad ids are carried through
        # the loop; the x ids alias them. Otherwise both id vectors are carried.
        def _pack_ids(g_ids, x_ids):
            return (g_ids + x_ids)

        # 2-stage ping-pong pipeline: MFMA the current LDS buffer while the next step's
        # global gather is in flight, then stage it into the *other* buffer -- so a single
        # barrier per step suffices (no store->barrier->compute serialization). Slot ids run
        # one step ahead of the data (loaded here, consumed next iteration) via iter_args.
        g_ids0, x_ids0 = load_slot_ids(c0)
        store_grad(gather_grad(g_ids0), c0)
        store_x(gather_x(x_ids0), c0)
        _barrier()
        g_ids_next, x_ids_next = load_slot_ids(step)

        loop = scf.ForOp(
            c0, num_slots, step, iter_args=acc_init + _pack_ids(g_ids_next, x_ids_next)
        )
        with ir.InsertionPoint(loop.body):
            s_base = loop.induction_variable
            accs = [loop.body.arguments[1 + i] for i in range(NACC)]
            g_ids = [loop.body.arguments[1 + NACC + i] for i in range(G_FILLS)]
            x_ids = [loop.body.arguments[1 + NACC + G_FILLS + i] for i in range(X_FILLS)]

            it = fx.Int32(arith.index_cast(T.i32, s_base)) // fx.Int32(WMMA_K)
            cur = it % fx.Int32(2)
            nxt = fx.Int32(1) - cur
            cur_g_byte = cur * fx.Int32(G_TILE_ELEMS * 2)
            cur_x_byte = cur * fx.Int32(X_TILE_ELEMS * 2)
            nxt_g_elem = arith.index_cast(T.index, nxt * fx.Int32(G_TILE_ELEMS))
            nxt_x_elem = arith.index_cast(T.index, nxt * fx.Int32(X_TILE_ELEMS))

            # Issue next step's data gather using the slot ids carried in (already resident).
            g_regs_next = gather_grad(g_ids)
            x_regs_next = gather_x(x_ids)
            # Prefetch the slot ids two steps ahead; carried out to the next iteration so
            # their global-load latency overlaps this step's MFMA.
            g_ids_nxt, x_ids_nxt = load_slot_ids(s_base + step + step)

            new_accs = compute(accs, cur_g_byte, cur_x_byte)

            # Pin the mask+store (consumers of the in-flight next-tile loads) *after* the
            # MFMA so their `s_waitcnt vmcnt` isn't hoisted in front of the matrix ops --
            # the matrix pipe then runs while those loads are still in flight.
            rocdl.sched_barrier(0)
            store_grad(g_regs_next, nxt_g_elem)
            store_x(x_regs_next, nxt_x_elem)
            _barrier()
            scf.YieldOp(new_accs + _pack_ids(g_ids_nxt, x_ids_nxt))

        accs = [loop.results[i] for i in range(NACC)]

        # Epilogue: C[m=n_feat, n=k_feat], lane holds 4 rows.
        E_NK_row = arith.index_cast(T.index, expert) * N_idx * K_idx
        for mi in range_constexpr(M_STEPS):
            for nj in range_constexpr(N_STEPS):
                acc = accs[mi * N_STEPS + nj]
                c_n = k_block_base + warp_k_base + fx.Int32(nj * WMMA_N) + lane_n
                c_n_idx = arith.index_cast(T.index, c_n)
                k_ok = arith.cmpi(arith.CmpIPredicate.ult, c_n_idx, K_idx)
                for ii in range_constexpr(C_FRAG):
                    n_out = (
                        n_block_base + warp_n_base + fx.Int32(mi * WMMA_M)
                        + lane_m_base * C_FRAG + fx.Int32(ii)
                    )
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
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gx = (N + block_n - 1) // block_n
        gy = (K + block_k - 1) // block_k
        gz = num_experts
        wgrad_kernel._func.__name__ = KERNEL_NAME
        wgrad_kernel(
            dW, X, GRAD, TOPK_W, SORTED, BLOCK_START, BLOCKS_PER_EXPERT,
            N, K, num_valid_tokens, top_k,
        ).launch(grid=(gx, gy, gz), block=(n_threads, 1, 1), stream=stream)

    return launch_wgrad

def _tr_read_frag(
    lds_off, stride, warp_col_base, col_const, lane_m_base, tr_k_group, tr_col_sub, buf_byte=None
):
    """ds_read_tr16_b64 A/B fragment for a 16-wide feature column base from a [slot,feat] tile.

    The atom's first feature column is ``warp_col_base + col_const`` where
    ``warp_col_base`` is the (runtime) per-warp base and ``col_const`` is the
    compile-time per-fragment shift (``mi*WMMA_M`` / ``nj*WMMA_N``). Keeping them
    separate lets the base LDS pointer stay loop- *and* fragment-invariant (so CSE
    collapses it to a single live address) while the per-fragment column shift and
    the lo->hi row shift fold into the ``ds_read`` immediate ``offset`` field
    instead of each consuming a VGPR. ``buf_byte`` optionally selects a ping-pong
    buffer (runtime byte offset). Returns an 8xbf16 MFMA fragment (contraction =
    32 slots, feature = 16).
    """
    col_run = warp_col_base + tr_col_sub * fx.Int32(4)
    row_lo = lane_m_base * fx.Int32(8) + tr_k_group  # 0..31 (bt_s == 0)
    base_elem = row_lo * fx.Int32(stride) + col_run
    base_byte = base_elem * fx.Int32(2) + fx.Int32(lds_off)
    if buf_byte is not None:
        base_byte = base_byte + buf_byte
    byte_idx = arith.index_cast(T.index, base_byte)
    byte_i64 = arith.index_cast(T.i64, byte_idx)
    base_ptr = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), byte_i64).result

    col_byte = 2 * col_const     # compile-time feature-column shift (bytes)
    hi_byte = 2 * 4 * stride     # compile-time lo->hi row shift (bytes, 4 rows apart)
    lo = _ds_read_tr_bf16x4(base_ptr, col_byte)
    hi = _ds_read_tr_bf16x4(base_ptr, col_byte + hi_byte)
    return lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7])


def _ds_read_tr_bf16x4(base_ptr, const_byte=0):
    # `const_byte` is a compile-time byte offset added via an inbounds getelementptr
    # so the AMDGPU backend folds it into the ds_read `offset:` immediate rather than
    # materializing a distinct address VGPR per fragment.
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


def _scale_bf16_vec(vec, w_f32):
    vf = arith.extf(T.vec(FILL_V, T.f32), vec)
    wv = vector.broadcast(T.vec(FILL_V, T.f32), w_f32)
    pf = arith.mulf(vf, wv)
    return arith.truncf(T.vec(FILL_V, T.bf16), pf)


def buffer_load_i32(rsrc, off_i32):
    return buffer_ops.buffer_load(rsrc, off_i32, vec_width=1, dtype=T.i32)


def buffer_load_i32_idx(rsrc, off_idx):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.i32)


def buffer_load_bf16_vec(rsrc, off_idx, v):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=v, dtype=T.bf16)


def buffer_load_f32(rsrc, off_idx):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.f32)


def buffer_store_bf16(rsrc, off_idx, val):
    buffer_ops.buffer_store(val, rsrc, off_idx)

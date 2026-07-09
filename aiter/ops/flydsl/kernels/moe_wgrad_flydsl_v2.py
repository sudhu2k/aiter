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
shuffled together. The tile is stride-padded (not XOR-swizzled): a row-dependent XOR
would break the hardware transpose alignment.
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

# Row-group XOR swizzle: an alternative to stride padding for LDS bank-conflict
# relief. When enabled it lets ``LDS_PAD`` drop to 0 (smaller tile -> higher
# occupancy) while still spreading the transpose read across banks. See
# ``_swz_col_i32`` for the correctness argument; A/B via MOE_WGRAD_SWIZZLE=1.
SWIZZLE = os.environ.get("MOE_WGRAD_SWIZZLE", "0") == "1"

WGRAD_BLOCK_M = 32  # contraction (slot) step; matches the align block_size

# Manual `s_waitcnt ; s_barrier` scaffolding (default off). NOTE: on the current
# pipeline this is a *no-op* vs `gpu.barrier()` -- ATT (att_v2_idxpf) shows the
# compiler already lowers the per-step barrier to `s_waitcnt lgkmcnt(0)` with no
# `vmcnt` wait, because the staged loads are consumed (vmcnt-waited) at their store
# site. The value of this helper is that it is the reusable primitive for the
# HK-style fine-grained mi-group barriers (explicit partial counts) that the
# read->MFMA `lgkmcnt` work needs; the flag just lets us A/B the plain swap.
BARRIER_LGKMCNT_ONLY = os.environ.get("MOE_WGRAD_BARRIER_LGKMCNT_ONLY", "0") == "1"

# When set, emit a CK-style DS_READ<->MFMA cadence (sched_group_barrier via
# sched_mfma/sched_dsrd) over the compute region so the LDS transpose reads interleave
# with the matrix ops instead of forming one leading ds_read train that hits a single
# lgkmcnt(0) with the whole backlog outstanding. Targets the read->MFMA lgkmcnt stall.
SCHED_CADENCE = os.environ.get("MOE_WGRAD_SCHED_CADENCE", "0") == "1"

# When set, double-buffer the A (grad) MFMA fragment across two distinct register sets:
# a ``sched_barrier`` pins each ``read_a(mi+1)`` above tile ``mi``'s MFMAs so the read
# cannot sink below them (which would let the allocator alias a_cur/a_next into the same
# regs and force a full ``lgkmcnt(0)``). Kept live, the two fragments get distinct regs
# and LLVM emits a *partial* ``lgkmcnt`` -- the next A read overlaps the current MFMAs
# instead of stalling in front of them. Costs +4 VGPR (stays within the 4-wave bucket).
# Default on (measured +2-11%, biggest tiles gain most); set to 0 to A/B the old path.
ABUF = os.environ.get("MOE_WGRAD_ABUF", "1") == "1"


def _waitcnt_barrier(vmcnt=63, lgkmcnt=63):
    """Emit ``s_waitcnt {vmcnt/lgkmcnt} ; s_barrier`` via inline asm.

    Lets the caller set explicit wait counts ahead of ``s_barrier`` instead of
    relying on the compiler's default lowering -- e.g. ``lgkmcnt=0`` with
    ``vmcnt=63`` (the "don't wait" sentinel) syncs only the LDS ping-pong. 63 is the
    unconstrained sentinel for both counters.
    """
    wc = []
    if vmcnt < 63:
        wc.append(f"vmcnt({vmcnt})")
    if lgkmcnt < 63:
        wc.append(f"lgkmcnt({lgkmcnt})")
    parts = []
    if wc:
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    _llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


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
    # Row-group XOR swizzle masks: fold the transpose-atom group into each
    # tile's FILL_V-chunk count so the swizzled column stays in range.
    if SWIZZLE and (block_n & (block_n - 1)) != 0:
        raise ValueError("MOE_WGRAD_SWIZZLE requires power-of-2 block_n")
    if SWIZZLE and (block_k & (block_k - 1)) != 0:
        raise ValueError("MOE_WGRAD_SWIZZLE requires power-of-2 block_k")
    SWZ_MASK_G = block_n // FILL_V - 1
    SWZ_MASK_X = block_k // FILL_V - 1
    G_TILE_ELEMS = WGRAD_BLOCK_M * SG
    X_TILE_ELEMS = WGRAD_BLOCK_M * SX
    G_FILLS = (WGRAD_BLOCK_M * block_n) // (n_threads * FILL_V)
    X_FILLS = (WGRAD_BLOCK_M * block_k) // (n_threads * FILL_V)

    # The slot ids are a property of the 32-slot contraction step, not of the grad/x
    # tile: for step s, row r has id sorted[base_slot + s + r] regardless of which tile
    # consumes it. When block_n == block_k the grad and x fill mappings (CPR_G vs CPR_X)
    # are identical, so g_ids[i] and x_ids[i] are the same load -- issue them once and
    # feed both gathers. (The compiler already CSEs the duplicate; this just keeps the
    # source honest and the carried iter_args minimal.)
    SHARE_SLOT_IDS = block_n == block_k

    KERNEL_NAME = (
        f"moe_wgrad_{dtype}{'_mw' if MUL_W else ''}"
        f"_{block_n}x{block_k}_w{warps_n}x{warps_k}"
        f"{f'_tk{topk}' if topk is not None else ''}"
        f"{f'_swz' if SWIZZLE else f'_p{LDS_PAD}'}_v2"
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
            # When block_n == block_k the x fill mapping is identical, so the x slot ids
            # are the same loads -- reuse g_ids instead of re-issuing them. Use a ternary
            # (not an if-statement): FlyDSL rewrites `if` into a scoped conditional, and the
            # else-branch comprehension must stay unevaluated when sharing.
            x_ids = g_ids if SHARE_SLOT_IDS else [
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
                if SWIZZLE:
                    feat_idx = arith.index_cast(T.index, _swz_col_i32(slot_i32, feat_i32, SWZ_MASK_G))
                vector.store(
                    valid.select(gvec, zero_v), g_lds,
                    [buf_elem + slot_idx * arith.index(SG) + feat_idx], alignment=16,
                )

        def store_x(raw, buf_elem):
            for i in range_constexpr(X_FILLS):
                xvec, valid, _ = raw[i]
                slot_idx, feat_idx, slot_i32, feat_i32 = _tile_idx(i, CPR_X)
                if SWIZZLE:
                    feat_idx = arith.index_cast(T.index, _swz_col_i32(slot_i32, feat_i32, SWZ_MASK_X))
                vector.store(
                    valid.select(xvec, zero_v), x_lds,
                    [buf_elem + slot_idx * arith.index(SX) + feat_idx], alignment=16,
                )

        def compute(accs, g_buf_byte, x_buf_byte):
            # B fragments (reused across all mi) are read once up front. A fragments are
            # read one-per-mi to keep VGPR pressure low (hoisting all of them costs
            # occupancy, which the large tile-count shapes depend on). The MFMA region
            # runs at raised priority so the matrix pipe stays fed while reads are in
            # flight instead of the scheduler round-robining to a stalled wave.
            def read_a(mi):
                col_base = warp_n_base + fx.Int32(mi * WMMA_M)
                return _tr_read_frag(
                    g_lds_off, SG, col_base, lane_m_base, tr_k_group, tr_col_sub,
                    SWZ_MASK_G, g_buf_byte,
                )

            b_frags = []
            for nj in range_constexpr(N_STEPS):
                col_base = warp_k_base + fx.Int32(nj * WMMA_N)
                b_frags.append(
                    _tr_read_frag(
                        x_lds_off, SX, col_base, lane_m_base, tr_k_group, tr_col_sub,
                        SWZ_MASK_X, x_buf_byte,
                    )
                )
            new_accs = [None] * NACC
            a_next = read_a(0)
            rocdl.s_setprio(1)
            for mi in range_constexpr(M_STEPS):
                a_cur = a_next
                if const_expr(mi + 1 < M_STEPS):
                    a_next = read_a(mi + 1)  # prefetch next A while MFMA-ing current
                    if ABUF:
                        # Fence the next-A read above this tile's MFMAs so it can't sink
                        # below them; keeps a_cur/a_next in distinct regs and turns the
                        # pre-MFMA wait into a partial lgkmcnt (overlap, not full stall).
                        rocdl.sched_barrier(0)
                for nj in range_constexpr(N_STEPS):
                    idx = mi * N_STEPS + nj
                    new_accs[idx] = rocdl.mfma_f32_16x16x32_bf16(
                        T.vec(C_FRAG, T.f32), [a_cur, b_frags[nj], accs[idx], 0, 0, 0]
                    )
            if SCHED_CADENCE:
                # Describe the intended schedule of the region above: the shared B
                # fragments (2 ds_read each) up front -- the first MFMA needs all of
                # them -- then, per mi, the A fragment (2 ds_read) interleaved right
                # before its N_STEPS MFMAs. This keeps reads spread through the MFMA
                # train so each lgkmcnt waits on only its group, not the full backlog.
                rocdl.sched_barrier(0)
                rocdl.sched_dsrd(2 * N_STEPS)
                for _mi in range_constexpr(M_STEPS):
                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(N_STEPS)
                rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            return new_accs

        def _barrier():
            # Per-step ping-pong sync. By default a plain workgroup barrier (compiler
            # emits vmcnt(0)+lgkmcnt(0)); with the flag, an LDS-only barrier that keeps
            # the hidden global loads / slot-id prefetch in flight across the barrier.
            if BARRIER_LGKMCNT_ONLY:
                _waitcnt_barrier(lgkmcnt=0)
            else:
                gpu.barrier()

        acc_init = [arith.constant_vector(0.0, T.vec(C_FRAG, T.f32)) for _ in range(NACC)]
        step = arith.index(WMMA_K)

        # When sharing slot ids (block_n == block_k), only the grad ids are carried through
        # the loop; the x ids alias them. Otherwise both id vectors are carried.
        def _pack_ids(g_ids, x_ids):
            return g_ids if SHARE_SLOT_IDS else (g_ids + x_ids)

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
            x_ids = g_ids if SHARE_SLOT_IDS else [
                loop.body.arguments[1 + NACC + G_FILLS + i] for i in range(X_FILLS)
            ]

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


def _swz_col_i32(row, col, chunk_mask):
    """Row-group XOR swizzle on the feature column (bf16 elems); no-op unless ``SWIZZLE``.

    Keyed on the 4-row transpose-atom group (``row >> 2``): within one
    ``ds_read_tr16_b64`` the 16 participating lanes all share the same
    ``lane_m_base`` (row // 8), hence the same group, so the XOR offset is
    constant across the instruction and the HW transpose stays coherent
    (a per-row XOR would shear the block and corrupt the transpose).

    The XOR is applied at FILL_V (8-bf16 = 16 B) granularity so the coalesced
    vector store stays aligned; ``chunk_mask = tile_cols//FILL_V - 1`` folds the
    group into the tile's chunk count so the swizzled column stays in range.
    The *same* function is applied on the LDS write and the transpose read, so
    it is a bijection on (row, col) -> physical slot and correct by construction
    regardless of ``chunk_mask``.
    """
    if not SWIZZLE:
        return col
    grp = (row >> fx.Int32(2)) & fx.Int32(chunk_mask)
    return col ^ (grp << fx.Int32(3))  # 3 == log2(FILL_V)


def _tr_read_frag(
    lds_off, stride, feat_col_base, lane_m_base, tr_k_group, tr_col_sub, swz_mask, buf_byte=None
):
    """ds_read_tr16_b64 A/B fragment for a 16-wide feature column base from a [slot,feat] tile.

    ``feat_col_base`` is the (runtime) element column of the atom's first feature.
    ``buf_byte`` optionally selects a ping-pong buffer (runtime byte offset).
    Returns an 8xbf16 MFMA fragment (contraction = 32 slots, feature = 16).
    """
    col_tr = feat_col_base + tr_col_sub * fx.Int32(4)
    row_lo = lane_m_base * fx.Int32(8) + tr_k_group  # 0..31 (bt_s == 0)
    row_hi = row_lo + fx.Int32(4)                    # lo/hi are 4 rows apart

    def _byte(row):
        # lo and hi land in different 4-row groups, so each needs its own swizzle
        # (can't just add 4 rows of bytes to the lo address once swizzled).
        col = _swz_col_i32(row, col_tr, swz_mask)
        elem = row * fx.Int32(stride) + col
        b = elem * fx.Int32(2) + fx.Int32(lds_off)
        if buf_byte is not None:
            b = b + buf_byte
        return b

    lo = _ds_read_tr_bf16x4(_byte(row_lo))
    hi = _ds_read_tr_bf16x4(_byte(row_hi))
    return lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7])


def _ds_read_tr_bf16x4(lds_byte_offset):
    byte_idx = arith.index_cast(T.index, lds_byte_offset)
    byte_i64 = arith.index_cast(T.i64, byte_idx)
    ptr = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), byte_i64).result
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

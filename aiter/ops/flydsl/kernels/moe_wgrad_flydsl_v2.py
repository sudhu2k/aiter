# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE weight-gradient (wgrad) grouped GEMM in FlyDSL -- v2 (LDS transpose).

Route-list contract (matches the Triton ``fused_route_list_moe_wgrad`` kernel in
TransformerEngine): the gradient operand is the *compact* ``[num_routes, N]`` buffer
and ``SORTED`` maps each padded route slot to a received-token row::

    dW[e][n, k] = sum_{routed slot s of e, valid} grad[route_start[e] + local(s), n]
                                                 * x[SORTED[s], k]

where ``local(s)`` is the slot's offset within expert ``e``'s block-padded range and
padding slots (``SORTED[s] == num_recv_tokens``) are masked to zero. The grad walk is a
plain contiguous row scan (no ``SORTED`` indirection); ``SORTED`` is only consulted for
the ``x`` gather token and the padding mask.

The contraction tile is staged through LDS and transposed on-read:

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
from flydsl.expr import arith, buffer_ops, const_expr, gpu, ptrtoint, range_constexpr, rocdl, vector
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

# Debug toggle: when 0, the DMA path uses an identity (no-op) chunk swizzle so the LDS
# tile is contiguous but un-permuted -- lets us isolate DMA-fill correctness from the
# XOR-swizzle correctness. Default 1 (swizzle on).
_DMA_SWZ_ON = int(os.environ.get("MOE_WGRAD_DMA_SWZ", "1"))

WGRAD_BLOCK_M = 32  # contraction (slot) step; matches the align block_size


@functools.lru_cache(maxsize=None)
def compile_moe_wgrad_v2(
    *,
    dtype: str = "bf16",
    block_n: int = 64,
    block_k: int = 64,
    warps_n: int = 1,
    warps_k: int = 1,
    accumulate: bool = False,
    out_dtype: str = "bf16",
    swap_gather: bool = False,
    dma_swizzle: bool = False,
    pipe_stages: int = 2,
):
    if dtype != "bf16":
        raise ValueError(f"moe_wgrad v2 flydsl kernel only supports bf16, got {dtype!r}")
    if out_dtype not in ("bf16", "fp32"):
        raise ValueError(f"moe_wgrad v2 out_dtype must be 'bf16' or 'fp32', got {out_dtype!r}")
    if block_n % (warps_n * WMMA_M) != 0 or block_k % (warps_k * WMMA_N) != 0:
        raise ValueError("block_n/block_k must be multiples of warps_*16")
    if dma_swizzle and swap_gather:
        raise ValueError("moe_wgrad v2 dma_swizzle does not support swap_gather yet")
    if pipe_stages < 2:
        raise ValueError("moe_wgrad v2 pipe_stages must be >= 2")
    if pipe_stages > 2 and not dma_swizzle:
        raise ValueError("moe_wgrad v2 pipe_stages > 2 requires dma_swizzle")
    # Number of LDS ping-pong buffers = pipeline depth; prefetch distance = NUM_BUF - 1.
    # Deeper pipelines give each in-flight DMA more MFMA iterations to hide behind (the
    # HK 4-wave recipe), at the cost of NUM_BUF x the LDS footprint.
    NUM_BUF = pipe_stages

    n_threads = warps_n * warps_k * WARP_SIZE
    if (WGRAD_BLOCK_M * block_n) % (n_threads * FILL_V) != 0:
        raise ValueError("32*block_n must be a multiple of n_threads*FILL_V")
    if (WGRAD_BLOCK_M * block_k) % (n_threads * FILL_V) != 0:
        raise ValueError("32*block_k must be a multiple of n_threads*FILL_V")

    gpu_arch = get_rocm_arch()
    WN = block_n // warps_n          # per-warp grad-feature span
    WK = block_k // warps_k          # per-warp x-feature span
    M_STEPS = WN // WMMA_M           # grad-feature atoms per warp (MFMA-M)
    N_STEPS = WK // WMMA_N           # x-feature atoms   per warp (MFMA-N)
    NACC = M_STEPS * N_STEPS

    # DMA fill writes each lane's 16B contiguously into LDS (no per-row pad possible),
    # so the swizzle path uses an un-padded stride and breaks bank conflicts with an
    # XOR chunk-swizzle instead (see _swz_chunk / the fill + transpose-read below).
    SG = block_n if dma_swizzle else block_n + LDS_PAD   # grad LDS row stride (bf16 elems)
    SX = block_k if dma_swizzle else block_k + LDS_PAD   # x LDS row stride
    # Swizzle granule = FILL_V bf16 (one 16B DMA unit). ``_swz_chunk`` XORs the feature
    # chunk index with the slot so consecutive contraction slots land on distinct banks.
    CPR_G_SWZ = block_n // FILL_V     # feature chunks per grad row
    CPR_X_SWZ = block_k // FILL_V     # feature chunks per x row
    
    G_TILE_ELEMS = WGRAD_BLOCK_M * SG
    X_TILE_ELEMS = WGRAD_BLOCK_M * SX
    G_FILLS = (WGRAD_BLOCK_M * block_n) // (n_threads * FILL_V)
    X_FILLS = (WGRAD_BLOCK_M * block_k) // (n_threads * FILL_V)

    out_is_f32 = out_dtype == "fp32"
    KERNEL_NAME = (
        f"moe_wgrad_routelist_{dtype}"
        f"_{block_n}x{block_k}_w{warps_n}x{warps_k}"
        f"_p{LDS_PAD}"
        f"{'_o32' if out_is_f32 else ''}{'_acc' if accumulate else ''}"
        f"{'_swp' if swap_gather else ''}{'_dsz' if dma_swizzle else ''}"
        f"{'_s' + str(pipe_stages) if pipe_stages != 2 else ''}_v2"
    )

    # LDS allocation: NUM_BUF-buffered grad tile + x tile (2 bytes/bf16). The non-DMA
    # path always ping-pongs 2 buffers; the DMA path uses NUM_BUF == pipe_stages.
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    g_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = g_lds_off + G_TILE_ELEMS * 2 * NUM_BUF
    x_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = x_lds_off + X_TILE_ELEMS * 2 * NUM_BUF
    # The DMA path backs LDS with a raw llvm addrspace(3) global (buffer_load_lds needs a
    # real global for M0; the memref allocator does not work). Same offsets/size.
    LDS_TOTAL_BYTES = allocator._align(allocator.ptr, 128)
    LDS_SYM = KERNEL_NAME + "_lds"

    @flyc.kernel(known_block_size=[n_threads, 1, 1])
    def wgrad_kernel(
        dW: fx.Pointer,          # [E, N, K] output (bf16, or fp32 when out_dtype='fp32')
        X: fx.Pointer,           # [num_recv_tokens, K] bf16 (received-token activations)
        GRAD: fx.Pointer,        # [num_routes, N] bf16 (compact per-route gradient)
        SORTED: fx.Pointer,      # [padded] i32 received-token row per route slot (sentinel = num_recv_tokens)
        BLOCK_START: fx.Pointer,      # [E] i32 (block units)
        BLOCKS_PER_EXPERT: fx.Pointer,  # [E] i32
        ROUTE_START: fx.Pointer,  # [E] i32 (compact first-route index = cumsum(counts) - counts)
        N: fx.Int32,
        K: fx.Int32,
        num_recv_tokens: fx.Int32,
    ):
        bf16 = T.bf16
        c0 = arith.constant(0, index=True)

        dW_rsrc = ptr_rsrc(dW)
        x_rsrc = ptr_rsrc(X)
        grad_rsrc = ptr_rsrc(GRAD)
        sorted_rsrc = ptr_rsrc(SORTED)
        bstart_rsrc = ptr_rsrc(BLOCK_START)
        bpe_rsrc = ptr_rsrc(BLOCKS_PER_EXPERT)
        rstart_rsrc = ptr_rsrc(ROUTE_START)

        if const_expr(dma_swizzle):
            # DMA path: raw addrspace(3) global backs LDS; reads + DMA both GEP off it.
            smem_raw_ptr = _llvm.mlir_addressof(ir.Type.parse("!llvm.ptr<3>"), LDS_SYM)
            g_lds = None
            x_lds = None
        else:
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
        nrecv_idx = arith.index_cast(T.index, num_recv_tokens)

        # DMA fill cannot mask padding slots in registers, so bound the x resource to its
        # real [num_recv, K] extent: the sentinel token (== num_recv) and any pipeline
        # overrun then read out-of-bounds -> hardware returns 0. A zeroed x column makes
        # the padding slot's outer product ``grad (x) 0 == 0``, so the (unbounded) grad
        # operand may safely read garbage for those slots.
        if const_expr(dma_swizzle):
            x_addr_i64 = arith.index_cast(T.i64, ptrtoint(X))
            x_nrec_bytes = nrecv_idx * K_idx * arith.index(2)
            x_rsrc = buffer_ops.create_buffer_resource_from_addr(
                x_addr_i64, num_records_bytes=x_nrec_bytes
            )

        n_block_base = n_tile * block_n
        k_block_base = k_tile * block_k
        n_base_idx = arith.index_cast(T.index, n_block_base)
        k_base_idx = arith.index_cast(T.index, k_block_base)

        # Per-expert routed-slot range. ``base_slot`` is the block-padded slot offset into
        # ``SORTED`` (holds the received-token row for the ``x`` gather); ``route_start_e``
        # is the compact first-route index into the ``[num_routes, N]`` grad buffer.
        bstart = buffer_load_i32(bstart_rsrc, expert)
        nblocks = buffer_load_i32(bpe_rsrc, expert)
        rstart = buffer_load_i32(rstart_rsrc, expert)
        base_slot = arith.index_cast(T.index, bstart) * arith.index(WGRAD_BLOCK_M)
        num_slots = arith.index_cast(T.index, nblocks) * arith.index(WGRAD_BLOCK_M)
        route_start_e_idx = arith.index_cast(T.index, rstart)

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

        def gather_grad(g_ids, slot_base_idx):
            # The grad operand is the compact ``[num_routes, N]`` buffer: the row for a
            # routed slot is ``route_start_e + (slot_base + slot_within_tile)`` -- a plain
            # contiguous walk that needs no ``SORTED`` indirection. ``SORTED`` is still read
            # (``g_ids``) only to get the slot's received-token, whose sentinel marks the
            # block-padding rows that must be masked to zero.
            #
            # ``in_range`` guards the last pipeline step: the gather runs one contraction tile
            # ahead, so on the final iteration ``slot_base == num_slots`` (one past the
            # expert). Those overrun slots read ``SORTED`` past the expert's region -- which,
            # with a tight ``[num_routes, N]`` grad, would compute an out-of-bounds
            # ``grad_row`` -- so we mask them to row 0 exactly like the Triton ``row <
            # num_slots`` mask. (The overrun tile's result is never consumed.)
            raw = []
            for i in range_constexpr(G_FILLS):
                slot_idx, feat_idx = _tile_idx(i, CPR_G)[:2]
                token = arith.index_cast(T.index, g_ids[i])
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult, slot_base_idx + slot_idx, num_slots
                )
                valid = arith.andi(
                    in_range, arith.cmpi(arith.CmpIPredicate.ult, token, nrecv_idx)
                )
                # Row source for the N-operand: default (FC1) is a contiguous route walk into a
                # compact ``[num_routes, N]`` grad; ``swap_gather`` (FC2) instead token-gathers a
                # ``[num_recv, N]`` buffer via ``SORTED`` (so the N feature = the gathered
                # operand's), letting the wgrad emit its natural transpose without a post-pass.
                route_row = route_start_e_idx + slot_base_idx + slot_idx
                grad_row = token if const_expr(swap_gather) else route_row
                goff = valid.select(grad_row, c0) * N_idx + n_base_idx + feat_idx
                gvec = buffer_load_bf16_vec(grad_rsrc, goff, FILL_V)
                raw.append((gvec, valid, None))
            return raw

        def gather_x(x_ids, slot_base_idx):
            # ``SORTED`` holds the received-token row directly (no ``slot // top_k``): the
            # activation gather is ``x[token, k_feat]`` with ``token = SORTED[route_pos]``.
            # ``in_range`` masks the same one-tile pipeline overrun as ``gather_grad``.
            raw = []
            for i in range_constexpr(X_FILLS):
                slot_idx, feat_idx = _tile_idx(i, CPR_X)[:2]
                token = arith.index_cast(T.index, x_ids[i])
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult, slot_base_idx + slot_idx, num_slots
                )
                valid = arith.andi(
                    in_range, arith.cmpi(arith.CmpIPredicate.ult, token, nrecv_idx)
                )
                # Row source for the K-operand: default (FC1) token-gathers a ``[num_recv, K]``
                # buffer via ``SORTED``; ``swap_gather`` (FC2) instead reads a compact
                # ``[num_routes, K]`` buffer by route position (the gather moves to the N-operand).
                route_row = route_start_e_idx + slot_base_idx + slot_idx
                x_row = route_row if const_expr(swap_gather) else token
                xoff = valid.select(x_row, c0) * K_idx + k_base_idx + feat_idx
                xvec = buffer_load_bf16_vec(x_rsrc, xoff, FILL_V)
                raw.append((xvec, valid, None))
            return raw

        def store_grad(raw, buf_elem):
            for i in range_constexpr(G_FILLS):
                gvec, valid, _ = raw[i]
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

        def _dma_one(rsrc, ids, slot_base_idx, cpr, feat_base_idx, dim_idx, lds_off,
                     buf_byte, n_fills, clamp_row):
            # Issue the global->LDS DMA for one operand: each lane streams FILL_V bf16
            # from ``global[row, swizzled_feat]`` straight into contiguous LDS (no VGPR
            # staging). The LDS destination base is wave-uniform (readfirstlane); the
            # hardware spreads lane L to base + L*16B, reconstructing the contiguous
            # ``phys_linear * FILL_V`` layout the swizzled transpose read expects.
            for i in range_constexpr(n_fills):
                phys = tid + fx.Int32(i * n_threads)
                slot = phys // fx.Int32(cpr)
                chunk = phys % fx.Int32(cpr)
                slot_idx = arith.index_cast(T.index, slot)
                token = arith.index_cast(T.index, ids[i])
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult, slot_base_idx + slot_idx, num_slots
                )
                valid = arith.andi(
                    in_range, arith.cmpi(arith.CmpIPredicate.ult, token, nrecv_idx)
                )
                if const_expr(clamp_row):
                    # grad: contiguous route walk; clamp overrun to row 0 for fault safety
                    # (its padding contribution is cancelled by the zeroed x column).
                    row_idx = valid.select(route_start_e_idx + slot_base_idx + slot_idx, c0)
                else:
                    # x: gather by received-token; sentinel/OOB row -> hardware 0.
                    row_idx = token
                swz = (slot & fx.Int32(cpr - 1)) if _DMA_SWZ_ON else fx.Int32(0)
                glob_feat = (chunk ^ swz) * fx.Int32(FILL_V)
                glob_feat_idx = arith.index_cast(T.index, glob_feat)
                voff_elem = row_idx * dim_idx + feat_base_idx + glob_feat_idx
                voff_byte = arith.index_cast(T.i32, voff_elem * arith.index(2))
                lds_perlane = (
                    fx.Int32(lds_off) + buf_byte + phys * fx.Int32(FILL_V * 2)
                )
                lds_base = rocdl.readfirstlane(T.i32, lds_perlane)
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, _gep_lds(smem_raw_ptr, lds_base), fx.Int32(FILL_V * 2),
                    voff_byte, fx.Int32(0), fx.Int32(0), fx.Int32(1),
                )

        def dma_fill(g_ids, x_ids, slot_base_idx, g_buf_byte, x_buf_byte):
            _dma_one(grad_rsrc, g_ids, slot_base_idx, CPR_G_SWZ, n_base_idx, N_idx,
                     g_lds_off, g_buf_byte, G_FILLS, clamp_row=True)
            _dma_one(x_rsrc, x_ids, slot_base_idx, CPR_X_SWZ, k_base_idx, K_idx,
                     x_lds_off, x_buf_byte, X_FILLS, clamp_row=False)

        def _dma_barrier(keep=0):
            # DMA lands on vmcnt (global load); drain it before the workgroup barrier so
            # all waves observe the freshly-staged LDS tile. ``keep`` leaves that many
            # vmem ops in flight (graduated wait) -- for a >2-stage pipeline this keeps the
            # just-issued tile's DMA streaming across the barrier so it overlaps the next
            # iteration's MFMA too (only the tile read next is fully drained). lgkmcnt(0)
            # retires this wave's ds_reads before the buffer is recycled NUM_BUF steps on.
            asm = f"s_waitcnt vmcnt({keep}) lgkmcnt(0)\ns_barrier"
            _llvm.InlineAsmOp(
                res=None, operands_=[], asm_string=asm,
                constraints="", has_side_effects=True, is_align_stack=False,
            )

        def compute(accs, g_buf_byte, x_buf_byte, dma_prefetch=None):
            # A fragments are read one-per-mi to keep VGPR pressure low (hoisting all of
            # them costs occupancy, which the large tile-count shapes depend on). B
            # fragments are loop-invariant across mi, so they are fetched exactly once --
            # but interleaved with the mi==0 MFMAs (see below) rather than as a serial
            # burst up front, so their ds_read latency overlaps compute. The MFMA region
            # runs at raised priority so the matrix pipe stays fed while reads are in
            # flight instead of the scheduler round-robining to a stalled wave.
            def read_a(mi):
                if const_expr(dma_swizzle):
                    return _tr_read_frag_swz(
                        smem_raw_ptr, g_lds_off, SG, CPR_G_SWZ, warp_n_base, mi * WMMA_M,
                        lane_m_base, tr_k_group, tr_col_sub, g_buf_byte,
                    )
                return _tr_read_frag(
                    g_lds_off, SG, warp_n_base, mi * WMMA_M,
                    lane_m_base, tr_k_group, tr_col_sub, g_buf_byte,
                )

            def read_b(nj):
                if const_expr(dma_swizzle):
                    return _tr_read_frag_swz(
                        smem_raw_ptr, x_lds_off, SX, CPR_X_SWZ, warp_k_base, nj * WMMA_N,
                        lane_m_base, tr_k_group, tr_col_sub, x_buf_byte,
                    )
                return _tr_read_frag(
                    x_lds_off, SX, warp_k_base, nj * WMMA_N,
                    lane_m_base, tr_k_group, tr_col_sub, x_buf_byte,
                )

            new_accs = [None] * NACC
            b_frags = [None] * N_STEPS

            if const_expr(dma_prefetch is not None):
                # DMA path: burst *all* current-tile transpose reads first, then issue the
                # next-tile DMA. Keeping the reads ahead of the DMA in program order stops
                # the compiler from planting an ``s_waitcnt vmcnt(0)`` in front of the first
                # ds_read (which would drain the whole prefetch before compute); the DMA
                # then streams under the MFMA burst instead.
                a_frags = [read_a(mi) for mi in range_constexpr(M_STEPS)]
                for nj in range_constexpr(N_STEPS):
                    b_frags[nj] = read_b(nj)
                # Fence the scheduler so the read burst stays *ahead* of the DMA issue --
                # otherwise the compiler hoists the next-tile buffer_load_lds above the
                # transpose reads and re-plants an ``s_waitcnt vmcnt(0)`` that drains the
                # prefetch before compute.
                rocdl.sched_barrier(0)
                dma_prefetch()
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
                for mi in range_constexpr(M_STEPS):
                    for nj in range_constexpr(N_STEPS):
                        idx = mi * N_STEPS + nj
                        new_accs[idx] = rocdl.mfma_f32_16x16x32_bf16(
                            T.vec(C_FRAG, T.f32),
                            [a_frags[mi], b_frags[nj], accs[idx], 0, 0, 0],
                        )
                    rocdl.sched_barrier(0)
                rocdl.s_setprio(0)
                return new_accs

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

        # NUM_BUF-stage ping-pong pipeline: MFMA the current LDS buffer while the next
        # step's global gather is in flight, then stage it into another buffer -- so a
        # single barrier per step suffices (no store->barrier->compute serialization).
        # Prefetch distance D == NUM_BUF - 1: the DMA path stages D tiles up front and, for
        # NUM_BUF > 2, keeps the most-recently-issued tile's DMA in flight across the
        # barrier (graduated ``vmcnt``) so it overlaps two MFMA steps instead of one. Slot
        # ids run D steps ahead of the data they address (loaded here, consumed D iterations
        # later once their vmem load has already retired) via iter_args.
        D = NUM_BUF - 1
        PER_TILE_DMA = G_FILLS + X_FILLS
        DMA_KEEP = (NUM_BUF - 2) * PER_TILE_DMA
        if const_expr(dma_swizzle):
            # Prologue: stage tiles 0..D-1 into buffers 0..D-1, then drain fully.
            for j in range_constexpr(D):
                j_slot = arith.index(j * WMMA_K)
                gidj, xidj = load_slot_ids(j_slot)
                dma_fill(
                    gidj, xidj, j_slot,
                    fx.Int32(j * G_TILE_ELEMS * 2), fx.Int32(j * X_TILE_ELEMS * 2),
                )
            _dma_barrier()
            # Carried ids address the first tile the loop prefetches (tile D).
            g_ids_next, x_ids_next = load_slot_ids(arith.index(D * WMMA_K))
        else:
            g_ids0, x_ids0 = load_slot_ids(c0)
            store_grad(gather_grad(g_ids0, c0), c0)
            store_x(gather_x(x_ids0, c0), c0)
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

            if const_expr(dma_swizzle):
                cur = it % fx.Int32(NUM_BUF)
                cur_g_byte = cur * fx.Int32(G_TILE_ELEMS * 2)
                cur_x_byte = cur * fx.Int32(X_TILE_ELEMS * 2)
                # Prefetch tile it+D into buffer (it+D) % NUM_BUF, D contraction tiles ahead.
                pf = (it + fx.Int32(D)) % fx.Int32(NUM_BUF)
                pf_g_byte = pf * fx.Int32(G_TILE_ELEMS * 2)
                pf_x_byte = pf * fx.Int32(X_TILE_ELEMS * 2)
                pf_slot = s_base + arith.index(D * WMMA_K)
                # Prefetch the slot ids D+1 steps ahead (carried in as the tile-(it+D) ids,
                # yielded here for tile it+D+1); D iterations of lead means their vmem load
                # has retired before ``dma_fill`` consumes them, so it does not force a
                # premature vmcnt that would drain the in-flight prefetch.
                g_ids_nxt, x_ids_nxt = load_slot_ids(s_base + arith.index((D + 1) * WMMA_K))
                # ``compute`` bursts the current-tile reads first, then invokes this
                # callback to issue the next tile's global->LDS DMA, so the DMA streams
                # under the MFMA burst without draining in front of the transpose reads.
                new_accs = compute(
                    accs, cur_g_byte, cur_x_byte,
                    dma_prefetch=lambda: dma_fill(
                        g_ids, x_ids, pf_slot, pf_g_byte, pf_x_byte
                    ),
                )
                # Graduated drain + barrier: keep DMA_KEEP vmem ops (the just-issued tile's
                # DMA for NUM_BUF > 2) in flight so it overlaps the next step's MFMA; the
                # tile read next iteration is fully drained.
                _dma_barrier(DMA_KEEP)
                scf.YieldOp(new_accs + _pack_ids(g_ids_nxt, x_ids_nxt))
            else:
                cur = it % fx.Int32(2)
                nxt = fx.Int32(1) - cur
                cur_g_byte = cur * fx.Int32(G_TILE_ELEMS * 2)
                cur_x_byte = cur * fx.Int32(X_TILE_ELEMS * 2)
                nxt_g_elem = arith.index_cast(T.index, nxt * fx.Int32(G_TILE_ELEMS))
                nxt_x_elem = arith.index_cast(T.index, nxt * fx.Int32(X_TILE_ELEMS))

                # Issue next step's data gather using the slot ids carried in (already
                # resident). The carried ids belong to slot base ``s_base + step`` (one
                # contraction tile ahead), so the compact grad rows walk from that base.
                g_regs_next = gather_grad(g_ids, s_base + step)
                x_regs_next = gather_x(x_ids, s_base + step)
                # Prefetch the slot ids two steps ahead; carried out to the next iteration
                # so their global-load latency overlaps this step's MFMA.
                g_ids_nxt, x_ids_nxt = load_slot_ids(s_base + step + step)

                new_accs = compute(accs, cur_g_byte, cur_x_byte)

                # Pin the mask+store (consumers of the in-flight next-tile loads) *after*
                # the MFMA so their `s_waitcnt vmcnt` isn't hoisted in front of the matrix
                # ops -- the matrix pipe then runs while those loads are still in flight.
                rocdl.sched_barrier(0)
                store_grad(g_regs_next, nxt_g_elem)
                store_x(x_regs_next, nxt_x_elem)
                _barrier()
                scf.YieldOp(new_accs + _pack_ids(g_ids_nxt, x_ids_nxt))

        accs = [loop.results[i] for i in range(NACC)]

        # Epilogue: C[m=n_feat, n=k_feat], lane holds 4 rows. Each dW element is owned by
        # exactly one workgroup (grid = N x K x E over disjoint output tiles), so when
        # ``accumulate`` is set the read-modify-write into the destination is race-free (no
        # atomics needed) -- this lets the caller fold the wgrad straight into ``main_grad`` /
        # ``.grad`` instead of materializing a scratch dW and doing a separate add/copy.
        out_ty = T.f32 if out_is_f32 else bf16
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
                        )  # f32 accumulator
                        out_off = E_NK_row + n_out_idx * K_idx + c_n_idx
                        if const_expr(accumulate):
                            prev = buffer_ops.buffer_load(
                                dW_rsrc, out_off, vec_width=1, dtype=out_ty
                            )
                            prev_f32 = prev if const_expr(out_is_f32) else arith.extf(T.f32, prev)
                            val = arith.addf(val, prev_f32)
                        store_val = val if const_expr(out_is_f32) else arith.truncf(bf16, val)
                        buffer_ops.buffer_store(store_val, dW_rsrc, out_off)
                        scf.YieldOp([])

    @flyc.jit
    def launch_wgrad(
        dW: fx.Pointer,
        X: fx.Pointer,
        GRAD: fx.Pointer,
        SORTED: fx.Pointer,
        BLOCK_START: fx.Pointer,
        BLOCKS_PER_EXPERT: fx.Pointer,
        ROUTE_START: fx.Pointer,
        N: fx.Int32,
        K: fx.Int32,
        num_recv_tokens: fx.Int32,
        num_experts: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        if const_expr(dma_swizzle):
            with ir.InsertionPoint(ctx.gpu_module_body):
                _llvm.GlobalOp(
                    global_type=ir.Type.parse(f"!llvm.array<{LDS_TOTAL_BYTES} x i8>"),
                    sym_name=LDS_SYM,
                    linkage=ir.Attribute.parse("#llvm.linkage<external>"),
                    addr_space=3,
                    alignment=1024,
                )
        else:
            allocator.finalized = False
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
        gx = (N + block_n - 1) // block_n
        gy = (K + block_k - 1) // block_k
        gz = num_experts
        wgrad_kernel._func.__name__ = KERNEL_NAME
        wgrad_kernel(
            dW, X, GRAD, SORTED, BLOCK_START, BLOCKS_PER_EXPERT, ROUTE_START,
            N, K, num_recv_tokens,
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


def _gep_lds(base_ptr, byte_i32):
    """GEP an LDS !llvm.ptr<3> by a runtime i8 byte offset off a real LDS base pointer.

    ``buffer_load_lds`` derives its M0 write base from the LDS pointer, which the backend
    only lowers correctly when the pointer is a GEP off a genuine addrspace(3) global (an
    ``inttoptr`` address is fine for ``ds_read`` but not for the DMA). We route the reads
    through the same base so alias analysis ties the DMA writes to the transpose reads.
    """
    return _llvm.getelementptr(
        ir.Type.parse("!llvm.ptr<3>"), rocdl._to_ir(base_ptr), [rocdl._to_ir(byte_i32)],
        [-(2 ** 31)], T.i8, None,
    )


def _tr_read_frag_swz(
    smem_base, lds_off, stride, cpr, warp_col_base, col_const,
    lane_m_base, tr_k_group, tr_col_sub, buf_byte,
):
    """Swizzled counterpart of ``_tr_read_frag`` for the DMA fill (un-padded LDS).

    The DMA writes each contraction tile contiguously (stride == feature span), so bank
    conflicts on the transpose read are broken by an XOR *chunk* swizzle instead of row
    padding: the physical feature chunk of logical ``(slot, feat)`` is
    ``(feat // FILL_V) XOR (slot & (cpr - 1))`` -- the *same* map the fill applies to the
    global gather column, so the read lands on exactly the element the DMA staged.

    ``ds_read_tr16_b64`` gathers per-lane (each lane reads its own 4 contiguous features
    and the transpose is a fixed cross-lane shuffle), so a per-lane swizzled address is
    correct with no change to the transpose semantics. Because the XOR mixes the feature
    bits, the per-fragment column shift and the lo->hi (4-slot) shift can no longer ride
    the ``ds_read`` immediate ``offset:`` -- both addresses are computed explicitly.
    """
    col_run = warp_col_base + tr_col_sub * fx.Int32(4)
    feat_log = col_run + fx.Int32(col_const)          # logical feature column
    chunk = feat_log // fx.Int32(FILL_V)
    within = feat_log % fx.Int32(FILL_V)
    row_lo = lane_m_base * fx.Int32(8) + tr_k_group   # contraction slot (0..31)

    def _read(slot):
        swz = (slot & fx.Int32(cpr - 1)) if _DMA_SWZ_ON else fx.Int32(0)
        phys_chunk = chunk ^ swz
        phys_feat = phys_chunk * fx.Int32(FILL_V) + within
        elem = slot * fx.Int32(stride) + phys_feat
        byte = elem * fx.Int32(2) + fx.Int32(lds_off) + buf_byte
        raw = rocdl.ds_read_tr16_b64(T.vec(4, T.bf16), _gep_lds(smem_base, byte)).result
        return fx.Vector(raw, (4,), fx.BFloat16)

    lo = _read(row_lo)
    hi = _read(row_lo + fx.Int32(4))
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


def buffer_load_i32(rsrc, off_i32):
    return buffer_ops.buffer_load(rsrc, off_i32, vec_width=1, dtype=T.i32)


def buffer_load_i32_idx(rsrc, off_idx):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.i32)


def buffer_load_bf16_vec(rsrc, off_idx, v):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=v, dtype=T.bf16)


def buffer_store_bf16(rsrc, off_idx, val):
    buffer_ops.buffer_store(val, rsrc, off_idx)

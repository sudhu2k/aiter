# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd


@triton.jit
def _wgrad_accumulate_tile(
    x_ptr,
    grad_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    base_slot,
    num_slots,
    pid_n,
    pid_k,
    N,
    K,
    num_valid_tokens,
    stride_xm,
    stride_xk,
    stride_gm,
    stride_gn,
    top_k,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CONTRACT_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    TRANS_FREE: tl.constexpr,
):
    """Accumulate ``dW[n-tile, k-tile] = sum_m trans(grad)[n, m] @ x[m, k]``.

    Walks the expert's contiguous routed-slot range ``[base_slot, base_slot + num_slots)``
    in ``CONTRACT_M``-sized steps -- the contraction (MFMA inner) dimension -- decoupled
    from the 32-wide align block so each ``tl.dot`` has a large inner dim. Both operands are
    gathered through ``sorted_token_ids``. Returns the fp32 ``[BLOCK_N, BLOCK_K]`` tile.

    When ``TRANS_FREE`` the ``grad`` operand is gathered directly in ``[BLOCK_N, CONTRACT_M]``
    layout so ``tl.dot(g, x)`` contracts over the token axis with no ``tl.trans`` -- avoiding
    the LDS transpose round-trip (and its barriers) at the cost of a column-gathered load.
    """
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_m = tl.arange(0, CONTRACT_M)

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for m0 in range(0, num_slots, CONTRACT_M):
        row = m0 + offs_m
        in_range = row < num_slots
        slots = tl.load(
            sorted_token_ids_ptr + base_slot + row,
            mask=in_range,
            other=num_valid_tokens,
        ).to(tl.int64)
        token_mask = in_range & (slots < num_valid_tokens)
        tokens = slots // top_k

        x_ptrs = x_ptr + tokens[:, None] * stride_xm + offs_k[None, :] * stride_xk
        if EVEN_K:
            x = tl.load(x_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            x = tl.load(
                x_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K),
                other=0.0,
            )

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + slots, mask=token_mask, other=0.0)

        if TRANS_FREE:
            # Gather grad already transposed: rows = N, cols = contraction (token slots).
            g_ptrs = grad_ptr + offs_n[:, None] * stride_gn + slots[None, :] * stride_gm
            if EVEN_N:
                g = tl.load(g_ptrs, mask=token_mask[None, :], other=0.0)
            else:
                g = tl.load(
                    g_ptrs,
                    mask=token_mask[None, :] & (offs_n[:, None] < N),
                    other=0.0,
                )
            if MUL_ROUTED_WEIGHT:
                g = (g.to(tl.float32) * moe_weight[None, :]).to(x.dtype)
            accumulator += tl.dot(g, x)
        else:
            g_ptrs = grad_ptr + slots[:, None] * stride_gm + offs_n[None, :] * stride_gn
            if EVEN_N:
                g = tl.load(g_ptrs, mask=token_mask[:, None], other=0.0)
            else:
                g = tl.load(
                    g_ptrs,
                    mask=token_mask[:, None] & (offs_n[None, :] < N),
                    other=0.0,
                )
            if MUL_ROUTED_WEIGHT:
                g = (g.to(tl.float32) * moe_weight[:, None]).to(x.dtype)
            accumulator += tl.dot(tl.trans(g), x)

    return accumulator


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    }
)
@triton.jit
def _fused_moe_wgrad_kernel(
    # Pointers to matrices
    x_ptr,             # activations, [num_tokens, K]
    grad_ptr,          # upstream grad rows, [num_tokens * top_k, N]
    dw_ptr,            # output weight grad, [num_experts, N, K]
    topk_weights_ptr,  # [num_tokens, top_k] flattened; indexed by routed slot
    sorted_token_ids_ptr,   # [padded] routed-slot ids grouped by expert
    block_start_ptr,        # [num_experts] first sorted block index per expert
    blocks_per_expert_ptr,  # [num_experts] number of sorted blocks per expert
    # Matrix dimensions
    N,
    K,
    num_valid_tokens,
    # Strides
    stride_xm,
    stride_xk,
    stride_gm,
    stride_gn,
    stride_we,
    stride_wn,
    stride_wk,
    top_k,
    # Meta-parameters
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CONTRACT_M: tl.constexpr,
    TRANS_FREE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Weight-gradient for a permute-free MoE grouped GEMM.

    Computes, per expert ``e``::

        dW[e][n, k] = sum_{(t, j): topk_ids[t, j] == e} grad[t, j, n] * X[t, k]

    The reduction runs over the routed token-slots (the *contraction* axis), gathered
    through ``sorted_token_ids`` -- the same expert-grouped, block-padded buffer built by
    ``moe_align_block_size`` for the forward gather-GEMM. Each program owns one
    ``(expert, N-tile, K-tile)`` output tile and walks that expert's slot range in
    ``CONTRACT_M`` steps, accumulating in fp32 registers and writing the tile exactly once
    (deterministic, no atomics). ``BLOCK_SIZE_M`` is the align block size (used to locate
    the expert's contiguous slot range); ``CONTRACT_M`` is the MFMA contraction tile.
    """
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_per_expert = num_pid_n * num_pid_k

    expert = pid // num_pid_per_expert
    pid_in_e = pid % num_pid_per_expert
    pid_n = pid_in_e // num_pid_k
    pid_k = pid_in_e % num_pid_k

    nblocks = tl.load(blocks_per_expert_ptr + expert)
    bstart = tl.load(block_start_ptr + expert)
    base_slot = bstart * BLOCK_SIZE_M
    num_slots = nblocks * BLOCK_SIZE_M

    accumulator = _wgrad_accumulate_tile(
        x_ptr, grad_ptr, topk_weights_ptr, sorted_token_ids_ptr,
        base_slot, num_slots, pid_n, pid_k,
        N, K, num_valid_tokens,
        stride_xm, stride_xk, stride_gm, stride_gn, top_k,
        MUL_ROUTED_WEIGHT, BLOCK_SIZE_N, BLOCK_SIZE_K, CONTRACT_M, EVEN_K, EVEN_N, TRANS_FREE,
    )

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    dw = accumulator.to(compute_type)
    dw_ptrs = (
        dw_ptr
        + expert * stride_we
        + offs_n[:, None] * stride_wn
        + offs_k[None, :] * stride_wk
    )
    dw_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    tl.store(dw_ptrs, dw, mask=dw_mask)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    }
)
@triton.jit
def _fused_moe_wgrad_persistent_kernel(
    # Pointers to matrices
    x_ptr,             # activations, [num_tokens, K]
    grad_ptr,          # upstream grad rows, [num_tokens * top_k, N]
    dw_ptr,            # output weight grad, [num_experts, N, K]
    topk_weights_ptr,  # [num_tokens, top_k] flattened; indexed by routed slot
    sorted_token_ids_ptr,   # [padded] routed-slot ids grouped by expert
    block_start_ptr,        # [num_experts] first sorted block index per expert
    blocks_per_expert_ptr,  # [num_experts] number of sorted blocks per expert
    # Matrix dimensions
    num_experts,
    N,
    K,
    num_valid_tokens,
    # Strides
    stride_xm,
    stride_xk,
    stride_gm,
    stride_gn,
    stride_we,
    stride_wn,
    stride_wk,
    top_k,
    # Meta-parameters
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CONTRACT_M: tl.constexpr,
    TRANS_FREE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Persistent variant of :func:`_fused_moe_wgrad_kernel`.

    Launches ``min(NUM_SMS, num_tiles)`` programs that grid-stride over the flattened
    ``(expert, N-tile, K-tile)`` output-tile space. Tiles are XCD-remapped and grouped via
    ``pid_grid`` (over the ``N x K`` plane, per expert) to promote L2 reuse of the gathered
    ``x``/``grad`` tiles. Each visited tile runs the same deterministic per-expert
    contraction as the non-persistent kernel and writes its output tile exactly once.
    """
    start_pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    tiles_per_expert = num_pid_n * num_pid_k
    num_tiles = num_experts * tiles_per_expert

    num_iters = tl.cdiv(num_tiles - start_pid, NUM_SMS)
    tile_id = start_pid
    for _ in range(0, num_iters):
        remapped = remap_xcd(tile_id, num_tiles, NUM_XCDS)
        expert = remapped // tiles_per_expert
        pid_in_e = remapped % tiles_per_expert
        # Reuse the 2D grouped mapping over the (N, K) plane for L2 locality.
        pid_n, pid_k = pid_grid(pid_in_e, num_pid_n, num_pid_k, GROUP_SIZE_M)

        nblocks = tl.load(blocks_per_expert_ptr + expert)
        bstart = tl.load(block_start_ptr + expert)
        base_slot = bstart * BLOCK_SIZE_M
        num_slots = nblocks * BLOCK_SIZE_M

        accumulator = _wgrad_accumulate_tile(
            x_ptr, grad_ptr, topk_weights_ptr, sorted_token_ids_ptr,
            base_slot, num_slots, pid_n, pid_k,
            N, K, num_valid_tokens,
            stride_xm, stride_xk, stride_gm, stride_gn, top_k,
            MUL_ROUTED_WEIGHT, BLOCK_SIZE_N, BLOCK_SIZE_K, CONTRACT_M, EVEN_K, EVEN_N, TRANS_FREE,
        )

        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        dw = accumulator.to(compute_type)
        dw_ptrs = (
            dw_ptr
            + expert * stride_we
            + offs_n[:, None] * stride_wn
            + offs_k[None, :] * stride_wk
        )
        dw_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.store(dw_ptrs, dw, mask=dw_mask)

        tile_id += NUM_SMS

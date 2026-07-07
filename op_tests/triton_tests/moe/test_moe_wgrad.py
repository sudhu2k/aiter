# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
import torch
import pytest

from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.moe.moe_wgrad import (
    fused_moe_wgrad,
    moe_wgrad_block_offsets,
    get_default_moe_wgrad_config,
    moe_wgrad_set_use_persistent_kernel,
)


@pytest.fixture(params=[False, True], ids=["default", "persistent"])
def persistent_kernel(request):
    """Run each wgrad test against both the default and persistent launches."""
    moe_wgrad_set_use_persistent_kernel(request.param)
    yield request.param
    moe_wgrad_set_use_persistent_kernel(False)


def _align(topk_ids, block_size, num_experts):
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    moe_align_block_size_triton(
        topk_ids.to(torch.int32),
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def _torch_wgrad(x, grad, topk_ids, num_experts, N, K, *, mul_routed_weight, topk_weights):
    """Reference dW[e] = sum_{(t,j): topk_ids[t,j]==e} grad[t,j]^T outer x[t]."""
    M, top_k = topk_ids.shape
    x32 = x.float()
    g32 = grad.float().reshape(M, top_k, N)
    if mul_routed_weight:
        g32 = g32 * topk_weights.float().unsqueeze(-1)
    dw = torch.zeros((num_experts, N, K), dtype=torch.float32, device=x.device)
    flat_e = topk_ids.reshape(-1)
    for slot in range(M * top_k):
        e = int(flat_e[slot].item())
        t = slot // top_k
        j = slot % top_k
        dw[e] += torch.outer(g32[t, j], x32[t])
    return dw


def _make_routing(M, E, top_k, seed=0, force_empty_expert=None):
    torch.manual_seed(seed)
    logits = torch.randn(M, E, device="cuda")
    topk_w, topk_ids = torch.topk(torch.softmax(logits, dim=1), top_k, dim=1)
    topk_ids = topk_ids.to(torch.int32)
    if force_empty_expert is not None:
        # Reroute any token hitting the target expert to a neighbor, leaving it empty.
        repl = (force_empty_expert + 1) % E
        topk_ids = torch.where(
            topk_ids == force_empty_expert,
            torch.full_like(topk_ids, repl),
            topk_ids,
        )
    return topk_ids, topk_w.float()


@pytest.mark.parametrize("M", [64, 512])
@pytest.mark.parametrize("E", [4, 8])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("N,K", [(128, 96), (256, 128)])
@pytest.mark.parametrize("mul_routed_weight", [False, True])
def test_fused_moe_wgrad(M, E, top_k, N, K, mul_routed_weight, persistent_kernel):
    if top_k > E:
        pytest.skip("top_k must be <= num experts")
    block_size = get_default_moe_wgrad_config()["BLOCK_SIZE_M"]
    dtype = torch.bfloat16

    topk_ids, topk_weights = _make_routing(M, E, top_k, seed=1)
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    grad = torch.randn(M * top_k, N, device="cuda", dtype=dtype)

    sorted_ids, _expert_ids, _npp = _align(topk_ids, block_size, E)
    block_start, blocks_per_expert = moe_wgrad_block_offsets(topk_ids, E, block_size)

    dw = torch.empty((E, N, K), device="cuda", dtype=dtype)
    fused_moe_wgrad(
        x,
        grad,
        dw,
        topk_weights,
        topk_ids,
        sorted_ids,
        block_start,
        blocks_per_expert,
        top_k,
        mul_routed_weight,
        tl.bfloat16,
    )

    ref = _torch_wgrad(
        x, grad, topk_ids, E, N, K,
        mul_routed_weight=mul_routed_weight, topk_weights=topk_weights,
    )
    torch.testing.assert_close(dw.float(), ref, atol=2e-1, rtol=2e-2)


def test_fused_moe_wgrad_empty_expert(persistent_kernel):
    """An expert with zero routed tokens must produce an all-zero dW plane."""
    M, E, top_k, N, K = 256, 8, 2, 128, 96
    block_size = get_default_moe_wgrad_config()["BLOCK_SIZE_M"]
    dtype = torch.bfloat16
    empty = 3

    topk_ids, topk_weights = _make_routing(M, E, top_k, seed=2, force_empty_expert=empty)
    assert (topk_ids == empty).sum().item() == 0

    x = torch.randn(M, K, device="cuda", dtype=dtype)
    grad = torch.randn(M * top_k, N, device="cuda", dtype=dtype)
    sorted_ids, _e, _n = _align(topk_ids, block_size, E)
    block_start, blocks_per_expert = moe_wgrad_block_offsets(topk_ids, E, block_size)
    assert int(blocks_per_expert[empty].item()) == 0

    dw = torch.empty((E, N, K), device="cuda", dtype=dtype)
    fused_moe_wgrad(
        x, grad, dw, topk_weights, topk_ids, sorted_ids,
        block_start, blocks_per_expert, top_k, False, tl.bfloat16,
    )
    assert torch.count_nonzero(dw[empty]) == 0

    ref = _torch_wgrad(
        x, grad, topk_ids, E, N, K, mul_routed_weight=False, topk_weights=topk_weights
    )
    torch.testing.assert_close(dw.float(), ref, atol=2e-1, rtol=2e-2)

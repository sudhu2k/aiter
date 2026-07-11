# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for the FlyDSL v2 (LDS ds_read_tr) MoE wgrad kernel."""

import sys

import torch
import triton

from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.flydsl.kernels.moe_wgrad_flydsl_v2 import compile_moe_wgrad_v2, WGRAD_BLOCK_M
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg, _run_compiled
from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.moe.moe_wgrad import moe_wgrad_block_offsets


def _align(topk_ids, block_size, num_experts):
    max_pad = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.full((max_pad,), topk_ids.numel(), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty((triton.cdiv(max_pad, block_size),), dtype=torch.int32, device="cuda")
    npp = torch.empty((1,), dtype=torch.int32, device="cuda")
    moe_align_block_size_triton(topk_ids.to(torch.int32), num_experts, block_size, sorted_ids, expert_ids, npp)
    return sorted_ids


def _torch_wgrad(x, grad, topk_ids, E, N, K, *, mul_routed_weight, topk_weights):
    M, top_k = topk_ids.shape
    x32 = x.float()
    g32 = grad.float().reshape(M, top_k, N)
    if mul_routed_weight:
        g32 = g32 * topk_weights.float().unsqueeze(-1)
    dw = torch.zeros((E, N, K), dtype=torch.float32, device=x.device)
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
        repl = (force_empty_expert + 1) % E
        topk_ids = torch.where(topk_ids == force_empty_expert, torch.full_like(topk_ids, repl), topk_ids)
    return topk_ids, topk_w.float()


def run_case(M, E, top_k, N, K, mul_routed_weight, block_n=64, block_k=64,
             warps_n=1, warps_k=1, seed=1, force_empty_expert=None):
    dtype = torch.bfloat16
    topk_ids, topk_weights = _make_routing(M, E, top_k, seed=seed, force_empty_expert=force_empty_expert)
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    grad = torch.randn(M * top_k, N, device="cuda", dtype=dtype)

    sorted_ids = _align(topk_ids, WGRAD_BLOCK_M, E)
    block_start, blocks_per_expert = moe_wgrad_block_offsets(topk_ids, E, WGRAD_BLOCK_M)

    dw = torch.zeros((E, N, K), device="cuda", dtype=dtype)
    topk_w = topk_weights.reshape(-1).to(torch.float32).contiguous()

    exe = compile_moe_wgrad_v2(
        dtype="bf16", mul_routed_weight=bool(mul_routed_weight),
        block_n=block_n, block_k=block_k, warps_n=warps_n, warps_k=warps_k,
        topk=int(top_k),
    )
    _run_compiled(
        exe, ptr_arg(dw), ptr_arg(x), ptr_arg(grad), ptr_arg(topk_w),
        ptr_arg(sorted_ids), ptr_arg(block_start), ptr_arg(blocks_per_expert),
        int(N), int(K), int(topk_ids.numel()), int(top_k), int(E),
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    ref = _torch_wgrad(x, grad, topk_ids, E, N, K, mul_routed_weight=mul_routed_weight, topk_weights=topk_weights)
    max_abs = (dw.float() - ref).abs().max().item()
    rel = max_abs / (ref.abs().max().item() + 1e-6)
    ok = torch.allclose(dw.float(), ref, atol=2e-1, rtol=2e-2)
    tag = (f"M={M} E={E} top_k={top_k} N={N} K={K} mrw={int(mul_routed_weight)} "
           f"bn={block_n} bk={block_k} w={warps_n}x{warps_k}"
           f"{' empty=' + str(force_empty_expert) if force_empty_expert is not None else ''}")
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}  max_abs={max_abs:.4f} rel={rel:.4f}")
    return ok


def main():
    if not is_flydsl_available():
        print("flydsl not available; skipping")
        return 0
    ok = True
    # Start with feature dims that are multiples of block sizes (no OOB) to isolate the transpose.
    ok &= run_case(64, 4, 2, 64, 64, False)
    ok &= run_case(64, 4, 2, 64, 64, True)
    ok &= run_case(512, 8, 4, 128, 128, False)
    ok &= run_case(512, 8, 4, 128, 128, True)
    ok &= run_case(256, 8, 2, 64, 64, False, seed=2, force_empty_expert=3)
    # OOB feature dims (N,K not multiples of the block) exercise the store guards + buffer-load clamping.
    ok &= run_case(128, 4, 2, 128, 96, False)
    ok &= run_case(128, 4, 2, 128, 96, True)
    ok &= run_case(512, 8, 4, 256, 128, True)
    # v2.1 multi-warp workgroups.
    ok &= run_case(512, 8, 4, 128, 128, False, block_n=128, block_k=128, warps_n=2, warps_k=2)
    ok &= run_case(512, 8, 4, 128, 128, True, block_n=128, block_k=128, warps_n=2, warps_k=2)
    ok &= run_case(512, 8, 4, 256, 128, False, block_n=128, block_k=128, warps_n=2, warps_k=2)
    ok &= run_case(512, 8, 4, 256, 128, True, block_n=128, block_k=64, warps_n=2, warps_k=1)
    ok &= run_case(512, 8, 4, 256, 256, False, block_n=256, block_k=128, warps_n=4, warps_k=2)
    ok &= run_case(256, 8, 2, 128, 96, False, block_n=128, block_k=128, warps_n=2, warps_k=2,
                   seed=2, force_empty_expert=3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

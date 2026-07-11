# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Head-to-head: FlyDSL v2 (LDS ds_read_tr) vs v1 (direct gather) vs Triton transfree."""

import sys
import torch
import triton
import triton.language as tl

from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.flydsl.moe_wgrad import flydsl_moe_wgrad
from aiter.ops.flydsl.kernels.moe_wgrad_flydsl_v2 import compile_moe_wgrad_v2, WGRAD_BLOCK_M
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg, _run_compiled
from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.moe.moe_wgrad import (
    fused_moe_wgrad, moe_wgrad_block_offsets, get_default_moe_wgrad_config,
)


def _align(topk_ids, block_size, E):
    max_pad = topk_ids.numel() + E * (block_size - 1)
    sorted_ids = torch.full((max_pad,), topk_ids.numel(), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty((triton.cdiv(max_pad, block_size),), dtype=torch.int32, device="cuda")
    npp = torch.empty((1,), dtype=torch.int32, device="cuda")
    moe_align_block_size_triton(topk_ids.to(torch.int32), E, block_size, sorted_ids, expert_ids, npp)
    return sorted_ids


def _routing(M, E, top_k, seed=1):
    torch.manual_seed(seed)
    logits = torch.randn(M, E, device="cuda")
    topk_w, topk_ids = torch.topk(torch.softmax(logits, dim=1), top_k, dim=1)
    return topk_ids.to(torch.int32), topk_w.float()


def _bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run(M, E, top_k, N, K, mrw=False):
    dtype = torch.bfloat16
    topk_ids, topk_w = _routing(M, E, top_k)
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    grad = torch.randn(M * top_k, N, device="cuda", dtype=dtype)
    sorted_ids = _align(topk_ids, WGRAD_BLOCK_M, E)
    bstart, bpe = moe_wgrad_block_offsets(topk_ids, E, WGRAD_BLOCK_M)
    dw = torch.zeros((E, N, K), device="cuda", dtype=dtype)
    topk_w_flat = topk_w.reshape(-1).contiguous()
    flops = 2.0 * (M * top_k) * N * K
    print(f"M={M} E={E} tk={top_k} N={N} K={K} mrw={int(mrw)}")

    # Triton transfree (default config, ~ceiling)
    tri_cfg = get_default_moe_wgrad_config()
    def tri():
        fused_moe_wgrad(x, grad, dw, topk_w_flat, topk_ids, sorted_ids, bstart, bpe,
                        top_k, mrw, tl.bfloat16, config=tri_cfg)
    t = _bench(tri)
    print(f"    triton transfree 128x128 : {t:8.3f} ms  ({flops/t/1e9:6.1f} TF/s)")

    # FlyDSL v1 (direct gather)
    for bn, bk in [(64, 64), (128, 128)]:
        def v1(bn=bn, bk=bk):
            flydsl_moe_wgrad(x, grad, dw, topk_w, topk_ids, sorted_ids, bstart, bpe,
                             top_k, mrw, block_n=bn, block_k=bk)
        try:
            t = _bench(v1)
            print(f"    flydsl v1 {bn}x{bk:<3}          : {t:8.3f} ms  ({flops/t/1e9:6.1f} TF/s)")
        except Exception as ex:
            print(f"    flydsl v1 {bn}x{bk}: FAILED {type(ex).__name__}: {str(ex)[:70]}")

    # FlyDSL v2 single-warp (LDS ds_read_tr)
    for bn, bk in [(64, 64), (128, 64)]:
        _bench_v2(dw, x, grad, topk_w_flat, sorted_ids, bstart, bpe, N, K, topk_ids, top_k, E,
                  mrw, bn, bk, 1, 1, flops)
    # FlyDSL v2.1 multi-warp workgroups
    for bn, bk, wn, wk in [
        (128, 128, 2, 2), (128, 128, 4, 2), (256, 128, 4, 2),
        (256, 256, 4, 4), (128, 256, 2, 4), (256, 128, 2, 2),
    ]:
        _bench_v2(dw, x, grad, topk_w_flat, sorted_ids, bstart, bpe, N, K, topk_ids, top_k, E,
                  mrw, bn, bk, wn, wk, flops)


def _bench_v2(dw, x, grad, topk_w_flat, sorted_ids, bstart, bpe, N, K, topk_ids, top_k, E,
              mrw, bn, bk, wn, wk, flops):
    try:
        exe = compile_moe_wgrad_v2(dtype="bf16", mul_routed_weight=bool(mrw),
                                   block_n=bn, block_k=bk, warps_n=wn, warps_k=wk,
                                   topk=int(top_k))
        def v2(exe=exe):
            _run_compiled(exe, ptr_arg(dw), ptr_arg(x), ptr_arg(grad), ptr_arg(topk_w_flat),
                          ptr_arg(sorted_ids), ptr_arg(bstart), ptr_arg(bpe),
                          int(N), int(K), int(topk_ids.numel()), int(top_k), int(E),
                          torch.cuda.current_stream())
        t = _bench(v2)
        print(f"    flydsl v2 {bn}x{bk} w{wn}x{wk:<2}     : {t:8.3f} ms  ({flops/t/1e9:6.1f} TF/s)")
    except Exception as ex:
        print(f"    flydsl v2 {bn}x{bk} w{wn}x{wk}: FAILED {type(ex).__name__}: {str(ex)[:70]}")


def main():
    if not is_flydsl_available():
        print("flydsl not available; skipping")
        return 0
    run(4096, 8, 8, 2048, 512)
    run(2048, 8, 8, 2048, 512)
    run(4096, 8, 8, 512, 2048)
    return 0


if __name__ == "__main__":
    sys.exit(main())

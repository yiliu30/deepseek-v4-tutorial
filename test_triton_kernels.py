#!/usr/bin/env python3
"""
Unit tests comparing Triton kernel reimplementations against tilelang reference.

Tests all 6 kernels:
  1. act_quant — FP8 block-wise quantization
  2. fp4_act_quant — FP4 block-wise quantization (inplace)
  3. fp8_gemm — FP8 GEMM with per-block scaling
  4. fp4_gemm — FP8 act × FP4 weight GEMM
  5. sparse_attn — Sparse attention
  6. hc_split_sinkhorn — Hyper-Connection split + Sinkhorn
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(42)

# Import both implementations
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = os.path.dirname(os.path.abspath(__file__))
tl_mod = load_module("_tilelang", os.path.join(base, "tilelang_kernel.py"))
tr_mod = load_module("_triton", os.path.join(base, "triton_kernel.py"))


def check_close(name, a, b, atol=1e-2, rtol=1e-2):
    """Check tensors are close, print stats on failure."""
    a_f, b_f = a.float(), b.float()
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    if max_diff <= atol and torch.allclose(a_f, b_f, atol=atol, rtol=rtol):
        print(f"  ✓ {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        return True
    else:
        print(f"  ✗ {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        # Show some mismatches
        mask = diff > atol
        n_bad = mask.sum().item()
        print(f"    {n_bad}/{a.numel()} elements exceed atol={atol}")
        if n_bad > 0 and n_bad < 20:
            idxs = mask.nonzero()[:10]
            for idx in idxs:
                idx_tuple = tuple(idx.tolist())
                print(f"    [{idx_tuple}]: ref={a_f[idx_tuple].item():.6f}, test={b_f[idx_tuple].item():.6f}")
        return False


def test_act_quant():
    print("\n=== test_act_quant ===")
    passed = True

    # Test non-inplace (returns fp8 + scale)
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    # Only test combos that make sense:
    # scale_fmt=None + scale_dtype=float32 (standard)
    # scale_fmt="ue8m0" + scale_dtype=float8_e8m0fnu (MXFP style)
    for block_size in [64, 128]:
        for scale_fmt, scale_dtype in [(None, torch.float32), ("ue8m0", torch.float8_e8m0fnu)]:
            label = f"bs={block_size}, fmt={scale_fmt}, sdtype={scale_dtype}"
            x_ref = x.clone()
            x_test = x.clone()
            ref_y, ref_s = tl_mod.act_quant(x_ref, block_size, scale_fmt, scale_dtype, False)
            test_y, test_s = tr_mod.act_quant(x_test, block_size, scale_fmt, scale_dtype, False)
            # Compare dequantized values (FP8 → float * scale)
            ref_deq = ref_y.float().view(-1, block_size) * ref_s.float().view(-1, 1)
            test_deq = test_y.float().view(-1, block_size) * test_s.float().view(-1, 1)
            passed &= check_close(f"non-inplace {label}", ref_deq, test_deq, atol=0.5, rtol=0.05)

    # Test inplace
    x_ref = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    x_test = x_ref.clone()
    tl_mod.act_quant(x_ref, 64, "ue8m0", torch.float8_e8m0fnu, True)
    tr_mod.act_quant(x_test, 64, "ue8m0", torch.float8_e8m0fnu, True)
    passed &= check_close("inplace", x_ref, x_test, atol=0.2, rtol=0.1)

    return passed


def test_fp4_act_quant():
    print("\n=== test_fp4_act_quant ===")
    passed = True

    # Test inplace (main use case)
    x_ref = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    x_test = x_ref.clone()
    tl_mod.fp4_act_quant(x_ref, 32, True)
    tr_mod.fp4_act_quant(x_test, 32, True)
    passed &= check_close("inplace bs=32", x_ref, x_test, atol=1.5, rtol=0.2)

    return passed


def test_fp8_gemm():
    print("\n=== test_fp8_gemm ===")
    passed = True

    M, N, K = 32, 256, 512
    group_size = 128

    # Create random BF16 matrices and quantize
    a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # Quantize A [M, K] → FP8 with per-row per-128 scale → a_s: [M, K/128]
    a_fp8, a_s = tl_mod.act_quant(a_bf16, group_size, None, torch.float32, False)

    # Quantize B [N, K] → FP8 with scale
    # B scale layout for fp8_gemm: [ceil(N/128), ceil(K/128)]
    # We need to quantize B row-by-row, then reshape scales
    b_fp8, b_s_raw = tl_mod.act_quant(b_bf16, group_size, None, torch.float32, False)
    # b_s_raw is [N, K/128], but fp8_gemm expects [N/128, K/128] (one scale per N-group)
    # Take mean scale per N-group as approximation
    b_s = b_s_raw.view(N // group_size, group_size, K // group_size).mean(dim=1)

    ref = tl_mod.fp8_gemm(a_fp8, a_s, b_fp8, b_s, torch.float32)
    test = tr_mod.fp8_gemm(a_fp8, a_s, b_fp8, b_s, torch.float32)

    passed &= check_close("fp8_gemm M=32,N=256,K=512", ref, test, atol=2.0, rtol=0.1)
    return passed


def test_fp4_gemm():
    print("\n=== test_fp4_gemm ===")
    passed = True

    M, N, K = 32, 256, 128
    act_group = 128
    weight_group = 32

    # Create FP8 activation with scale
    a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    a_fp8, a_s = tl_mod.act_quant(a_bf16, act_group, None, torch.float32, False)

    # Create FP4 weight with scale — need float32 scales for tilelang fp4_gemm
    b_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    b_fp4, b_s_e8m0 = tl_mod.fp4_act_quant(b_bf16, weight_group, False)
    b_s = b_s_e8m0.float()  # tilelang fp4_gemm expects float32 scales

    ref = tl_mod.fp4_gemm(a_fp8, a_s, b_fp4, b_s, torch.float32)
    test = tr_mod.fp4_gemm(a_fp8, a_s, b_fp4, b_s, torch.float32)

    passed &= check_close("fp4_gemm M=32,N=256,K=128", ref, test, atol=2.0, rtol=0.1)
    return passed


def test_sparse_attn():
    print("\n=== test_sparse_attn ===")
    passed = True

    B, S, H, D = 2, 8, 4, 64
    T_kv = 32
    K_topk = 16
    scale = D ** -0.5

    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, T_kv, D, device="cuda", dtype=torch.bfloat16)
    attn_sink = torch.randn(H, device="cuda", dtype=torch.float32)
    # Random indices in [0, T_kv), some -1 for masking
    topk_idxs = torch.randint(0, T_kv, (B, S, K_topk), device="cuda", dtype=torch.int32)
    topk_idxs[:, :, -2:] = -1  # mask last 2

    # Reference: PyTorch fallback from kernel.py
    from kernel import sparse_attn as pytorch_sparse_attn
    ref = pytorch_sparse_attn(q, kv, attn_sink, topk_idxs, scale)
    test = tr_mod.sparse_attn(q, kv, attn_sink, topk_idxs, scale)

    passed &= check_close("sparse_attn B=2,S=8,H=4,D=64,K=16", ref, test, atol=0.05, rtol=0.02)

    # Larger test matching real model dims
    B2, S2, H2, D2 = 1, 4, 16, 512
    T2, K2 = 128, 64
    q2 = torch.randn(B2, S2, H2, D2, device="cuda", dtype=torch.bfloat16)
    kv2 = torch.randn(B2, T2, D2, device="cuda", dtype=torch.bfloat16)
    sink2 = torch.randn(H2, device="cuda", dtype=torch.float32)
    idx2 = torch.randint(0, T2, (B2, S2, K2), device="cuda", dtype=torch.int32)

    ref2 = pytorch_sparse_attn(q2, kv2, sink2, idx2, D2 ** -0.5)
    test2 = tr_mod.sparse_attn(q2, kv2, sink2, idx2, D2 ** -0.5)
    passed &= check_close("sparse_attn B=1,S=4,H=16,D=512,K=64", ref2, test2, atol=0.05, rtol=0.02)

    return passed


def test_hc_split_sinkhorn():
    print("\n=== test_hc_split_sinkhorn ===")
    passed = True

    hc = 4
    mix_dim = (2 + hc) * hc  # 24
    B, S = 2, 8

    mixes = torch.randn(B, S, mix_dim, device="cuda", dtype=torch.float32)
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = torch.randn(mix_dim, device="cuda", dtype=torch.float32)

    ref_pre, ref_post, ref_comb = tl_mod.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, 20, 1e-6)
    test_pre, test_post, test_comb = tr_mod.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, 20, 1e-6)

    passed &= check_close("pre", ref_pre, test_pre, atol=1e-3, rtol=1e-3)
    passed &= check_close("post", ref_post, test_post, atol=1e-3, rtol=1e-3)
    passed &= check_close("comb", ref_comb, test_comb, atol=1e-3, rtol=1e-3)

    return passed


if __name__ == "__main__":
    all_passed = True
    all_passed &= test_act_quant()
    all_passed &= test_fp4_act_quant()
    all_passed &= test_fp8_gemm()
    all_passed &= test_fp4_gemm()
    all_passed &= test_sparse_attn()
    all_passed &= test_hc_split_sinkhorn()

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    sys.exit(0 if all_passed else 1)

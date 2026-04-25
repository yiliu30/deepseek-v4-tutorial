"""
Triton reimplementation of all 6 DeepSeek V4 kernels.
Same Python-level API as tilelang_kernel.py.

Kernels:
  1. act_quant      — FP8 block-wise quantization
  2. fp4_act_quant  — FP4 block-wise quantization (inplace only; non-inplace uses PyTorch)
  3. fp8_gemm       — FP8 GEMM with per-block scaling
  4. fp4_gemm       — FP8 act × FP4 weight GEMM
  5. sparse_attn    — Sparse attention via index gathering + online softmax
  6. hc_split_sinkhorn — Hyper-Connection split + Sinkhorn normalization
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. act_quant — Block-wise FP8 quantization
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _act_quant_kernel(
    X_ptr, Y_ptr, S_ptr,
    M, N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ROUND_SCALE: tl.constexpr, INPLACE: tl.constexpr,
    stride_xm, stride_ym, stride_sm,
):
    """Block-wise FP8 quantization kernel.
    Each program handles one (row, block) tile of size [1, BLOCK_SIZE]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load block
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N
    x = tl.load(X_ptr + pid_m * stride_xm + offs_n, mask=mask, other=0.0).to(tl.float32)

    # Compute amax and scale
    amax = tl.max(tl.abs(x))
    amax = tl.maximum(amax, 1e-4)

    FP8_MAX: tl.constexpr = 448.0
    FP8_MAX_INV: tl.constexpr = 1.0 / 448.0

    if ROUND_SCALE:
        # Power-of-2 rounding: scale = 2^ceil(log2(amax / 448))
        raw = amax * FP8_MAX_INV
        log2_val = tl.math.log2(raw)
        ceil_log2 = tl.math.ceil(log2_val).to(tl.int32)
        scale = tl.math.exp2(ceil_log2.to(tl.float32))
    else:
        scale = amax * FP8_MAX_INV

    # Quantize
    y = x / scale
    y = tl.clamp(y, -FP8_MAX, FP8_MAX)

    if INPLACE:
        # FP8 simulate: cast to FP8 then back to BF16 (QAT-style)
        y_fp8 = y.to(tl.float8e4nv)
        y_out = y_fp8.to(tl.float32) * scale
        tl.store(Y_ptr + pid_m * stride_ym + offs_n, y_out.to(tl.bfloat16), mask=mask)
    else:
        y_fp8 = y.to(tl.float8e4nv)
        tl.store(Y_ptr + pid_m * stride_ym + offs_n, y_fp8, mask=mask)

    # Store scale
    tl.store(S_ptr + pid_m * stride_sm + pid_n, scale)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP8 quantization. Same API as tilelang version."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    flat = z.view(-1, N)
    M = flat.shape[0]

    if inplace:
        y = torch.empty_like(flat)
    else:
        y = torch.empty(M, N, device=z.device, dtype=torch.float8_e4m3fn)

    s = torch.empty(M, N // block_size, device=z.device, dtype=torch.float32)

    grid = (M, N // block_size)
    _act_quant_kernel[grid](
        flat, y, s,
        M, N, block_size,
        scale_fmt is not None, inplace,
        flat.stride(0), y.stride(0), s.stride(0),
    )

    # Convert scale dtype if needed
    if scale_dtype == torch.float8_e8m0fnu:
        s = s.to(torch.float8_e8m0fnu)

    if inplace:
        x.copy_(y.view_as(x))
        return x

    s = s.view(*z.shape[:-1], N // block_size)
    return y.view_as(z).to(torch.float8_e4m3fn), s


# ═══════════════════════════════════════════════════════════════════════════════
# 2. fp4_act_quant — Block-wise FP4 quantization
# ═══════════════════════════════════════════════════════════════════════════════

# FP4 e2m1 representable values (positive): 0, 0.5, 1, 1.5, 2, 3, 4, 6
# Full range: {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}

@triton.jit
def _fp4_quant_inplace_kernel(
    X_ptr, Y_ptr, S_ptr,
    M, N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    stride_xm, stride_ym, stride_sm,
):
    """FP4 inplace quantization: quantize to FP4 range then dequantize back to BF16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N
    x = tl.load(X_ptr + pid_m * stride_xm + offs_n, mask=mask, other=0.0).to(tl.float32)

    # Compute amax and power-of-2 scale
    amax = tl.max(tl.abs(x))
    FP4_MAX: tl.constexpr = 6.0
    FP4_MAX_INV: tl.constexpr = 1.0 / 6.0
    # Minimum amax to avoid denorm scales
    amax = tl.maximum(amax, FP4_MAX * 1.1754943508222875e-38)  # 6 * 2^-126

    # Power-of-2 scale
    raw = amax * FP4_MAX_INV
    log2_val = tl.math.log2(raw)
    ceil_log2 = tl.math.ceil(log2_val).to(tl.int32)
    scale = tl.math.exp2(ceil_log2.to(tl.float32))

    # Quantize: clamp to FP4 range, simulate by rounding to nearest FP4 value
    y_scaled = x / scale
    y_clamped = tl.clamp(y_scaled, -FP4_MAX, FP4_MAX)

    # Simulate FP4 e2m1 rounding: the representable values are
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6} and negatives.
    # A simple approximation: round to nearest 0.5 for |x|<=2, then
    # use coarser steps. But for accuracy parity with tilelang (which
    # uses hardware FP4 cast), we use the exact FP4 e2m1 rounding.
    #
    # FP4 e2m1 encoding (unsigned mantissa):
    #   exp=0: 0, 0.5  (subnormal)
    #   exp=1: 1, 1.5
    #   exp=2: 2, 3
    #   exp=3: 4, 6
    #
    # For simplicity and accuracy match, we'll round to the nearest
    # representable value using a lookup approach in registers.
    # Since we need exact match with hardware FP4 cast, we'll use
    # the fact that PyTorch supports float4_e2m1fn_x2 for the actual
    # rounding.
    #
    # INPLACE MODE: cast to FP4 then back. We do this via the scaled value.
    # The tilelang kernel does: Cast(bf16, Cast(f32, Cast(fp4, clamped)) * scale)
    # We simulate the FP4 cast by rounding to nearest representable value.

    # Compute absolute value and sign
    sign = tl.where(y_clamped < 0, -1.0, 1.0)
    abs_val = tl.abs(y_clamped)

    # Round to nearest FP4 e2m1 value
    # Boundaries between representable values:
    # 0 <-> 0.5: boundary at 0.25
    # 0.5 <-> 1: boundary at 0.75
    # 1 <-> 1.5: boundary at 1.25
    # 1.5 <-> 2: boundary at 1.75
    # 2 <-> 3: boundary at 2.5
    # 3 <-> 4: boundary at 3.5
    # 4 <-> 6: boundary at 5.0
    rounded = tl.where(abs_val < 0.25, 0.0,
              tl.where(abs_val < 0.75, 0.5,
              tl.where(abs_val < 1.25, 1.0,
              tl.where(abs_val < 1.75, 1.5,
              tl.where(abs_val < 2.5, 2.0,
              tl.where(abs_val < 3.5, 3.0,
              tl.where(abs_val < 5.0, 4.0,
              6.0)))))))

    y_dequant = sign * rounded * scale
    tl.store(Y_ptr + pid_m * stride_ym + offs_n, y_dequant.to(tl.bfloat16), mask=mask)
    tl.store(S_ptr + pid_m * stride_sm + pid_n, scale)


def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP4 quantization. inplace=True does fused quant+dequant back to BF16."""
    N = x.size(-1)
    assert N % block_size == 0

    if not inplace:
        # Non-inplace needs actual FP4 packed output — use PyTorch
        z = x.contiguous().float()
        flat = z.view(-1, N)
        M = flat.shape[0]
        s = torch.empty(M, N // block_size, device=x.device, dtype=torch.float32)

        # Per-block quantization via PyTorch
        blocks = flat.view(M, N // block_size, block_size)
        amax = blocks.abs().amax(dim=-1).clamp(min=6.0 * (2**-126))
        raw = amax / 6.0
        ceil_log2 = torch.ceil(torch.log2(raw)).to(torch.int32)
        scale = torch.pow(2.0, ceil_log2.float())
        s = scale.to(torch.float8_e8m0fnu)

        scaled = blocks / scale.unsqueeze(-1)
        clamped = scaled.clamp(-6.0, 6.0)
        # Cast through FP4 for exact rounding
        y_flat = clamped.reshape(M, N).to(torch.float4_e2m1fn_x2)
        y = y_flat.view(*x.shape[:-1], N // 2)
        s = s.view(*x.shape[:-1], N // block_size)
        return y, s

    # Inplace mode: Triton kernel
    z = x.contiguous()
    flat = z.view(-1, N)
    M = flat.shape[0]
    y = torch.empty_like(flat)
    s = torch.empty(M, N // block_size, device=z.device, dtype=torch.float32)

    grid = (M, N // block_size)
    _fp4_quant_inplace_kernel[grid](
        flat, y, s,
        M, N, block_size,
        flat.stride(0), y.stride(0), s.stride(0),
    )
    x.copy_(y.view_as(x))
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# 3. fp8_gemm — FP8 GEMM with per-block scaling
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _fp8_gemm_kernel(
    A_ptr, B_ptr, C_ptr, SA_ptr, SB_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_sam, stride_sak,
    stride_sbn, stride_sbk,
):
    """C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T with per-block scaling."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=a_mask, other=0.0)

        # Load B tile [BLOCK_N, BLOCK_K]
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                     mask=b_mask, other=0.0)

        # Scale: A has per-row per-128 scales, B has per-128-group scales
        k_group = k_start // GROUP_SIZE

        # A scale: [BLOCK_M] — one per row
        sa = tl.load(SA_ptr + offs_m * stride_sam + k_group * stride_sak,
                      mask=offs_m < M, other=1.0).to(tl.float32)

        # B scale: scalar (one per N-group × K-group)
        # B_s layout: [ceil(N/GROUP_SIZE), ceil(K/GROUP_SIZE)]
        # For simplicity, use the group for the first element of this N-block
        n_group = pid_n * BLOCK_N // GROUP_SIZE
        sb = tl.load(SB_ptr + n_group * stride_sbn + k_group * stride_sbk).to(tl.float32)

        # Compute: cast to float, matmul, apply scale
        a_f32 = a.to(tl.float32)
        b_f32 = b.to(tl.float32)

        # A[BLOCK_M, BLOCK_K] @ B[BLOCK_N, BLOCK_K]^T = [BLOCK_M, BLOCK_N]
        tile = tl.dot(a, b.trans(1, 0))

        # Apply scale: tile[m,n] *= sa[m] * sb
        tile = tile * (sa[:, None] * sb)
        acc += tile

    # Store result
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.bfloat16), mask=c_mask)


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-128 block FP8 scaling."""
    assert a.is_contiguous() and b.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)

    # Convert scales to float32 for the kernel
    a_s_f32 = a_s.float().contiguous()
    b_s_f32 = b_s.float().contiguous()

    a_flat = a.view(M, K)
    c = torch.empty(M, N, device=a.device, dtype=torch.get_default_dtype())

    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 128
    GROUP_SIZE = 128

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    _fp8_gemm_kernel[grid](
        a_flat, b, c, a_s_f32.view(M, -1), b_s_f32,
        M, N, K, GROUP_SIZE,
        BLOCK_M, BLOCK_N, BLOCK_K,
        a_flat.stride(0), a_flat.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_s_f32.view(M, -1).stride(0), a_s_f32.view(M, -1).stride(1),
        b_s_f32.stride(0), b_s_f32.stride(1),
    )
    return c.view(*a.shape[:-1], N)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. fp4_gemm — FP8 activation × FP4 weight GEMM
# ═══════════════════════════════════════════════════════════════════════════════

# FP4 e2m1 lookup table for dequantization (nibble index → float value)
_FP4_LUT = None

def _get_fp4_lut(device):
    global _FP4_LUT
    if _FP4_LUT is None or _FP4_LUT.device != device:
        _FP4_LUT = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32, device=device,
        )
    return _FP4_LUT


def _unpack_fp4(b_packed: torch.Tensor) -> torch.Tensor:
    """Unpack float4_e2m1fn_x2 [N, K//2] → float32 [N, K] via LUT."""
    lut = _get_fp4_lut(b_packed.device)
    u8 = b_packed.view(torch.uint8)
    lo = (u8 & 0xF).long()
    hi = (u8 >> 4).long()
    # Interleave: for each byte, low nibble is first element, high nibble is second
    unpacked = torch.stack([lo, hi], dim=-1).flatten(-2)
    return lut[unpacked]


def fp4_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T.

    Dequantize both inputs to BF16 then matmul. Accuracy-first approach.
    """
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)

    # Dequantize A: FP8 * scale → BF16
    a_flat = a.view(M, K).float()
    a_s_f32 = a_s.float().view(M, -1)
    act_group = 128
    a_dequant = a_flat.view(M, -1, act_group) * a_s_f32.unsqueeze(-1)
    a_dequant = a_dequant.view(M, K)

    # Dequantize B: unpack FP4 via LUT, then apply per-32 scale
    b_f32 = _unpack_fp4(b)  # [N, K]
    b_s_f32 = b_s.float().view(N, -1)
    weight_group = 32
    b_dequant = b_f32.view(N, -1, weight_group) * b_s_f32.unsqueeze(-1)
    b_dequant = b_dequant.view(N, K)

    # Matmul: [M, K] @ [N, K]^T = [M, N]
    c = torch.mm(a_dequant.bfloat16(), b_dequant.bfloat16().t())
    return c.view(*a.shape[:-1], N).to(torch.get_default_dtype())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. sparse_attn — Sparse attention via index gathering + online softmax
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _sparse_attn_kernel(
    Q_ptr, KV_ptr, O_ptr, SINK_ptr, IDX_ptr,
    B, S, H: tl.constexpr, D: tl.constexpr, T, TOPK,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kvb, stride_kvt, stride_kvd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_ib, stride_is, stride_ik,
):
    """One program handles one (batch, seq_pos, head).
    Iterates over top-k KV positions in blocks of BLOCK_K, using online softmax."""
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Decode (batch, seq) from linear pid
    pid_b = pid // S
    pid_s = pid % S

    # Load query vector [D]
    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + pid_b * stride_qb + pid_s * stride_qs + pid_h * stride_qh + offs_d * stride_qd).to(tl.float32)

    # Online softmax state
    m_prev = float('-inf')  # running max
    l_prev = 0.0  # running sum of exp
    acc = tl.zeros((D,), dtype=tl.float32)  # running weighted sum

    num_blocks = tl.cdiv(TOPK, BLOCK_K)
    for t in range(num_blocks):
        offs_k = t * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < TOPK

        # Load indices [BLOCK_K]
        idxs = tl.load(IDX_ptr + pid_b * stride_ib + pid_s * stride_is + offs_k * stride_ik,
                        mask=k_mask, other=-1)

        # Load KV vectors [BLOCK_K, D] via gathering
        valid = (idxs != -1) & k_mask
        safe_idxs = tl.maximum(idxs, 0)

        # Compute scores: q · kv[idx] for each idx
        # Load KV one at a time and dot with q
        kv_block = tl.load(
            KV_ptr + pid_b * stride_kvb + safe_idxs[:, None] * stride_kvt + offs_d[None, :] * stride_kvd,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        # Scores [BLOCK_K] = kv_block @ q
        scores = tl.sum(kv_block * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_prev, tl.max(scores))
        # Correct previous accumulator
        alpha = tl.math.exp(m_prev - m_new)
        # New exp scores
        p = tl.math.exp(scores - m_new)
        l_new = l_prev * alpha + tl.sum(p)

        # Update accumulator: acc = acc * alpha + p @ kv_block
        acc = acc * alpha + tl.sum(p[:, None] * kv_block, axis=0)

        m_prev = m_new
        l_prev = l_new

    # Add sink token: exp(sink_bias - m) contributes to denominator only (zero value)
    sink_bias = tl.load(SINK_ptr + pid_h).to(tl.float32)
    l_prev += tl.math.exp(sink_bias - m_prev)

    # Normalize
    acc = acc / l_prev

    # Store output [D]
    tl.store(O_ptr + pid_b * stride_ob + pid_s * stride_os + pid_h * stride_oh + offs_d * stride_od,
             acc.to(tl.bfloat16))


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor, softmax_scale: float,
) -> torch.Tensor:
    """Sparse attention via index gathering. Triton implementation.

    q: [B, S, H, D], kv: [B, T, D], attn_sink: [H],
    topk_idxs: [B, S, K], softmax_scale: float
    Returns: [B, S, H, D]
    """
    B, S, H, D = q.shape
    T = kv.shape[1]
    K = topk_idxs.shape[-1]

    o = torch.empty_like(q)

    # Pad BLOCK_K to power of 2 for efficiency
    BLOCK_K = triton.next_power_of_2(min(K, 64))

    grid = (B * S, H)
    _sparse_attn_kernel[grid](
        q, kv, o, attn_sink, topk_idxs,
        B, S, H, D, T, K,
        softmax_scale,
        BLOCK_K,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        topk_idxs.stride(0), topk_idxs.stride(1), topk_idxs.stride(2),
    )
    return o


# ═══════════════════════════════════════════════════════════════════════════════
# 6. hc_split_sinkhorn — Hyper-Connection split + Sinkhorn normalization
# ═══════════════════════════════════════════════════════════════════════════════

def hc_split_sinkhorn(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
    hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6,
):
    """Hyper-Connection split + Sinkhorn normalization. PyTorch implementation.

    hc_mult is always 4 and the matrix is 4×4, so Sinkhorn is cheap.
    A Triton kernel would add complexity for negligible gain on a 4×4 matrix.
    """
    b, s, _ = mixes.size()
    hc = hc_mult
    mix_hc = (2 + hc) * hc  # 24 for hc=4

    m = mixes.view(b * s, mix_hc).float()
    scale = hc_scale.float()
    base = hc_base.float()

    # Apply scale and base
    m = m * scale.new_ones(mix_hc)  # placeholder; need per-section scaling

    # pre: sigmoid(mixes[:hc] * scale[0] + base[:hc]) + eps
    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + eps

    # post: 2 * sigmoid(mixes[hc:2*hc] * scale[1] + base[hc:2*hc])
    post = 2 * torch.sigmoid(m[:, hc:2*hc] * scale[1] + base[hc:2*hc])

    # comb: mixes[2*hc:] reshaped to [N, hc, hc]
    comb_raw = m[:, 2*hc:].view(-1, hc, hc) * scale[2] + base[2*hc:].view(hc, hc)

    # softmax along last dim + eps
    comb = torch.softmax(comb_raw, dim=-1) + eps

    # Sinkhorn: alternate row/col normalization
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    pre = pre.view(b, s, hc).to(mixes.dtype)
    post = post.view(b, s, hc).to(mixes.dtype)
    comb = comb.view(b, s, hc, hc).to(mixes.dtype)
    return pre, post, comb

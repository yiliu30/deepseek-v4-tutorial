"""
Hybrid kernel.py for DeepSeek V4 on RTX 5090 D (SM 12.0).

Uses real tilelang kernels for 5/6 functions:
  ✓ act_quant      — FP8 block-wise quantization
  ✓ fp4_act_quant  — FP4 block-wise quantization
  ✓ fp8_gemm       — FP8 GEMM with per-block scaling
  ✓ fp4_gemm       — FP4 GEMM
  ✓ hc_split_sinkhorn — Hyper-Connection split + Sinkhorn normalization

  ✗ sparse_attn    — PyTorch fallback (tilelang kernel needs 141KB shared memory,
                      RTX 5090 D only supports 99KB optin max)
"""
import os

# Import 5 real tilelang kernels from the original kernel.py via importlib.util
# (avoids circular import since both files are named "kernel")
import importlib.util

_orig_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tilelang_kernel.py"
)
_spec = importlib.util.spec_from_file_location("_tilelang_kernel", _orig_path)
_orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig)

act_quant = _orig.act_quant
fp4_act_quant = _orig.fp4_act_quant
fp8_gemm = _orig.fp8_gemm
fp4_gemm = _orig.fp4_gemm
hc_split_sinkhorn = _orig.hc_split_sinkhorn


# ── sparse_attn: PyTorch fallback (shared memory limit on RTX 5090 D) ────────

import torch


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Sparse attention via index gathering. Pure PyTorch fallback.

    q: [B, S, H, D] — queries
    kv: [B, T, D] — key-value (shared across heads, MLA style)
    attn_sink: [H] — per-head learnable bias added before softmax
    topk_idxs: [B, S, K] — indices into kv's T dimension (-1 = masked)
    softmax_scale: 1/sqrt(d)

    Returns: [B, S, H, D]
    """
    b, s, h, d = q.shape
    k = topk_idxs.shape[-1]

    # Gather KV by topk indices: [B, S, K, D]
    topk_idxs = topk_idxs.to(q.device)
    mask = topk_idxs == -1  # [B, S, K]
    safe_idxs = topk_idxs.clamp(min=0).long()  # [B, S, K]

    # kv_gathered[b, s, k, :] = kv[b, safe_idxs[b,s,k], :]
    idx_expanded = safe_idxs.unsqueeze(-1).expand(b, s, k, d)  # [B, S, K, D]
    kv_expanded = kv.unsqueeze(1).expand(b, s, kv.shape[1], d)  # [B, S, T, D]
    kv_gathered = torch.gather(kv_expanded, 2, idx_expanded)  # [B, S, K, D]

    # Q @ K^T → scores [B, S, H, K]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), kv_gathered.float()) * softmax_scale

    # Mask out -1 positions
    scores = scores.masked_fill(mask.unsqueeze(2), float("-inf"))

    # Add attn_sink as virtual "sink" token (zero value, learnable bias)
    sink_score = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    scores_with_sink = torch.cat([scores, sink_score], dim=-1)  # [B, S, H, K+1]

    # Softmax → remove sink weight → weighted sum
    attn_weights = torch.softmax(scores_with_sink, dim=-1)
    attn_weights = attn_weights[..., :-1]  # [B, S, H, K]

    o = torch.einsum("bshk,bskd->bshd", attn_weights, kv_gathered.float())
    return o.to(q.dtype)

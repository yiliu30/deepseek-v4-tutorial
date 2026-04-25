"""
Kernel dispatch for DeepSeek V4 on RTX 5090 D (SM 12.0).

Backend selection via KERNEL_BACKEND env var:
  - "triton" → Pure Triton kernels (all 6 including sparse_attn)
  - "tilelang" (default) → 5 tilelang kernels + PyTorch sparse_attn fallback
                           (tilelang sparse_attn needs 141KB shared memory,
                            RTX 5090 D only supports 99KB optin max)
"""
import os

BACKEND = os.environ.get("KERNEL_BACKEND", "triton").lower()

if BACKEND == "triton":
    from triton_kernel import (
        act_quant, fp4_act_quant, fp8_gemm, fp4_gemm,
        sparse_attn, hc_split_sinkhorn,
    )
else:
    # Import 5 real tilelang kernels via importlib (avoids circular import)
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

    # sparse_attn: PyTorch fallback (shared memory limit on RTX 5090 D)
    import torch

    def sparse_attn(
        q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor, softmax_scale: float
    ) -> torch.Tensor:
        """Sparse attention via index gathering. Pure PyTorch fallback."""
        b, s, h, d = q.shape
        k = topk_idxs.shape[-1]

        topk_idxs = topk_idxs.to(q.device)
        mask = topk_idxs == -1
        safe_idxs = topk_idxs.clamp(min=0).long()

        idx_expanded = safe_idxs.unsqueeze(-1).expand(b, s, k, d)
        kv_expanded = kv.unsqueeze(1).expand(b, s, kv.shape[1], d)
        kv_gathered = torch.gather(kv_expanded, 2, idx_expanded)

        scores = torch.einsum("bshd,bskd->bshk", q.float(), kv_gathered.float()) * softmax_scale
        scores = scores.masked_fill(mask.unsqueeze(2), float("-inf"))

        sink_score = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
        scores_with_sink = torch.cat([scores, sink_score], dim=-1)

        attn_weights = torch.softmax(scores_with_sink, dim=-1)
        attn_weights = attn_weights[..., :-1]

        o = torch.einsum("bshk,bskd->bshd", attn_weights, kv_gathered.float())
        return o.to(q.dtype)

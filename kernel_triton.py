"""
Drop-in kernel replacement using Triton instead of tilelang.

To use: set KERNEL_BACKEND=triton before importing model.py,
or rename this file to kernel.py.

All 6 kernels (including sparse_attn) are Triton-based.
No tilelang dependency required.
"""
from triton_kernel import (
    act_quant,
    fp4_act_quant,
    fp8_gemm,
    fp4_gemm,
    sparse_attn,
    hc_split_sinkhorn,
)

__all__ = [
    "act_quant",
    "fp4_act_quant",
    "fp8_gemm",
    "fp4_gemm",
    "sparse_attn",
    "hc_split_sinkhorn",
]

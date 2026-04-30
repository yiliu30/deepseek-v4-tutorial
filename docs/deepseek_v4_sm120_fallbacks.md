# DeepSeek V4 — SM80/SM120 Fallback Kernel Map

> **Scope**: DeepSeek V4 (and V3.2) sparse MLA inference on GPUs that lack
> DeepGEMM and FlashMLA support — primarily **SM80** (A100) and **SM120**
> (RTX 6000D / consumer Blackwell).
>
> On SM90 (H100/H200) and SM100 (B200), vLLM uses DeepGEMM CUDA kernels and
> FlashMLA for peak throughput. On unsupported architectures every DeepGEMM
> call-site has a fallback path — either a **Triton kernel** or a **PyTorch
> reference implementation**.

## Architecture overview

The DeepSeek V4 sparse MLA pipeline has four major stages, each with one or
more kernel dispatch points:

```
 ┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────┐
 │  1. Indexer  │────▶│ 2. Sparse Attn   │────▶│ 3. FP8 Einsum     │────▶│ 4. O-project │
 │  (topk sel) │     │  (decode/prefill) │     │  (KV absorption)  │     │  (output)    │
 └─────────────┘     └──────────────────┘     └───────────────────┘     └──────────────┘
```

## Dispatch point summary

| # | Stage | Function | SM90/SM100 (DeepGEMM) | SM80/SM120 (Fallback) | Gate |
|---|-------|----------|-----------------------|-----------------------|------|
| 1a | Indexer prefill | `fp8_fp4_mqa_logits()` | DeepGEMM CUDA kernel | `fp8_mqa_logits_triton()` — Triton | `is_deep_gemm_supported()` |
| 1b | Indexer decode  | `fp8_fp4_paged_mqa_logits()` | DeepGEMM paged CUDA kernel | `fp8_paged_mqa_logits_triton()` — Triton | `is_deep_gemm_supported()` |
| 2  | Sparse MLA decode | `flash_mla_sparse_fwd()` | FlashMLA CUDA kernel | `triton_sparse_mla_attention()` — Triton split-KV | `TRITON_MLA_SPARSE` backend selection |
| 3a | Sparse attn prefill (SWA) | `flash_mla_sparse_fwd()` | FlashMLA CUDA kernel | `_ref_sparse_attn_prefill()` — PyTorch SDPA | `use_dsv4_reference_kernels()` |
| 3b | Sparse attn decode (SWA) | `flash_mla_sparse_fwd()` | FlashMLA CUDA kernel | `_ref_sparse_attn_decode_gather()` — PyTorch SDPA | `use_dsv4_reference_kernels()` |
| 4  | FP8 einsum (KV absorption) | `deep_gemm.fp8_einsum()` | DeepGEMM FP8 GEMM | `_deepseek_v4_fp8_einsum_fallback()` — dequant + `torch.einsum` | `use_dsv4_reference_kernels()` |
| 5  | O-projection | fused inv-RoPE + FP8 quant → DeepGEMM einsum | Fused CUDA kernel | inv-RoPE → BF16 → Marlin GEMM / `torch.bmm` | `use_dsv4_reference_kernels()` |
| 6  | K-cache dequant | `_dequantize_and_gather_k_kernel` | Triton (FP8→BF16 with `float8e4nv`) | `deepseek_v4_dequant_gather_sm80` — Triton (uint8 bitcast, no `float8e4nv`) | `use_dsv4_reference_kernels()` |

## Detailed description of each fallback

### 1. Indexer MQA logits (topk selection)

**File**: `vllm/model_executor/layers/sparse_attn_indexer.py`

The sparse attention indexer computes per-head logits between queries and
compressed KV entries to select the top-k KV positions. On SM90/SM100 this
uses DeepGEMM's `fp8_fp4_mqa_logits` (prefill) and `fp8_fp4_paged_mqa_logits`
(decode), which operate on FP8 or MXFP4 inputs directly in CUDA.

**SM80/SM120 replacement**: Triton kernels `fp8_mqa_logits_triton()` and
`fp8_paged_mqa_logits_triton()` from
`vllm/v1/attention/ops/mqa_logits_triton.py`. These cast FP8 to BF16, run a
BF16 matmul with FP32 accumulator, apply ReLU + head-weight reduction, and
mask invalid positions to `-inf`. FP4 cache is **not supported** on this path.

**Gate**: `is_deep_gemm_supported()` — checks `VLLM_USE_DEEP_GEMM` env var,
library availability, and GPU architecture support.

### 2. Sparse MLA decode attention

**File**: `vllm/v1/attention/backends/mla/triton_mla_sparse.py`

The `TRITON_MLA_SPARSE` backend replaces FlashMLA Sparse for the main
decode attention over the selected top-k KV positions. It uses a split-KV
Triton kernel (`triton_sparse_mla_attention`) that partitions the top-k
indices across multiple KV splits, runs independent online-softmax passes,
and merges with a log-sum-exp reduction.

**Gate**: User selects via `--attention-config.backend TRITON_MLA_SPARSE`.
This is a standalone backend in the registry — it doesn't conditionally
dispatch; if selected, it always uses Triton.

### 3. SWA (Sliding Window Attention) sparse attention

**File**: `vllm/model_executor/layers/deepseek_v4_attention.py`

The compressed + SWA attention stages on SM90/SM100 use `flash_mla_sparse_fwd`
(FlashMLA). On SM80/SM120, these fall back to reference PyTorch
implementations:

- **Prefill**: `_ref_sparse_attn_prefill()` — gathers KV entries by topk
  indices, applies `torch.nn.functional.scaled_dot_product_attention`, then
  merges compressed and SWA outputs via N-way LSE merge with attention sink.
- **Decode**: `_ref_sparse_attn_decode_gather()` — similar gather + SDPA
  approach with separate compressed/SWA accumulation and LSE-based merge.

**Gate**: `use_dsv4_reference_kernels()` — returns `True` when
`is_deep_gemm_supported()` is False or FlashMLA Sparse is unavailable.

### 4. FP8 einsum (latent KV absorption)

**File**: `vllm/model_executor/layers/deepseek_v4_attention.py`

The "absorption" step fuses W_Q and W_KV projections via FP8 einsum
(`bm,bhm->bh` and `bh,bhm->bm`). On SM90/SM100, `deep_gemm.fp8_einsum()`
runs a highly optimized TMA-based CUDA kernel.

**SM80/SM120 replacement**: `_deepseek_v4_fp8_einsum_fallback()` — dequantizes
FP8 inputs to BF16, runs `torch.einsum()`, and re-scales. Functionally
correct but significantly slower.

**Einsum recipe selection** (only used on the DeepGEMM path):
- SM90 (Hopper): scale-factor block granularity `(1, 128, 128)`
- SM100 (Blackwell): `(1, 1, 128)` — different TMA layout

### 5. O-projection (output projection)

**File**: `vllm/model_executor/layers/deepseek_v4_attention.py`

On SM90/SM100 the output path is fused: `fused_inv_rope_fp8_quant` applies
inverse RoPE and FP8 quantization in one kernel, then DeepGEMM einsum
computes the low-rank `wo_a` projection.

**SM80/SM120 replacement**: Separate inverse RoPE (`_apply_inv_rope_to_o`),
BF16 dequant, then standard Marlin GEMM (`wo_b(wo_a(o))`) or `torch.bmm`
for multi-group configs.

### 6. K-cache dequantization

**File**: `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`

The FP8 K-cache needs dequantization for reference attention paths. The SM90
Triton kernel uses native `float8e4nv` type support. SM80 lacks this Triton
type, so a separate kernel (`_dequantize_and_gather_k_kernel_sm80`) operates
on raw `uint8` and applies manual FP8→FP32 conversion.

## How to run on SM80/SM120

```bash
# Minimal command — TRITON_MLA_SPARSE handles indexer + decode attention,
# reference kernels handle SWA/einsum/O-proj automatically.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V3 \
    -tp 4 \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --attention-config.backend TRITON_MLA_SPARSE

# For offline inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/basic/offline_inference/generate.py \
    --model deepseek-ai/DeepSeek-V4-Flash \
    -tp 4 \
    --kv-cache-dtype fp8 \
    --max-model-len 2084 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --attention-config.backend TRITON_MLA_SPARSE
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_USE_DEEP_GEMM` | `1` | Master switch for DeepGEMM. Set `0` to force all fallbacks even on SM90. |
| `VLLM_SM120_DISABLE_DEEPGEMM` | `0` | SM120-specific override to disable DeepGEMM dispatch in `deep_gemm.py` wrappers. |
| `VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION` | `0` | Force reference attention even if FlashMLA is available. |
| `VLLM_SM120_TRITON_MLA` | `0` | Enable Triton MLA path in `deepseek_v4_attention.py` (alternative to `TRITON_MLA_SPARSE` backend). |

## Performance expectations

The fallback paths are **functionally correct** but substantially slower than
the DeepGEMM/FlashMLA paths:

- **Indexer logits**: Triton kernels are ~2–4× slower than DeepGEMM
- **Sparse MLA decode**: Triton split-KV is ~3–7× slower than FlashMLA Sparse
- **FP8 einsum**: `torch.einsum` fallback is ~5–10× slower (no FP8 compute)
- **SWA attention**: PyTorch SDPA is ~2–5× slower than FlashMLA

These fallbacks enable **functional correctness and development** on
consumer/non-Hopper GPUs, not production throughput parity.

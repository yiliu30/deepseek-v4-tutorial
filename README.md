# DeepSeek V4 Tutorial

Study and run DeepSeek V4 (Flash) on consumer GPUs (8× RTX 5090 D).

```bash
I'm DeepSeek 👋
>>> The capital of France is
The capital of France is Paris.<｜end▁of▁sentence｜>
```

## Hardware & Software

| Component | Spec |
|-----------|------|
| GPU | 8× NVIDIA GeForce RTX 5090 D (32GB each, SM 12.0 Blackwell) |
| Driver | 580.142 |
| PyTorch | 2.11.0 |
| tilelang | 0.1.9 |
| Model | DeepSeek-V4-Flash (43 layers, 256 experts, 4096 hidden dim, ~149GB FP4) |

## Kernel Support Status (RTX 5090 D)

| Kernel | TileLang | Triton | Note
|--------|--------|-------|-------|
| `act_quant` (FP8) | ✅ | ✅ | Block-wise FP8 quantization |
| `fp4_act_quant` (FP4) | ✅ | ✅ | Block-wise FP4 quantization |
| `fp8_gemm` | ✅ | ✅ | FP8 GEMM with per-block scaling |
| `fp4_gemm` | ✅ | ✅ | FP8 activation × FP4 weight GEMM |
| `hc_split_sinkhorn` | ✅ | ✅ | Hyper-Connection split + Sinkhorn normalization |
| `sparse_attn` | ⚠️ PyTorch fallback | ✅ | Needs 141KB shared memory; RTX 5090 D max is 99KB |

The `sparse_attn` kernel allocates all 64 attention heads × 512 dims in shared memory simultaneously. A100 (164KB) and H100 (228KB) can handle this; consumer Blackwell (99KB optin max) cannot. The PyTorch fallback uses `torch.gather` + `einsum` and produces identical results.


## Quick Start

### 1. Environment Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch>=2.10.0 tilelang==0.1.9 transformers safetensors
```

### 2. Download Model

```bash
# Download DeepSeek-V4-Flash from HuggingFace
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir /path/to/DeepSeek-V4-Flash
```

### 3. Convert Weights

Convert HuggingFace checkpoint to 8-way model-parallel shards:
- [Yi30/DeepSeek-V4-Flash-Converted-EP8](https://huggingface.co/Yi30/DeepSeek-V4-Flash-Converted-EP8)

```bash
python convert.py \
    --hf-ckpt-path /path/to/DeepSeek-V4-Flash \
    --save-path /path/to/DeepSeek-V4-Flash-converted \
    --n-experts 256 \
    --model-parallel 8
```

Copy the inference config:
```bash
cp config_flash.json /path/to/DeepSeek-V4-Flash-converted/config.json
```

### 4. Run Inference

```bash
torchrun --nproc-per-node 8 generate.py \
    --ckpt-path /path/to/DeepSeek-V4-Flash-converted \
    --config /path/to/DeepSeek-V4-Flash-converted/config.json \
    --input-file test_prompt.txt \
    --max-new-tokens 100 --temperature 0.6
```

For interactive chat:
```bash
torchrun --nproc-per-node 8 generate.py \
    --ckpt-path /path/to/DeepSeek-V4-Flash-converted \
    --config /path/to/DeepSeek-V4-Flash-converted/config.json \
    --interactive
```

### 5. Benchmark

```bash
torchrun --nproc-per-node 8 benchmark.py \
    --ckpt-path /path/to/DeepSeek-V4-Flash-converted \
    --config /path/to/DeepSeek-V4-Flash-converted/config.json \
    --prompt-lens 128,512,1024 --max-new-tokens 20 --n-runs 3
```

### 6. Architecture Walkthrough (tiny model)

Run the full model with random weights to trace tensor shapes through all components:

```bash
python walkthrough.py
```

This exercises all V4-specific components: SWA-only, C4A (compressor+indexer), C128A, MoE, Hyper-Connections.

## File Structure

```
├── README.md                    # This file
├── pyproject.toml               # Project config
├── config_flash.json            # Inference config for V4-Flash (ModelArgs format)
├── model.py                     # DeepSeek V4 reference model (with device fixes)
├── kernel.py                    # Hybrid kernel: 5 tilelang + 1 PyTorch fallback
├── fast_hadamard_transform.py   # Pure-PyTorch Hadamard transform stub
├── convert.py                   # Weight conversion from HF to inference format
├── generate.py                  # Text generation script
├── benchmark.py                 # Prefill/decode throughput benchmark
├── walkthrough.py               # Annotated tensor-shape trace through all components
├── run_tiny.py                  # Simpler driver for tiny model
├── test_prompt.txt              # Sample test prompt
└── docs/
    ├── deepseek_v4_walkthrough.md   # Full architecture walkthrough with tensor shapes
    └── kv_cache_analysis.md         # First-principles KV cache reduction analysis
```

## More Analysis
- [KV Cache](./docs/v4_pro_vs_flash_arch_and_kvcache.md)

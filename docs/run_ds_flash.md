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
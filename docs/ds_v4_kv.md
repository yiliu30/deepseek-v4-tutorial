# DeepSeek V4 KV Cache Design

## The Problem

Transformer attention is O(S²) in memory and compute. For long sequences
(1M+ tokens), storing and attending to all KV pairs is prohibitive. DeepSeek V4
solves this with **three orthogonal KV reduction methods**, combined differently
per layer.

## Three Primitives

### 1. Sliding Window Attention (SWA)

**What**: Only attend to the most recent `window_size` tokens (128 in V4).

**Result**: `[128, 512-dim]` per layer — fixed cost regardless of sequence length.

### 2. Token Compression

**What**: Learn to compress `ratio` consecutive tokens into 1 compressed KV
vector via gated pooling (softmax over learned scores × weighted sum of KV
states).

Two variants:
- **C4A (ratio=4, overlap=True)**: Overlapping 8-token windows (4 current + 4
  prior) for smoother boundaries → `[S/4, 512-dim]`
- **C128A (ratio=128, overlap=False)**: Non-overlapping 128-token blocks →
  `[S/128, 512-dim]`

### 3. Indexer (Learned Sparse Selection)

**What**: Instead of attending to ALL compressed positions, learn to select the
top-k most relevant ones per query. Uses its own smaller compressor to build a
128-dim scoring cache.

**Result**: Indexer KV cache `[S/4, 128-dim]` — used only for scoring, not
attention.

## Layer Configurations (V4 Pro)

### 1. Compressed Sparse Attention (CSA) — 30 layers

Combines: **SWA + C4A Compression + Indexer**

Three KV caches per layer:
- SWA: `[128, 512-dim]` — sliding window of raw tokens
- C4A main KV: `[S/4, 512-dim]` — compressed entries for attention
- C4A indexer KV: `[S/4, 128-dim]` — smaller cache for top-k scoring

### 2. Heavily Compressed Attention (HCA) — 31 layers

Combines: **SWA + C128A Compression** (no indexer — too few entries to select from)

Two KV caches per layer:
- SWA: `[128, 512-dim]` — sliding window of raw tokens
- C128A main KV: `[S/128, 512-dim]` — heavily compressed entries, attend to all

## 1M Token Example (V4 Pro, BF16)

All entries stored as BF16: main KV = 512 × 2 = **1024 bytes/slot**,
indexer KV = 128 × 2 = **256 bytes/slot**.

| Layer Type | KV Cache | Count | Slots/Layer | Bytes/Slot | Subtotal |
|---|---|---|---|---|---|
| CSA | SWA | 30 | 128 | 512 × 2 = 1,024 | 0.004 GiB |
| CSA | C4A main KV | 30 | 1M / 4 = 262,144 | 512 × 2 = 1,024 | 7.50 GiB |
| CSA | C4A indexer KV | 30 | 1M / 4 = 262,144 | 128 × 2 = 256 | 1.88 GiB |
| HCA | SWA | 31 | 128 | 512 × 2 = 1,024 | 0.004 GiB |
| HCA | C128A main KV | 31 | 1M / 128 = 8,192 | 512 × 2 = 1,024 | 0.25 GiB |
| **Total** | | **61** | | | **~9.62 GiB** |

> Naive full-attention BF16 (61 layers × 1M × 1024 B) = **~61 GiB** → **6.3× savings**.

> **Note**: vLLM actually stores the KV cache in FP8 with a mixed layout:
> 448B FP8 NoPE + 128B BF16 RoPE + 7B UE8M0 scales + 1B pad = **576 bytes/slot**
> for main KV, and 128B FP8 + 4B FP32 scale = **132 bytes/slot** for indexer KV.
> This reduces the total from ~9.62 GiB to **~5.39 GiB** at 1M tokens.
> (See `deepseek_v4_attention.py` lines 428–433 and 1972–1976.)

### Where the Budget Goes

```
C4A main KV ████████████████████████████████████ 78.0%  (30 × 1M/4 × 512×2 = 7.50 GiB)
C4A indexer ██████████          19.5%  (30 × 1M/4 × 128×2 = 1.88 GiB)
C128A       ██                   2.6%  (31 × 1M/128 × 512×2 = 0.25 GiB)
SWA         ▏                    0.1%  ((30+31) × 128 × 512×2 = 0.008 GiB)
```

**Takeaway**: C4A dominates the budget (~97%). The indexer adds ~20% overhead
on top of C4A main KV — the cost of learned sparse selection. C128A and SWA
are nearly free.

## CSA Pipeline (4096 Token Example)

[csa](./ds_v4_csa_pipeline.svg)

### Step 0: SWA Cache
Store recent 128 raw tokens.
→ `[128, 512-dim]`

### Step 1: Main Compressor
Compress 4096 tokens into 1024 entries via overlapping gated pooling (8-token
window with stride 4).
→ `[4096 tokens] → [4096/4 = 1024, 512-dim]` in main KV cache

### Step 2: Indexer Compressor
Separate compressor produces a smaller 128-dim representation of the same
compressed positions (FP8 or MXFP4 in vLLM).
→ `[4096 tokens] → [4096/4 = 1024, 128-dim]` in indexer KV cache

### Step 3: Indexer Scoring (via DeepGEMM)
Score each compressed position against the indexer's own query (64 heads, 128-dim
— separate from the main attention's 128 heads, 512-dim Q) using
`fp8_fp4_mqa_logits`. Produces per-position logits for ranking.
→ `Indexer_Q [1, 64 heads, 128-dim] @ K [1024, 128-dim]ᵀ → logits [1, 1024]`

### Step 4: Top-k Selection
Select top-512 compressed positions from the 1024 candidates.
→ `logits [1, 1024] → indices [1, 512]`

### Step 5: Sparse Attention (via FlashMLA)
Attend to selected compressed entries + SWA entries using FlashMLA
(FP8 KV dequantized to BF16 before tensor cores).
→ `Q × gathered_K [512+128 = 640, 512-dim] → output [1, 512-dim]`

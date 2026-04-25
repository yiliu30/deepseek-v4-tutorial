# DeepSeek V4: Pro vs Flash Architecture Comparison

Side-by-side comparison of V4-Pro (61 layers, full-size) and V4-Flash (43 layers, distilled). Both share the same novel attention architecture (C4A + C128A + SWA), same KV cache design, and same MoE backbone — Flash is a smaller, faster variant.

---

## 1. Architecture Comparison

### Parameters That Differ

| Parameter | V4-Pro | V4-Flash | Notes |
|-----------|--------|----------|-------|
| **hidden_size** | 7168 | 4096 | Model width |
| **num_hidden_layers** | 61 | 43 | Depth |
| **num_attention_heads** | 128 | 64 | Q heads (KV is always shared via MLA) |
| **q_lora_rank** | 1536 | 1024 | Q low-rank bottleneck |
| **o_groups** | 16 | 8 | Output projection groups |
| **n_routed_experts** | 384 | 256 | Expert pool size |
| **moe_intermediate_size** | 3072 | 2048 | Per-expert FFN width |
| **routed_scaling_factor** | 2.5 | 1.5 | Expert output scaling |

### Parameters That Are Identical

| Parameter | Value | Notes |
|-----------|-------|-------|
| **head_dim** | 512 | Shared KV dimension (MLA) |
| **qk_rope_head_dim** | 64 | Positional encoding dims |
| **o_lora_rank** | 1024 | Output low-rank bottleneck |
| **index_n_heads / index_head_dim** | 64 / 128 | Indexer scoring config |
| **index_topk** | 512 | Max selected compressed positions |
| **sliding_window** | 128 | SWA window (causal gap-filling) |
| **max_position_embeddings** | 1,048,576 | 1M tokens |
| **num_experts_per_tok** | 6 | Active experts per token |
| **n_shared_experts** | 1 | Always-on shared expert |
| **rope_theta / compress_rope_theta** | 10,000 / 160,000 | Base / compressed RoPE frequency |
| **YaRN factor** | 16 | Context extension scaling |
| **quant_method** | fp8 e4m3, ue8m0 scales | Weight quantization |
| **weight_block_size** | [128, 128] | FP8 block quantization |
| **hc_mult / hc_sinkhorn_iters** | 4 / 20 | Hyper-Connection config |
| **num_nextn_predict_layers** | 1 | Next-token prediction head |

**Key takeaway**: Flash is Pro with fewer layers, fewer heads, smaller hidden size, and fewer experts — but the **attention mechanism is identical** (same head_dim, same indexer config, same alternating C4A/C128A pattern in the middle layers, same window size).

---

## 2. Layer Type Distribution

Both models use three attention types, assigned per-layer via `compress_ratios`:

| Type | compress_ratio | What it does |
|------|---------------|-------------|
| **SWA** | 0 | Sliding window attention only — local context |
| **C4A** | 4 | Compress 4:1 + indexer sparse selection + window |
| **C128A** | 128 | Compress 128:1 + window (no indexer) |

### Layer Counts

| Layer Type | V4-Pro (61 layers) | V4-Flash (43 layers) |
|-----------|-------------------|---------------------|
| **SWA** | 0 main + 1 nextn = 1 | 2 main + 1 nextn = 3 |
| **C4A** | 30 | 21 |
| **C128A** | 31 | 20 |

### Layer Pattern

**V4-Pro**: Starts with C128A, alternates C4A/C128A, ends with SWA (nextn only)
```
Layer  0: C128A          ← no SWA at the start!
Layer  1: C128A
Layer  2: C4A
Layer  3: C128A
Layer  4: C4A
  ...                    ← alternating C4A / C128A
Layer 59: C128A
Layer 60: C4A
Layer 61: SWA (nextn)    ← only SWA layer
```

**V4-Flash**: Starts with 2 SWA layers, then alternates C4A/C128A, ends with SWA (nextn)
```
Layer  0: SWA            ← 2 pure window layers at the start
Layer  1: SWA
Layer  2: C4A
Layer  3: C128A
Layer  4: C4A
  ...                    ← alternating C4A / C128A
Layer 41: C128A
Layer 42: C4A
Layer 43: SWA (nextn)
```

**Design rationale**:
- **Pro has no early SWA** — it jumps straight into compressed attention (C128A). This makes sense for a larger model that can learn good compression from layer 0.
- **Flash has 2 early SWA layers** — the smaller model benefits from a few layers of raw local attention before the compression layers kick in. Think of it as a "warming up" phase where the model builds initial representations before compressing.
- **Both end with SWA** — the nextn prediction layer uses simple window attention.
- **Alternating C4A/C128A** — every other layer sees fine-grained (4:1) vs coarse (128:1) compressed context. This gives the model both detailed and global views at every pair of layers.

---

## 3. KV Cache Architecture

### What Each Layer Type Caches

Each cached KV entry is a **512-element bf16 vector** = **1024 bytes**.

(V4 only caches the 512 nope dims, NOT the 64 rope dims — rope is recomputed on the fly. This saves 64×2 = 128 bytes/entry compared to V3.2 which cached 512+64 = 576 elements.)

| Layer Type | Cache Components | Slots per Layer | What's Stored |
|-----------|-----------------|----------------|--------------|
| **SWA** | Window buffer | `window_size` = 128 | Circular buffer of recent KV vectors |
| **C4A** | Window + compressed + indexer | 128 + seq/4 + seq/4 (indexer) | Window buffer + compressed KV pool + smaller indexer KV for scoring |
| **C128A** | Window + compressed | 128 + seq/128 | Window buffer + heavily compressed KV pool |

### Bytes Per Entry

| Cache Type | Elements | Dtype | Bytes/Entry |
|-----------|----------|-------|------------|
| Main KV (shared) | 512 | bf16 | **1024** |
| Indexer KV (C4A only) | 128 | bf16 | **256** |

### Why the Short Window (128)?

The `sliding_window = 128` is surprisingly small. It exists to solve a **causal gap problem** with C128A:

With C128A (128:1 compression), the first compressed token summarizes positions 0–127. A query at position 100 **cannot attend to this compressed token** (it would see information from positions 101–127 that it shouldn't, violating causality). The 128-slot window ensures the query can still access raw positions 0–100 via uncompressed attention.

For C4A (4:1), the gap is only 3 positions, but the same window covers it.

---

## 4. KV Cache at 1M Tokens: Concrete Numbers

### Setup
- Sequence length: 1,048,576 tokens (1M = 1024 × 1024)
- Dtype: bf16 (2 bytes per element)
- Main KV entry: 512 × 2 = 1024 bytes
- Indexer KV entry: 128 × 2 = 256 bytes

### V4-Pro (61 layers)

| Layer Type | Count | Slots/Layer | Bytes/Slot | Subtotal |
|-----------|-------|-------------|-----------|---------|
| SWA (main) | 0 | 128 | 1024 | **0 GiB** |
| C4A main KV | 30 | 128 + 262,144 = 262,272 | 1024 | **7.50 GiB** |
| C4A indexer KV | 30 | 262,144 | 256 | **1.88 GiB** |
| C128A | 31 | 128 + 8,192 = 8,320 | 1024 | **0.25 GiB** |
| **Total** | | | | **~9.62 GiB** |

Per-layer breakdown (matches vLLM blog):
- C4A layer: (128 + 262,144) × 1024 + 262,144 × 256 = **~320.1 MiB**
- C128A layer: (128 + 8,192) × 1024 = **~8.1 MiB**

Naive baseline (no compression, 61 layers × 1M × 1024 B): **~61.0 GiB**
→ **6.3× savings**

### V4-Flash (43 layers)

| Layer Type | Count | Slots/Layer | Bytes/Slot | Subtotal |
|-----------|-------|-------------|-----------|---------|
| SWA (main) | 2 | 128 | 1024 | **0.00 GiB** |
| C4A main KV | 21 | 128 + 262,144 = 262,272 | 1024 | **5.25 GiB** |
| C4A indexer KV | 21 | 262,144 | 256 | **1.31 GiB** |
| C128A | 20 | 128 + 8,192 = 8,320 | 1024 | **0.16 GiB** |
| **Total** | | | | **~6.73 GiB** |

Naive baseline (no compression, 43 layers × 1M × 1024 B): **~43.0 GiB**
→ **6.4× savings**

### Where the Budget Goes (Visual Breakdown)

```
V4-Pro KV Cache at 1M tokens (~9.62 GiB total)
══════════════════════════════════════════════════

C4A main KV ████████████████████████████████████ 78.0%  (7.50 GiB)
C4A indexer ██████████            19.5%  (1.88 GiB)
C128A       ██                     2.5%  (0.24 GiB)
SWA         ▏                      0.0%  (0.00 GiB)

V4-Flash KV Cache at 1M tokens (~6.73 GiB total)
══════════════════════════════════════════════════

C4A main KV ████████████████████████████████████ 78.1%  (5.25 GiB)
C4A indexer ██████████            19.5%  (1.31 GiB)
C128A       ██                     2.4%  (0.16 GiB)
SWA         ▏                      0.0%  (0.00 GiB)
```

**Key observations**:
- **C4A dominates** (~97.5% of cache) — the 4:1 compression is the workhorse
- **C128A is nearly free** — 128:1 compression makes these layers very cache-efficient
- **SWA is negligible** — only 128 slots per layer
- **The indexer adds ~20% overhead** on top of C4A main KV — this is the cost of learned sparse selection
- **Both models have the same 78/20/2 split** — the architecture scales proportionally

### Compared to V3.2

| Model | KV Cache at 1M | Per-token per-layer |
|-------|---------------|-------------------|
| V3.2 (61 layers) | ~83.9 GiB | 1152 B (512+64 elements × 2) + 256 B indexer |
| V4-Pro (61 layers) | ~9.62 GiB | Varies by layer type |
| V4-Flash (43 layers) | ~6.73 GiB | Varies by layer type |

V4-Pro achieves **~8.7× reduction** vs V3.2 (matching the vLLM blog), primarily through:
1. **Dropping rope dims from cache** (576 → 512 elements, recompute rope on the fly)
2. **Compressed attention** (not all layers store full sequence)
3. **Two compression ratios** (C128A layers are nearly free)

---

## 5. Cache Lifecycle: 1M Token Example

Walk through what happens as a 1M-token sequence is processed.

### Phase 1: Prefill (tokens 0 → 1,048,575)

Processed in chunks (e.g., 256 tokens at a time). For each chunk:

```
Chunk [0:256] arrives
  ├─ SWA layers: write 128 most recent tokens to window buffer
  ├─ C4A layers:
  │   ├─ Window: write 128 most recent to circular buffer
  │   ├─ Compressor: compress 256 tokens → 64 compressed (256/4)
  │   │   └─ Write 64 entries to compressed KV cache at positions [0:64]
  │   └─ Indexer compressor: compress → write to indexer cache
  └─ C128A layers:
      ├─ Window: write 128 most recent
      └─ Compressor: compress 256 → 2 entries (256/128)
          └─ Write 2 entries at positions [0:2]
```

After all 4096 chunks of 256 tokens:

```
Final cache state per layer type:
  SWA:   window[128] filled with tokens [1048448:1048575]
  C4A:   window[128] + compressed[262144] + indexer[262144]
  C128A: window[128] + compressed[8192]
```

### Phase 2: Decode (token 1,048,576, 1,048,577, ...)

Each new token:

```
Token at position P arrives
  ├─ SWA layer:
  │   └─ Write KV to window[P % 128], attend to window only
  │
  ├─ C4A layer:
  │   ├─ Write KV to window[P % 128]
  │   ├─ Compressor accumulates; every 4th token fires:
  │   │   └─ Compress 4 accumulated → 1 entry, append to compressed cache
  │   ├─ Indexer scores all compressed positions, selects top-512
  │   └─ sparse_attn over window[128] + selected[≤512] positions
  │
  └─ C128A layer:
      ├─ Write KV to window[P % 128]
      ├─ Compressor accumulates; every 128th token fires:
      │   └─ Compress 128 → 1 entry, append to compressed cache
      └─ sparse_attn over window[128] + all compressed positions

Total KV reads per decode step:
  SWA:   128 entries
  C4A:   128 + 512 = 640 entries (window + selected)
  C128A: 128 + 8192+ = ~8320 entries (window + all compressed)
```

### Cache Growth Over Time

```
Position    SWA slots   C4A compressed   C128A compressed
      0         1              0                 0
    128       128             32                 1
  1,024       128            256                 8
  8,192       128          2,048                64
 65,536       128         16,384               512
262,144       128         65,536             2,048
1,048,576     128        262,144             8,192   ← 1M tokens
```

C4A compressed slots grow linearly at rate 1/4, C128A at rate 1/128. The window stays fixed at 128.

---

## Summary

V4-Pro and V4-Flash are the **same architecture at different scales**. The attention mechanism — MLA with C4A/C128A compressed attention, learned indexer, and sparse gather-based computation — is identical. Flash trades ~30% fewer layers, ~43% smaller hidden dim, and ~33% fewer experts for proportionally faster inference, while keeping the same 1M context capability and similar 6× KV cache compression ratio.

### Ref
- https://vllm.ai/blog/deepseek-v4
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json
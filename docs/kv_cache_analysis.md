# DeepSeek V4 KV Cache Reduction: First Principles Analysis

## The Problem

Transformer attention is O(S²) in memory and compute. For long sequences (128K+ tokens), storing and attending to all KV pairs is prohibitive. DeepSeek V4 solves this with **three orthogonal KV reduction methods**, combined differently per layer.

## Three Primitives

### 1. Sliding Window (SWA)

**What**: Only attend to the most recent `window_size` tokens (128 in V4).

**Why it works**: Most language tasks have strong locality — a token's meaning depends heavily on its immediate neighbors. The window acts as a "working memory" for local context.

**Tradeoff**: Loses all information beyond the window. Token at position 1000 cannot see token at position 0.

**Implementation**: Circular buffer in `kv_cache[:, :window_size]`. During decode, new KV overwrites `kv_cache[:, pos % window_size]`. Index generation via `get_window_topk_idxs()`.

```
Position:  0  1  2  3  ... 126 127 | 128 129 130 ...
Window:                  [───────128 tokens───────]
           ↑ dropped                  ↑ new token overwrites slot 0
```

### 2. Token Compression

**What**: Learn to compress `ratio` consecutive tokens into 1 compressed KV vector via gated pooling.

**Why it works**: Not every token carries equal information. A learned gating mechanism (softmax over scores + weighted sum) distills the essential information from a group of tokens into a single representative vector. This is like a learned "summary" operation.

**Tradeoff**: Lossy — the compressed vector cannot perfectly reconstruct all `ratio` original tokens. Higher ratio = more loss but smaller cache.

**Implementation**: The `Compressor` module:
```
tokens [ratio tokens, D] → wkv projection → wgate scoring → softmax-weighted sum → [1, D]
                                                    ↑
                                              APE (absolute positional encoding within group)
```

Two variants:
- **ratio=4 (overlap=True)**: Overlapping windows for smoother boundaries. `kv_state` holds both current window and overlap from previous window.
- **ratio=128 (overlap=False)**: Simple non-overlapping pooling. Entire 128-token blocks compressed to 1 vector.

### 3. Indexer (Learned Sparse Selection)

**What**: Instead of attending to ALL compressed positions, learn to select the top-k most relevant ones per query.

**Why it works**: Even among compressed positions, most are irrelevant to any given query. The indexer scores each compressed position against each query and picks the best ones — like a learned retrieval system.

**Tradeoff**: Adds compute (scoring pass) but saves much more in attention compute. Only works when there are enough compressed positions to be selective about (hence only used with ratio=4, not ratio=128).

**Implementation**: The `Indexer` module:
```
Q (from main attention's q_lora) → wq_b → RoPE → Hadamard → FP4
Compressed KV (from Indexer's own Compressor) → [n_compressed, 128]
Score = ReLU(Q @ KV^T) * importance_weights → topk selection
```

---

## Three Layer Configurations

DeepSeek V4 combines these three primitives into three layer types. The key insight is that **each combination serves a different attention range**:

### a. SWA-only: `ratio=0` → just sliding window

```
compress_ratio = 0
Components: sliding window only
Layers: 0, 1, 7 (first, second, and last layers)
```

**What each query sees**:
```
         [────── window_size=128 ──────]
... t₉₃  t₉₄  t₉₅  t₉₆ ... t₁₂₇  Q    ← attends only to recent 128 tokens
         ↑                            ↑
      window start               current position
```

**KV cache**: `[B, 128, 512]` — circular window buffer only.

**Index construction** (line 507):
```python
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
# → [B, S, 128]   indices into window positions
# No compression, no indexer — that's it.
```

**Why these layers**: Early layers (0, 1) focus on local patterns (syntax, short phrases). Last layer (7) needs precise local context for next-token prediction. Global context comes from the middle layers.

**RoPE**: Uses base `rope_theta=10000` with no YaRN scaling (line 479). Pure local attention doesn't need extended-context rope tricks.

---

### b. C4A: `ratio=4` → sliding window + indexer + compression

```
compress_ratio = 4
Components: sliding window + Compressor(overlap) + Lightning Indexer
Layers: 2, 4, 6 (middle layers, odd-indexed in compress_ratios)
```

**What each query sees**:
```
         [────── window=128 ──────]  [── selected compressed positions ──]
... t₉₆  t₉₇ ... t₁₂₇  Q           c₃  c₇  c₁₅  c₂₂  ...  c₃₀
         ↑              ↑             ↑                          ↑
     window start   current      Indexer picks top-k from all compressed
```

**KV cache**: `[B, 128 + max_seq_len//4, 512]` = `[B, 1152, 512]`
- Slots `[0:128]` = circular window buffer (raw tokens)
- Slots `[128:]` = compressed positions (4 tokens → 1)

**Index construction** (lines 507-514):
```python
# Step 1: Window indices
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)  # [B, S, 128]

# Step 2: Indexer selects from compressed positions
compress_topk_idxs = self.indexer(x, qr, start_pos, offset)     # [B, S, topk]
# Indexer uses its own Compressor + scoring + topk

# Step 3: Concatenate
topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)  # [B, S, 128+topk]
```

**The Indexer's role**: With 4:1 compression, a 4096-token sequence produces 1024 compressed positions. Attending to all 1024 + 128 window = 1152 is still expensive. The Indexer selects the most relevant ~32-512 compressed positions per query, keeping attention sparse.

**Why overlap**: With ratio=4, boundaries between compression groups matter. Overlapping windows let each compressed position incorporate information from the adjacent group, smoothing the boundary effect.

**RoPE**: Uses `compress_rope_theta=40000` with YaRN scaling (line 476). Compressed positions represent longer-range context, needing extended-range positional encoding.

---

### c. C128A: `ratio=128` → sliding window + compression (no indexer)

```
compress_ratio = 128
Components: sliding window + Compressor(no overlap)
Layers: 3, 5 (middle layers, even-indexed in compress_ratios)
```

**What each query sees**:
```
         [────── window=128 ──────]  [all compressed]
... t₉₆  t₉₇ ... t₁₂₇  Q           c₀  (c₁  c₂ ...)
         ↑              ↑             ↑
     window start   current      ALL compressed positions attended to
                                 (so few that no selection needed)
```

**KV cache**: `[B, 128 + max_seq_len//128, 512]` = `[B, 160, 512]`
- Slots `[0:128]` = circular window buffer
- Slots `[128:]` = compressed positions (128 tokens → 1)

**Index construction** (lines 507-514):
```python
# Step 1: Window indices
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)      # [B, S, 128]

# Step 2: Static indices — ALL compressed positions (no selection needed)
compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)  # [B, S, n_compressed]
# For 128-token prefill: only 1 compressed position → [B, S, 1]
# For 4096-token context: 32 compressed positions → [B, S, 32]

# Step 3: Concatenate
topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)      # [B, S, 128+n_compressed]
```

**Why no indexer**: At 128:1 compression, a 4096-token sequence has only 32 compressed positions. Attending to all 32 is cheap — no selection needed. The indexer's overhead (scoring, topk) isn't worth it for so few candidates.

**Why no overlap**: At 128:1, each compressed position summarizes a large block. Overlap would require maintaining 2×128=256 tokens of state, doubling memory for marginal quality gain on such coarse compression.

**RoPE**: Same as C4A — `compress_rope_theta=40000` with YaRN.

---

## Why This Combination?

The three layer types form a **multi-scale attention hierarchy**:

```
                    ┌─────────────────────────────────────────────┐
                    │          Token position timeline            │
                    │  0 ─────────────────────────────── 4096     │
                    └─────────────────────────────────────────────┘
                                                            ▲
SWA-only (L0,1,7):  .............................  [════128════]
                                                   local detail

C4A (L2,4,6):       .............................  [════128════]
                     [c₃][c₇][c₁₅]......[c₂₂]  ← selected by Indexer
                     fine-grained compressed (4:1), sparse selection

C128A (L3,5):        .............................  [════128════]
                     [C₀]         [C₁]          ← ALL attended
                     coarse compressed (128:1), attend to everything
```

| Range | Method | Resolution | Selection |
|-------|--------|-----------|-----------|
| **Local** (last 128 tokens) | Sliding window | Full (1:1) | All positions |
| **Medium** (all context, 4:1) | Compression + Indexer | Medium (4:1) | Learned top-k |
| **Global** (all context, 128:1) | Compression only | Coarse (128:1) | All positions |

This is analogous to **image pyramids** in computer vision: you look at nearby things in high resolution, and distant things in progressively lower resolution. The Indexer acts like a "saliency detector" for the medium-resolution level.

---

## Decode Behavior

During decode (one token at a time), the three methods behave differently:

| Method | Every step | Periodic | Never |
|--------|-----------|----------|-------|
| Window write | ✓ (always writes to circular buffer) | | |
| C4A compress | | Every 4 steps (pos%4==3) | |
| C128A compress | | Every 128 steps (pos%128==127) | |
| C4A indexer | ✓ (always scores & selects) | | |

```
Decode step:  128  129  130  131  132  133  134  135  ...  255  256
Window write:  ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓   ...   ✓    ✓
C4A compress:            ✓              ✓              ...        ✓
C128A compress:                                        ...   ✓
C4A indexer:   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓   ...   ✓    ✓
```

When compression doesn't fire (most decode steps), attention uses existing compressed positions from cache. The compressor accumulates state in `kv_state`/`score_state` buffers until `ratio` tokens are collected, then fires once.

---

## Memory Analysis

For a 4096-token sequence, KV cache per layer:

| Layer type | Window | Compressed | Total KV entries | vs. full attention |
|------------|--------|------------|------------------|--------------------|
| SWA-only | 128 | 0 | 128 | **32× reduction** |
| C4A | 128 | up to 1024 (selected ~32-512) | 128 + 1024 | **3.5× reduction** |
| C128A | 128 | 32 | 160 | **25× reduction** |
| Full attention | 4096 | 0 | 4096 | baseline |

Combined with MLA (single 512-dim KV per position instead of 64×512), the total KV cache is:
- MLA: **64× reduction** (shared KV across heads)
- Sparse: **3.5-32× reduction** (depending on layer type)
- Combined: **~224-2048× reduction** vs standard MHA with full attention

---

## Code Evidence

From `Attention.__init__` (model.py:466-471):
```python
if self.compress_ratio:                              # ratio > 0
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    if self.compress_ratio == 4:                     # C4A only
        self.indexer = Indexer(args, self.compress_ratio)
    else:
        self.indexer = None                          # C128A: no indexer
```

From `Attention.forward` (model.py:507-514):
```python
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)    # always: window
if self.compress_ratio:                                             # if ratio > 0
    offset = kv.size(1) if start_pos == 0 else win
    if self.indexer is not None:                                    # C4A: learned selection
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:                                                           # C128A: all compressed positions
        compress_topk_idxs = get_compress_topk_idxs(ratio, ...)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

From `ModelArgs` (model.py:65):
```python
compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
#                               a  a  b   c   b   c   b  a
```

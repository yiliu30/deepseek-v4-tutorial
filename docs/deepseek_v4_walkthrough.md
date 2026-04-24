# DeepSeek V4 Architecture Walkthrough

Complete tensor-shape trace of the DeepSeek V4 reference model (`model.py`) using a tiny config (dim=4096, 7 layers, 5.6B params). Exercises all V4-specific components.

## Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| dim | 4096 | Hidden dimension |
| n_layers | 7 | Transformer blocks |
| n_heads | 64 | Attention heads |
| head_dim | 512 | KV dimension (shared across heads via MLA) |
| rope_head_dim | 64 | Rotary positional encoding dimension |
| q_lora_rank | 1024 | Q low-rank bottleneck |
| o_lora_rank | 1024 | O low-rank bottleneck |
| o_groups | 8 | Grouped O projection (8 heads per group) |
| hc_mult | 4 | Hyper-Connection multiplicity |
| window_size | 128 | Sliding window size |
| n_routed_experts | 8 | MoE routed experts |
| n_activated_experts | 2 | Top-k experts per token |
| index_topk | 512 | Lightning Indexer top-k |
| index_head_dim | 128 | Indexer head dimension |
| compress_ratios | (0,0,4,128,4,128,4) | Per-layer compression |

## Layer Type Map

| Layer | Ratio | Type | Components |
|-------|-------|------|------------|
| 0, 1 | 0 | SWA-only | Sliding window attention only |
| 2, 4, 6 | 4 | C4A | Compressor (overlap) + Lightning Indexer + sparse attn |
| 3, 5 | 128 | C128A | Compressor (no overlap) + static indices + sparse attn |

---

## Architecture Overview

```
input_ids [B, S]
    │
    ▼
Embedding → [B, S, 4096]
    │
    ▼  HC expand (repeat ×4)
[B, S, 4, 4096]     ◄── hc_mult copies of hidden state
    │
    ├──── Block 0 (SWA-only) ────┐
    ├──── Block 1 (SWA-only) ────┤
    ├──── Block 2 (C4A) ─────────┤  Each block: HC_pre → Attn → HC_post → HC_pre → MoE → HC_post
    ├──── Block 3 (C128A) ───────┤
    ├──── Block 4 (C4A) ─────────┤
    ├──── Block 5 (C128A) ───────┤
    ├──── Block 6 (C4A) ─────────┘
    │
    ▼  HC head (reduce 4 copies → 1)
[B, S, 4096]
    │
    ▼  RMSNorm → Linear
Logits [B, 129280]
```

---

## Component-by-Component Data Flow

### 1. Hyper-Connections (HC)

Replaces the standard residual connection. Instead of `x + sublayer(x)`, maintains 4 parallel copies and uses learned Sinkhorn-normalized matrices to mix them.

**HC Pre** (reduce 4 → 1 for sublayer input):
```
x: [B, S, 4, 4096]                    ◄── 4 copies of hidden state
    │
    ▼  flatten
x_flat: [B, S, 16384]                 ◄── hc*D = 4*4096
    │
    ▼  RMS normalization
rsqrt: [B, S, 1]
    │
    ▼  mixes = x_flat @ hc_fn^T * rsqrt
mixes: [B, S, 24]                     ◄── mix_hc = (2+hc)*hc = 24
    │                                      hc_fn: [24, 16384]
    ▼  hc_split_sinkhorn
pre:  [B, S, 4]                        ◄── sigmoid + eps
post: [B, S, 4]                        ◄── 2 * sigmoid
comb: [B, S, 4, 4]                     ◄── softmax + 20 Sinkhorn iterations → doubly stochastic
    │
    ▼  y = sum(pre * x, dim=hc)
y: [B, S, 4096]                        ◄── weighted combination of 4 copies → single state
```

**HC Post** (expand 1 → 4 after sublayer):
```
sublayer_out: [B, S, 4096]
residual:     [B, S, 4, 4096]
    │
    ▼  post * sublayer_out + comb @ residual
output: [B, S, 4, 4096]               ◄── back to 4 copies
```

The Sinkhorn normalization ensures `comb` is doubly-stochastic (rows and columns each sum to 1), which preserves information flow across the 4 copies without collapse or explosion.

---

### 2. Multi-head Latent Attention (MLA)

Key innovation: Q and KV both use low-rank projections, and KV is shared across all heads (single 512-dim vector per token position).

#### Q Projection (low-rank bottleneck)
```
x: [B, S, 4096]
    │
    ▼  wq_a (4096 → 1024)
qr: [B, S, 1024]                      ◄── q_lora_rank bottleneck (also reused by Indexer)
    │
    ▼  q_norm (RMSNorm)
q: [B, S, 1024]
    │
    ▼  wq_b (1024 → 64*512 = 32768)
q: [B, S, 64, 512]                    ◄── unflatten into n_heads × head_dim
    │
    ▼  per-head RMS normalization
q: [B, S, 64, 512]
    │
    ▼  RoPE on q[..., -64:]           ◄── only last 64 dims get positional encoding
q: [B, S, 64, 512]                    ◄── head_dim = 448 (nope) + 64 (rope)
```

#### KV Projection (single shared latent)
```
x: [B, S, 4096]
    │
    ▼  wkv (4096 → 512)
kv: [B, S, 512]                       ◄── ONE vector per position (shared across 64 heads!)
    │
    ▼  kv_norm (RMSNorm)
kv: [B, S, 512]
    │
    ▼  RoPE on kv[..., -64:]          ◄── last 64 dims = rope_head_dim
kv: [B, S, 512]
    │
    ▼  FP8-sim on kv[..., :-64]       ◄── QAT: nope dims get FP8 quantization noise
kv: [B, S, 512]
```

Why this works: Q has per-head projections `[B,S,64,512]`, but KV is just `[B,S,512]`. During attention, each head's Q attends to the same shared KV. This is the "Multi-head Latent Attention" trick — KV cache is 64× smaller than standard MHA.

#### O Projection (grouped low-rank)
```
attn_out: [B, S, 64, 512]
    │
    ▼  reshape to groups
o: [B, S, 8, 4096]                    ◄── 8 groups, each with 8 heads × 512 = 4096 dims
    │
    ▼  wo_a: einsum("bsgd,grd->bsgr")
o: [B, S, 8, 1024]                    ◄── wo_a: [8, 1024, 4096] per-group low-rank
    │
    ▼  flatten → wo_b (8192 → 4096)
x: [B, S, 4096]                       ◄── RowParallelLinear (all_reduce in TP)
```

---

### 3. Sparse Attention

All attention in V4 is sparse — uses `topk_idxs` to gather relevant KV positions rather than attending to all positions.

#### SWA-only layers (ratio=0)
```
topk_idxs: [B, S, 128]                ◄── window_size=128 most recent positions
kv:        [B, S, 512]                ◄── S ≤ 128 for prefill in this config
    │
    ▼  sparse_attn(q, kv, attn_sink, topk_idxs, scale)
    │   - gather KV by indices
    │   - Q[B,S,64,512] @ KV_gathered[B,S,128,512]^T → scores[B,S,64,128]
    │   - add attn_sink[64] as virtual token (learnable bias)
    │   - softmax → weighted sum
    │
output: [B, S, 64, 512]
```

#### C4A layers (ratio=4, with Indexer)
```
window_idxs:   [B, S, 128]            ◄── sliding window
indexer_idxs:  [B, S, 32]             ◄── Lightning Indexer selects from compressed positions
combined_idxs: [B, S, 160]            ◄── concatenated

kv (prefill):  [B, 160, 512]          ◄── 128 raw + 32 compressed positions
    │
    ▼  sparse_attn
output: [B, S, 64, 512]
```

#### C128A layers (ratio=128, static indices)
```
window_idxs:     [B, S, 128]
compress_idxs:   [B, S, 1]            ◄── only 1 compressed position (128 tokens → 1)
combined_idxs:   [B, S, 129]

kv (prefill):    [B, 129, 512]        ◄── 128 raw + 1 compressed
    │
    ▼  sparse_attn
output: [B, S, 64, 512]
```

---

### 4. Compressor

Learned gated pooling that compresses `ratio` consecutive tokens into 1 compressed KV vector.

#### C4A Compressor (ratio=4, overlap=True)
```
x: [B, 128, 4096]
    │
    ├─▶ wkv (4096 → 1024)             ◄── 2× head_dim because overlap=True (512 overlap + 512 normal)
    │   kv: [B, 128, 1024]
    │
    ├─▶ wgate (4096 → 1024)
    │   score: [B, 128, 1024]
    │
    ▼  unflatten into groups of 4
kv:    [B, 32, 4, 1024]
score: [B, 32, 4, 1024] + APE         ◄── Absolute Positional Encoding [4, 1024]
    │
    ▼  overlap_transform               ◄── creates overlapping windows for smooth boundaries
kv:    [B, 32, 8, 512]                ◄── 4 overlap + 4 normal slots, each 512-dim
score: [B, 32, 8, 512]
    │
    ▼  softmax(score, dim=2) * kv → sum(dim=2)
kv: [B, 32, 512]                      ◄── 32 compressed positions
    │
    ▼  RMSNorm → RoPE on [..., -64:] → FP8-sim on nope dims
kv: [B, 32, 512]                      ◄── written to kv_cache[:, win:]
```

#### C128A Compressor (ratio=128, no overlap)
```
x: [B, 128, 4096]
    │
    ├─▶ wkv (4096 → 512)              ◄── no overlap → 1× head_dim
    ├─▶ wgate (4096 → 512)
    │
    ▼  unflatten into groups of 128
kv:    [B, 1, 128, 512]
score: [B, 1, 128, 512] + APE[128, 512]
    │
    ▼  softmax(score, dim=2) * kv → sum(dim=2)
kv: [B, 1, 512]                       ◄── entire 128-token sequence → 1 vector
    │
    ▼  RMSNorm → RoPE → FP8-sim
kv: [B, 1, 512]                       ◄── written to kv_cache
```

#### Decode behavior
During decode, compressor accumulates tokens in state buffers. C4A fires every 4 tokens (`pos%4==3`). C128A fires every 128 tokens. When not firing, returns `None` and attention uses only the existing cached compressed positions.

---

### 5. Lightning Indexer

Selects which compressed KV positions to attend to. Only present in C4A layers (ratio=4).

```
x: [B, S, 4096]                       ◄── raw input (same as attention input)
qr: [B, S, 1024]                      ◄── q_lora_rank output (reused from main Q projection)
    │
    ├─▶ wq_b (1024 → 64*128 = 8192)
    │   q: [B, S, 64, 128]            ◄── indexer uses smaller head_dim=128 (vs 512 for main attn)
    │   │
    │   ▼  RoPE → Hadamard rotation → FP4-sim
    │   q: [B, S, 64, 128]
    │
    ├─▶ Indexer's own Compressor (ratio=4, head_dim=128, rotate=True)
    │   │  Uses Hadamard rotation + FP4 quantization
    │   kv: [B, 32, 128]              ◄── 32 compressed positions at 128-dim
    │
    ├─▶ weights_proj (4096 → 64)
    │   weights: [B, S, 64]            ◄── per-head importance weights
    │
    ▼  Scoring:
index_score = einsum("bshd,btd->bsht", q, kv)  ◄── [B, S, 64, 32]
index_score = (relu(index_score) * weights).sum(dim=heads)  ◄── [B, S, 32]
    │
    ▼  topk selection
topk_idxs: [B, S, 32]                 ◄── min(512, 32) = 32 compressed positions selected
```

The Indexer is lightweight: it uses 128-dim heads (vs 512 for main attention), FP4 quantization with Hadamard rotation for further compression, and ReLU gating to select relevant compressed chunks.

---

### 6. MoE (Mixture of Experts)

```
x: [B, S, 4096]
    │
    ▼  flatten to [B*S, 4096]
x: [256, 4096]                         ◄── all tokens processed independently
    │
    ▼  Gate (sqrtsoftplus scoring)
    │   scores = sqrt(softplus(x @ gate_weight^T))    ◄── gate_weight: [8, 4096]
    │   scores += bias                                  ◄── bias shifts for selection only
    │   indices = topk(scores, k=2)                    ◄── top-2 experts per token
    │
weights: [256, 2]                      ◄── normalized routing weights
indices: [256, 2]                      ◄── which 2 of 8 experts
    │
    ├─▶ For each activated expert:
    │   Expert(SwiGLU):
    │     w1(4096 → 4096) → SiLU
    │     w3(4096 → 4096) → optional clamp
    │     gate = SiLU(w1(x)) * w3(x)
    │     w2(4096 → 4096)
    │   y[idx] += expert(x[idx]) * weight
    │
    ├─▶ Shared expert (always active):
    │   Same SwiGLU architecture, no weight scaling
    │   y += shared_expert(x)
    │
output: [B, S, 4096]
```

---

## Prefill vs Decode Shape Comparison

| Component | Prefill (B=2, S=128) | Decode (B=2, S=1, pos=128) |
|-----------|---------------------|---------------------------|
| HC input | [2, 128, 4, 4096] | [2, 1, 4, 4096] |
| HC mixes | [2, 128, 24] | [2, 1, 24] |
| Q | [2, 128, 64, 512] | [2, 1, 64, 512] |
| KV (raw) | [2, 128, 512] | [2, 1, 512] |
| Window indices | [2, 128, 128] | [2, 1, 128] |
| **C4A KV (prefill)** | **[2, 160, 512]** (128+32) | reads from kv_cache [2, 1152, 512] |
| **C128A KV (prefill)** | **[2, 129, 512]** (128+1) | reads from kv_cache [2, 160, 512] |
| C4A Indexer idxs | [2, 128, 32] | [2, 1, 32] |
| C4A combined idxs | [2, 128, 160] | [2, 1, 160] |
| C128A static idxs | [2, 128, 1] | [2, 1, 1] |
| Attn output | [2, 128, 64, 512] | [2, 1, 64, 512] |
| O grouped | [2, 128, 8, 4096] | [2, 1, 8, 4096] |
| MoE flat | [256, 4096] | [2, 4096] |
| Gate weights | [256, 2] | [2, 2] |
| Logits | [2, 129280] | [2, 129280] |

---

## KV Cache Layout

Each attention layer allocates: `[max_batch, window_size + max_seq_len//ratio, head_dim]`

| Layer type | Cache shape | Layout |
|------------|-------------|--------|
| SWA-only | [4, 128, 512] | Circular window buffer (128 slots) |
| C4A | [4, 1152, 512] | [128 window \| 1024 compressed] (4096/4=1024 max compressed) |
| C128A | [4, 160, 512] | [128 window \| 32 compressed] (4096/128=32 max compressed) |

The Indexer has its own separate cache: `[4, 1024, 128]` (head_dim=128 for indexing).

---

## Key Design Insights

1. **MLA = single KV per position**: KV is just 512-dim (not 64×512), giving 64× KV cache savings vs standard MHA. The "multi-head" part happens only in Q (which has per-head projections).

2. **Compression hierarchy**: C4A (fine-grained, 4:1) + C128A (coarse, 128:1) together provide multi-scale context compression. C4A has overlap for smoother boundaries; C128A does not.

3. **Indexer is cheap**: Uses 128-dim heads (vs 512), FP4 quantization, and Hadamard rotation. It selects *which* compressed positions matter, not how to attend to them.

4. **HC replaces residual**: Instead of `x + f(x)`, maintains 4 copies mixed through Sinkhorn-normalized matrices. This is the "Hyper-Connection" from the paper — more expressive than simple residuals.

5. **Everything is sparse**: Even SWA-only layers use `sparse_attn` with gather-based indexing. No dense attention anywhere.

6. **attn_sink**: Each head has a learned scalar bias added before softmax, acting as a virtual "sink" token that absorbs attention probability. This is V4's version of attention sinks.

---

## Files

| File | Purpose |
|------|---------|
| `model.py` | Reference DeepSeek V4 model (symlinked from official repo) |
| `kernel.py` | Pure-PyTorch stubs replacing tilelang kernels |
| `fast_hadamard_transform.py` | Pure-PyTorch Hadamard transform |
| `walkthrough.py` | This trace script |
| `walkthrough_output.txt` | Full 983-line trace output |
| `run_tiny.py` | Simpler driver (less verbose) |

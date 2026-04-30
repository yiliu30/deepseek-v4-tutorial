# DeepSeek V4 C4A Attention: Core Logic and Flow

This document explains the **C4A (compress-4 attention)** code path in vLLM's
DeepSeek V4 implementation, using the concrete 13-token example from the
[interactive C4A visualization](/assets/interactive_pages/c4a.html) to ground
every concept.

C4A is the attention variant used in ~30 of the 61 transformer layers. It
combines three attention patterns into one layer:

1. **Sliding-window attention (SWA)** over the last 128 uncompressed tokens
2. **Compressed sparse attention** over the top-512 compressed KV entries
3. **An attention sink** bias term (loaded from model weights)

The remaining ~31 layers use C128A (128× compression, no indexer needed because
the compressed pool is small enough for full attention).

---

## The 13-Token Example: What Each Token Produces

Consider a sequence of 13 tokens (positions 0–12). Each token passing through a
C4A layer produces five outputs from linear projections:

```
Token i → hidden_states[i]
  ├─► fused_wqa_wkv ──► (qr, kv)     ← merged linear, split into q-route & shared key-value
  │       ├─► q   (query, after RMSNorm + wq_b)
  │       └─► k(v) (shared key-value latent, after RMSNorm)
  ├─► main compressor.fused_wkv_wgate ──► (Ca, Za)   ← KV & gating score for main cache
  └─► indexer.compressor.fused_wkv_wgate ──► (Cb, Zb) ← KV & gating score for indexer cache
```

In the visualization, these appear as five horizontal rows:

```
Row         Color       What it is
─────────   ─────────   ──────────────────────────────────────────────
k(v)        orange      Shared key-value vectors (SWA targets)
q           blue        Query vectors
C^compress  gradient    Compressed entries (output of compressor)
Ca & Za     purple      Main compressor inputs (per-token)
Cb & Zb     green       Indexer compressor inputs (per-token)
```

---

## Component 1: The Compressor — How Ca & Za Become C^compress

**Files:** `vllm/model_executor/layers/deepseek_compressor.py`

The compressor takes per-token `(Ca, Za)` pairs and produces one compressed
entry every 4 positions, using an **overlapping window of 8 tokens with stride
4**.

### Concrete example with 13 tokens

The visualization places compressed entries at positions 3, 7, and 11:

```
Tokens:    0  1  2  3  4  5  6  7  8  9  10 11 12
           ├──────────┤                              ← C0 reads Ca&Za from [0,1,2,3]
                    ├──────────┤                      ← C1 reads Ca&Za from [4,5,6,7]
                             ├──────────────┤         ← C2 reads Ca&Za from [8,9,10,11]
```

But C4A also uses an **overlapping** contribution (`coff = 2` in code). The
`Cb & Zb` row captures this overlap — each compressed entry also reads from the
**previous 4 tokens**:

```
C0 (pos 3):  Ca&Za from [0,1,2,3]                   ← no prior block to overlap
C1 (pos 7):  Ca&Za from [4,5,6,7] + Cb&Zb from [0,1,2,3]  ← overlaps with C0's block
C2 (pos 11): Ca&Za from [8,9,10,11] + Cb&Zb from [4,5,6,7] ← overlaps with C1's block
```

This is exactly what the visualization shows when you hover over a C^compress
token — the arrows fan out to both `Ca & Za` (current block) and `Cb & Zb`
(previous block).

### Why C1 needs tokens [0,1,2,3] — the information boundary problem

Without overlap, each compressed entry would be a **lossy summary of exactly 4
tokens**. Consider what happens at the boundary between C0 and C1:

```
C0 = compress([0,1,2,3])    C1 = compress([4,5,6,7])
         ┃                           ┃
    Lost: how token 3 ──────── relates to token 4
```

Token 3 and token 4 are adjacent in the original sequence, but after
compression they live in separate compressed entries with no shared context.
Any cross-boundary dependency (e.g., "the subject in token 3 is the antecedent
for the pronoun in token 5") is invisible to the model — C0 doesn't know about
token 4, and C1 doesn't know about token 3.

The **overlapping window** fixes this. By giving C1 access to the *previous*
block's tokens [0,1,2,3], the compressor sees the full local context across
the boundary:

```
C1 = compress([0,1,2,3] ∪ [4,5,6,7])
              ↑ overlap    ↑ current
              (Cb & Zb)    (Ca & Za)
```

The softmax gating mechanism (`score_states`) learns which of these 8 tokens
matter most. For C1, it might upweight tokens 3–5 (the boundary region) and
downweight tokens 0–1 (already well-captured by C0). This means:

- **C0** captures tokens [0,1,2,3] — the start of the sequence
- **C1** captures tokens [4,5,6,7] *in context of* [0,1,2,3] — preserving
  cross-boundary information
- The gating weights are learned, so the model decides how much overlap context
  to use per layer

This is directly visible in the kernel code
(`fused_compress_quant_cache.py` line 87–88):

```python
start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1   # For C1 at pos 7: start = 7 - 2*4 + 1 = 0
tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)   # 8 tokens: [0,1,2,3,4,5,6,7]
```

The `head_offset` on line 99 splits the 8 tokens into two halves within the
state cache: tokens `[0..3]` index into the `Cb & Zb` portion (offset by
`HEAD_SIZE`), while tokens `[4..7]` index into the `Ca & Za` portion. Both
halves are then jointly softmax-gated and weighted-summed (lines 116–129):

```python
score = tl.load(row_base[:, None] + STATE_WIDTH + block[None, :], ...)
score = tl.softmax(score, dim=0)                         # softmax over 8 tokens
kv = tl.load(row_base[:, None] + block[None, :], ...)
compressed_kv = tl.sum(kv * score, axis=0)               # weighted sum → 1 compressed entry
```

> **Why not a larger window?** The window is `coff × compress_ratio = 2 × 4 = 8`
> tokens. Using 3× or more would add compute without much benefit — each
> compressed entry already sees its own 4 tokens plus 4 tokens of boundary
> context. The 2× multiplier is the sweet spot where boundary information is
> preserved without the compression ratio degrading (each C entry still
> summarizes a stride-4 window, just with overlapping context).

### Why two sub-vectors per token: Ca & Za vs Cb & Zb

The naming `Ca & Za` / `Cb & Zb` might suggest two separate compressors, but
they come from **one linear layer** with a doubled output dimension.

The compressor's `fused_wkv_wgate` projects each token's hidden state into
`2 × head_dim` for both kv and score:

```python
# deepseek_compressor.py line 220-222
self.fused_wkv_wgate = MergedColumnParallelLinear(
    hidden_size,
    [coff * head_dim, coff * head_dim],  # coff=2 for C4A → 2×512 each
    ...
)
```

This gives each token **two sub-vectors**: `kv = [kv_a | kv_b]`, each of size
`head_dim`. The compression kernel then uses them asymmetrically:

```
For C1 at position 7, window = [0,1,2,3,4,5,6,7]:

  Overlap tokens [0,1,2,3]:  head_offset = 0          → reads kv_a (first half)
  Current tokens [4,5,6,7]:  head_offset = HEAD_SIZE   → reads kv_b (second half)
```

```python
# fused_compress_quant_cache.py line 99
head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
```

**Why two sub-vectors instead of one?** The overlap tokens and current tokens
play fundamentally different roles:

- **Current tokens** (`kv_b`, shown as `Ca & Za`): These are the "primary"
  content that this compressed entry is responsible for summarizing. The model
  needs a representation optimized for "what happened in positions [4,5,6,7]."

- **Overlap tokens** (`kv_a`, shown as `Cb & Zb`): These provide boundary
  context from the previous block. The model needs a *different* representation
  — one optimized for "how does the previous block's content relate to the
  current block?" rather than "what is the previous block's content."

If both roles shared the same projection, the model would be forced to use one
representation for two purposes. The doubled projection lets the model learn
separate representations: `kv_b` for primary content, `kv_a` for overlap
context. The learned softmax gating (score_a and score_b, same split) then
decides how to weight these 8 contributions.

Both sub-vectors are jointly softmax-gated and weighted-summed into one
compressed entry (lines 116–129):

```python
score = tl.softmax(score, dim=0)          # softmax across all 8 tokens
compressed_kv = tl.sum(kv * score, axis=0) # → one vector of head_dim
```

> **Visualization mapping:** The visualization shows `Ca & Za` and `Cb & Zb`
> as separate rows. These correspond to `kv_b`/`score_b` (current-block half)
> and `kv_a`/`score_a` (overlap half) of the same compressor's output —
> **not** two different compressor modules. The indexer has its own separate
> compressor (`DeepseekCompressor` in `DeepseekV4Indexer`), but that one feeds
> the indexer's 128-dim KV cache, not the main attention's 512-dim KV cache.

### How compression works in code

The compressor maintains a **rolling state cache** per request
(`CompressorStateCache`), treated as a sliding-window buffer of size 8
(`coff * compress_ratio = 2 * 4`).

**Kernel 1: `_save_partial_states_kernel`** (Triton) — runs for every token:
```
state_cache[slot] = {
    kv_state:    W_kv @ hidden_states,
    score_state: W_gate @ hidden_states + APE[position % 4]
}
```
The APE (Absolute Positional Encoding) marks each token's position within the
compression window so the gating knows which tokens are which.

**Kernel 2: `_fused_kv_compress_norm_rope_insert_sparse_attn`** (Triton) — at
compression boundaries (every 4 positions), reads 8 states and produces:
```
C_j = Σ softmax(score_states) · kv_states    ← weighted sum
    → RMSNorm → RoPE(anchor_pos = 4j) → FP8 quant → write to paged KV cache
```

The RoPE anchor position for each compressed entry is shown in the visualization
as `RoPE: 0`, `RoPE: 4`, `RoPE: 8` beneath C0, C1, C2 respectively.

---

## Component 2: The Indexer — Selecting Which Compressed Entries to Attend

**File:** `vllm/model_executor/layers/deepseek_v4_attention.py` (class `DeepseekV4Indexer`)

At 1M context with 4× compression, there are ~250k compressed entries. Attending
to all of them is too expensive. The **indexer** selects the **top-512 most
relevant** entries for each query token.

The indexer is **only present in C4A layers**. C128A layers compress to ~8k
entries, small enough for full attention.

### How it works

The indexer has its own lightweight KV cache (128-dim heads, vs 512-dim for main
MLA) and its own compressor (producing the `Cb & Zb` entries):

```
hidden_states ──► indexer.compressor ──► 128-dim compressed indexer KV cache
qr            ──► indexer.wq_b       ──► indexer query (64 heads × 128 dim)
hidden_states ──► weights_proj       ──► per-head scalar weights (64 values)
```

The selection flow (`SparseAttnIndexer` in `sparse_attn_indexer.py`):

1. `fused_indexer_q_rope_quant`: apply RoPE to query, quantize to FP8/FP4,
   scale by `softmax_scale` and per-head weights
2. Compute dot products between query and indexer KV cache
3. Select **top-512** indices → write into `topk_indices_buffer`

The `topk_indices_buffer` is a single pre-allocated tensor shared across all
C4A layers in the model (allocated once in `DeepseekV4ForCausalLM.__init__`).
Each layer's indexer overwrites it during its forward pass, and the main
attention reads it immediately after.

---

## Component 3: Attention — How q Attends to k(v) and C^compress

**File:** `vllm/model_executor/layers/deepseek_v4_attention.py` (class `DeepseekV4MLAAttention`)

### What the visualization shows when you hover over a query

Hovering over a query token in the visualization reveals three types of arrows:

**1. Green arrows to k(v) — Sliding Window Attention:**
Query q_i attends to all k(v) tokens at positions ≤ i (in the real model,
limited to a window of 128 tokens). The visualization shows all 13 tokens since
they fit within the window.

**2. Green arrows to C^compress — Sparse Compressed Attention:**
Query q_i attends only to compressed entries that satisfy the **causality
constraint**:

```
Causal condition: i >= 4j + 3
```

where j is the compressed entry index. This ensures q_i only sees information
from tokens at positions ≤ i. The concrete pattern in the visualization:

```
q0–q2:   no compressed entries (too early, 4×0+3=3 > 2)
q3–q6:   attend to C0 only
q7–q10:  attend to C0 and C1
q11–q12: attend to C0, C1, and C2
```

**3. Gold arrow to sink — Attention Sink:**
Every query also has an attention sink connection. The sink is a per-head bias
loaded from model weights (`nn.Parameter` with `requires_grad=False`), merged
with the SWA and compressed attention outputs via log-sum-exp.

### How this maps to code

**Decode path** (`_forward_decode`):
```python
flash_mla_with_kvcache(
    q=q,
    k_cache=swa_cache,                          # k(v) sliding window
    indices=swa_indices,                         # which SWA slots to attend
    topk_length=swa_lens,                        # how many SWA tokens per request
    extra_k_cache=kv_cache,                      # compressed main KV cache
    extra_indices_in_kvcache=topk_indices,        # top-512 from indexer
    extra_topk_length=topk_lens,                 # how many valid compressed entries
    attn_sink=self.attn_sink,                    # per-head bias (loaded from weights)
)
```

The FlashMLA kernel fuses all three attention patterns (SWA + compressed sparse +
sink) into a single GPU kernel, merging results via log-sum-exp internally.

**Prefill path** (`_forward_prefill`): processes in chunks of 4 sequences:
1. Gather all compressed KV into a bf16 workspace buffer
2. Gather SWA KV into the same buffer at offset `N = ceil(max_model_len / compress_ratio)`
3. `combine_topk_swa_indices` merges top-k and SWA indices into one tensor
4. `flash_mla_sparse_fwd` runs attention over the combined buffer

---

## Component 4: Inverse RoPE — Why It's Needed After Attention

**File:** `vllm/model_executor/layers/deepseek_v4_attention.py`
(`DeepseekV4MultiHeadLatentAttentionWrapper.forward`)

Because k(v) is **shared** as both key and value (MLA latent attention), the
attention output carries absolute position information through the RoPE rotation
matrices. The math:

```
Standard:  output_i = Σ softmax(q_i · R(j-i) · k_j) · v_j      ← only relative positions
Shared kv: output_i = Σ softmax(q_i · R(j-i) · k_j) · R(j) · k_j  ← absolute R(j) leaks in!
Fix:       R(-i) · output_i = Σ softmax(...) · R(j-i) · k_j      ← relative again
```

In code, this is `fused_inv_rope_fp8_quant` — a fused kernel that applies
`R(-i)` (inverse rotation) and quantizes to FP8 in one pass, feeding the
subsequent `wo_a` batched matmul.

---

## How the Components Work Together: Multi-Stream Execution

**File:** `vllm/model_executor/layers/deepseek_v4_attention.py` (`attention_impl`)

After the initial projection (`fused_wqa_wkv → qr, kv`), three independent
work items need to happen before attention can run. For C4A layers, vLLM
overlaps them across two CUDA streams:

```
                    ┌─── DEFAULT STREAM ───────────────┐  ┌─── AUX STREAM ──────────────────────┐
                    │                                  │  │                                      │
hidden_states ──────┤  Indexer:                         │  │  SWA KV insert:                      │
qr ─────────────────┤  ├─ indexer.compressor(hidden)    │  │  ├─ _fused_qnorm_rope_kv_insert      │
                    │  │  └─ compress Cb,Zb → indexer  │  │  │  └─ Q: per-head RMSNorm + RoPE     │
                    │  │     KV cache (128-dim)         │  │  │     KV: RoPE + FP8 quant + insert  │
                    │  ├─ wq_b(qr) → indexer query     │  │  │                                    │
                    │  ├─ weights_proj → per-head wts   │  │  Main compressor:                     │
                    │  ├─ fused_indexer_q_rope_quant    │  │  ├─ compress Ca,Za → main KV cache    │
                    │  └─ indexer_op → topk_indices     │  │  │  (512-dim, fused kernel)            │
                    │                                  │  │                                      │
                    └──────────── sync ────────────────┘  └──────────────────────────────────────┘
                                        │
                                        ▼
                            MLA Attention (fuses SWA + sparse + sink)
                                        │
                                        ▼
                            Inverse RoPE + FP8 quant → wo_a → wo_b → output
```

The indexer runs on the default stream because it's heavier (contains its own
compressor + quantized matmul + top-k selection). The main compressor and SWA
insert share the auxiliary stream.

In code:
```python
maybe_execute_in_parallel(
    lambda: self.indexer(hidden_states, qr, positions, ...),  # fn0 = default stream
    kv_insert_and_compress,                                    # fn1 = aux stream
    self.ln_events[0], self.ln_events[1], self.aux_stream,
)
```

---

## Putting It All Together: Full C4A Data Flow

Here is the complete picture, mapping visualization elements to code modules:

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ Per-token projections (every token 0–12)                                │
 │                                                                         │
 │   hidden_states ──► fused_wqa_wkv ──► split ──► qr (q-route latent)    │
 │                                              └──► kv (shared key-value) │
 │                                                                         │
 │   hidden_states ──► main compressor.fused_wkv_wgate ──► Ca, Za          │
 │   hidden_states ──► indexer.compressor.fused_wkv_wgate ──► Cb, Zb       │
 └─────────────────────────────────────────────────────────────────────────┘
                               │
     ┌─────────────────────────┼──────────────────────────────┐
     ▼                         ▼                              ▼
 ┌──────────┐          ┌──────────────┐              ┌──────────────┐
 │ SWA KV   │          │ Main         │              │ Indexer       │
 │ Insert   │          │ Compressor   │              │              │
 │          │          │              │              │ Own compressor│
 │ kv + RoPE│          │ Ca,Za → C_j  │              │ Cb,Zb → 128d │
 │ + FP8    │          │ (stride 4,   │              │ KV cache     │
 │ → SWA    │          │  window 8)   │              │              │
 │   cache  │          │ → 512d main  │              │ q + weights  │
 │          │          │   KV cache   │              │ → top-512    │
 │ (orange  │          │              │              │   indices    │
 │  tokens) │          │ (gradient    │              │              │
 │          │          │  tokens)     │              │ (green       │
 └────┬─────┘          └──────┬───────┘              │  tokens)     │
      │                       │                      └──────┬───────┘
      │                       │                             │
      │    ┌──────────────────┴─────────────────────────────┘
      │    │
      ▼    ▼
 ┌─────────────────────────────────────────────────────┐
 │ MLA Attention (FlashMLA kernel)                      │
 │                                                      │
 │  For each query q_i:                                 │
 │    1. Attend to k(v) in SWA window    ← orange row   │
 │    2. Attend to top-512 C^compress    ← gradient row │
 │    3. Add attention sink bias         ← gold token   │
 │    4. Merge via log-sum-exp                          │
 │                                                      │
 │  Causality for compressed entries:                   │
 │    q0–q2:  []           (too early)                  │
 │    q3–q6:  [C0]                                      │
 │    q7–q10: [C0, C1]                                  │
 │    q11–12: [C0, C1, C2]                              │
 └──────────────────────┬──────────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────────┐
 │ Output Projection                                    │
 │                                                      │
 │  attention_output                                    │
 │    → fused_inv_rope_fp8_quant  (undo R(j), quant)   │
 │    → fp8_einsum(wo_a)          (batched matmul)      │
 │    → wo_b                      (final projection)    │
 │    → output                                          │
 └─────────────────────────────────────────────────────┘
```

---

## Component 5: FlashMLA — How SWA and Compressed Tokens Are Fused

**Files:** `vllm/v1/attention/ops/flashmla.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`

Inside FlashMLA, SWA and compressed tokens are handled with **identical math**.
The **only** difference is which cache pointer and index array the producer
warpgroup reads from. There is no difference in RoPE handling (RoPE is
already baked into the cached KV before FlashMLA sees it), no difference in
the QK/PV gemm, and no difference in softmax. Once a KV block is loaded into
shared memory, the consumer warpgroup cannot tell whether it came from SWA or
compressed cache.

### Decode: Two Caches, One Fused Kernel

During decode, SWA and compressed tokens are passed as **separate inputs** to
`flash_mla_with_kvcache`:

```python
flash_mla_with_kvcache(
    q=q,                                    # [num_tokens, 1, 512]
    # --- SWA group ---
    k_cache=swa_cache,                      # [num_blocks, 64, head_dim]
    indices=swa_indices,                    # [num_tokens, max_swa_len]
    topk_length=swa_lens,                   # [num_tokens]  (≤128 valid)
    # --- Compressed group ---
    extra_k_cache=kv_cache,                 # [num_blocks, 64, head_dim]
    extra_indices_in_kvcache=topk_indices,  # [num_tokens, 512]
    extra_topk_length=topk_lens,            # [num_tokens]  (≤512 valid)
    # --- Sink ---
    attn_sink=self.attn_sink,               # [num_heads, 1, 512]
    head_dim_v=512,
    out=output.unsqueeze(1),                # [num_tokens, 1, 512]
)
```

### Inside the FlashMLA Kernel: Three-Phase Split-KV Architecture

**Source:** `FlashMLA/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`,
`FlashMLA/csrc/smxx/decode/combine/combine.cu`

The decode kernel uses a **split-KV** design with three thread groups
(warpgroups) per CTA, processing SWA and compressed tokens as a **single
unified block stream**:

```
┌──────────────────────────────────────────────────────────────────┐
│  Warpgroup 0 (Consumer): QK gemm → softmax → accumulate scores  │
│  Warpgroup 1 (Consumer): PV gemm → accumulate output             │
│  Warpgroup 2 (Producer): Load & dequantize KV blocks from GMEM  │
└──────────────────────────────────────────────────────────────────┘
```

#### How SWA and Compressed Blocks Are Linearized

The tile scheduler (`get_mla_metadata_kernel`) computes a **total block count**
per request that includes both SWA and compressed tokens:

```c++
// From get_decoding_sched_meta.cu (line 36-41)
cur_s_k = topk_length ? __ldg(topk_length + i) : topk;  // SWA tokens
if (cur_s_k == 0) cur_s_k = 1;
if (extra_topk) {
    cur_s_k = ceil(cur_s_k, block_size_n);     // Pad SWA to block boundary
    cur_s_k += extra_topk_length ? __ldg(extra_topk_length + i) : extra_topk;
                                                 // Append compressed tokens
}
```

This creates a linear sequence of blocks where **SWA blocks come first,
compressed blocks come after**:

```
Block indices for one request:
[0 .. num_orig_kv_blocks-1] = SWA blocks  (indices → params.kv)
[num_orig_kv_blocks .. end] = Compressed blocks  (extra_indices → params.extra_kv)
```

The scheduler then distributes these linearized blocks across SM partitions
using a greedy load-balancing algorithm, splitting requests across partitions
if needed.

#### Producer Warpgroup: Dispatching to the Right Cache

The producer warpgroup (warpgroup 2) loads KV blocks into shared memory. For
each block, it checks whether it's an "original" (SWA) or "extra" (compressed)
block using a compile-time dispatch:

```c++
// From splitkv_mla.cuh (line 483-504)
auto process_one_block = [&](int block_idx, auto is_extra_block_t, ...) {
    if constexpr (!IS_EXTRA_BLOCK) {
        // SWA block: read from primary cache
        indices_base = gIndices + block_idx * TOPK_BLOCK_SIZE;
        k_ptr = params.kv;
        page_block_size = params.page_block_size;
        k_block_stride = params.stride_kv_block;
    } else {
        // Compressed block: read from extra cache
        indices_base = gExtraIndices + (block_idx - num_orig_kv_blocks) * TOPK_BLOCK_SIZE;
        k_ptr = params.extra_kv;
        page_block_size = params.extra_page_block_size;
        k_block_stride = params.stride_extra_kv_block;
    }
    // ... dequantize FP8 → bf16, write to shared memory
};
```

The actual loop processes SWA blocks first, then compressed blocks:

```c++
// Process all SWA blocks (line 656-658)
for (block_idx = start; block_idx < min(num_orig_kv_blocks, end); block_idx++)
    process_one_block(block_idx, IsOrigBlock{}, ...);

// Process all compressed blocks (line 660-666)
if (num_orig_kv_blocks < end)
    process_one_block(max(start, num_orig_kv_blocks), IsExtraBlock{}, IsFirstExtraBlock{});
for (block_idx = max(start, num_orig_kv_blocks)+1; block_idx < end; block_idx++)
    process_one_block(block_idx, IsExtraBlock{}, ...);
```

#### Consumer Warpgroup: Identical Math for Both Types

Once a KV block lands in shared memory, the consumer warpgroup processes it
identically regardless of whether it came from SWA or compressed cache:

```c++
// Same for every block (line 218-270):
for (int block_idx = start; block_idx < end; block_idx++) {
    // 1. QK gemm: rP = Q @ K^T  (in shared memory)
    gemm(tiled_mma_QK, sQ, sK, rP);

    // 2. Softmax with online rescaling
    scale_softmax(rP, rS, rO, scale, sScale, rM, rL, is_kv_valid, ...);

    // 3. PV gemm: rO += softmax(QK^T) @ V
    gemm(tiled_mma_PV, rS, sV, rO);
}
```

The running max (`rM`) and sum (`rL`) are maintained across **all** blocks —
SWA and compressed alike — using the standard online softmax algorithm. There
is no separate accumulator per cache type.

#### Tensor Core MMA Instructions: Precision Pipeline

Both SM90 (Hopper) and SM100 (Blackwell) kernels store the KV cache in FP8
but **dequantize to bf16 before the MMA**. All tensor core operations are
**bf16 × bf16 → f32** on both architectures. The difference is the MMA
instruction family and operand placement.

##### SM90 (Hopper): WGMMA — Warpgroup Matrix Multiply-Accumulate

The SM90 sparse-FP8 decode kernel (`csrc/sm90/decode/sparse_fp8/config.h`,
lines 113–131) uses four WGMMA variants:

```
Q@K GEMM (WG0):
  GMMA::MMA_64x64x16_F32BF16BF16_SS    (Q in smem, K in smem)
  GMMA::MMA_64x64x16_F32BF16BF16_RS    (Q in registers, K in smem)
  → PTX: wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16

P@V GEMM (WG0 local-P / WG1 remote-P):
  GMMA::MMA_64x256x16_F32BF16BF16_RS   (P in registers, V in smem)
  GMMA::MMA_64x256x16_F32BF16BF16_SS   (P in smem, V in smem)
  → PTX: wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
```

| Stage | A operand | B operand | Accumulator | WGMMA tile shape |
|-------|-----------|-----------|-------------|------------------|
| Q@K   | Q (bf16)  | K (bf16)  | rP (f32)    | 64×64×16         |
| P@V   | P (bf16)  | V (bf16)  | rO (f32)    | 64×256×16        |

Operand placement notes:
- **Two P@V variants**: WG0 uses `RS` (P in registers from softmax output),
  WG1 uses `SS` (P from shared memory after WG0 writes it via `stmatrix`).
- **P@V uses 64×256 tiles** (4× wider N than Q@K's 64×64) because
  `HEAD_DIM_V = 512` and each warpgroup handles half (`HEAD_DIM_V/2 = 256`).

##### SM100 (Blackwell): UMMA via `tcgen05.mma` — Tensor Memory Accelerator

The SM100 sparse-FP8 decode kernel (`csrc/sm100/decode/head64/config.h`,
lines 196–202) uses Blackwell's **UMMA** (Unified Matrix Multiply-Accumulate)
with **Tensor Memory (TMEM)** for the accumulator:

```
Q@K GEMM (MMA warp in WG1):
  SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, 64, 128>
    A: Tensor Memory (Q)     B: Shared Memory (K)     C: Tensor Memory (P)
  → PTX: tcgen05.mma.ws.cta_group::1.kind::f16 [tmem_C], [tmem_A], smem_desc_B, idescE, ...
  N = 128 = B_TOPK*2 because SM100 uses "dual GEMM" (processes two K tiles
  in one MMA instruction, effectively doubling throughput)

P@V GEMM (MMA warp in WG1):
  SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, 64, 256>
    A: Shared Memory (S)     B: Shared Memory (V)     C: Tensor Memory (O)
  → PTX: tcgen05.mma.ws.cta_group::1.kind::f16 [tmem_C], smem_desc_A, smem_desc_B, idescE, ...

Note on "kind::f16": In PTX, `kind::f16` is a **category** covering fp16, bf16,
and tf32 — it does NOT mean fp16 specifically. The actual operand data type is
encoded in the **instruction descriptor** (`idescE`), where CUTLASS's
`UMMA::make_instr_desc<bf16, bf16, float, ...>()` sets `a_format_ = BF16` and
`b_format_ = BF16` (via `to_UMMAFormat<bfloat16_t>() → F16F32Format::BF16`).
So the real computation is **bf16 × bf16 → f32**, not fp16.
```

| Stage | A operand | B operand | Accumulator | UMMA tile shape | PTX instruction |
|-------|-----------|-----------|-------------|-----------------|-----------------|
| Q@K   | Q (bf16, TMEM) | K (bf16, SMEM) | P (f32, TMEM) | 64×128×16 | `tcgen05.mma.ws...kind::f16` |
| P@V   | S (bf16, SMEM) | V (bf16, SMEM) | O (f32, TMEM) | 64×256×16 | `tcgen05.mma.ws...kind::f16` |

Key SM100 differences from SM90:
- **Tensor Memory (TMEM)** replaces registers for Q and accumulators (P, O).
  TMEM is a 512-column × 128-row scratchpad private to each SM, addressed by
  column index. The kernel allocates columns 0–255 for O, 256–399 for Q,
  and 400–463 for P.
- **`TS` (TMEM-SMEM) for Q@K**: Q lives in TMEM (loaded via UTCCP from SMEM),
  K in shared memory. No register file pressure for Q.
- **`SS` (SMEM-SMEM) for P@V**: Both S and V in shared memory. The softmax
  output S is written to SMEM by WG0, V is dequantized into SMEM by WG2.
- **Dual GEMM**: Q@K uses N=128 (2×B_TOPK) to process two 64-token blocks
  in a single MMA call, exploiting SM100's higher throughput.
- **Single MMA warp**: Only one warp (warp 4, elected via `elect_one_sync()`)
  issues `tcgen05.mma` instructions, unlike SM90 where an entire warpgroup
  cooperates on WGMMA.

##### Comparison: SM90 vs SM100 MMA

| Property | SM90 (Hopper) | SM100 (Blackwell) |
|----------|---------------|-------------------|
| MMA family | WGMMA (`wgmma.mma_async`) | UMMA (`tcgen05.mma`) |
| Input dtype | bf16 × bf16 | bf16 × bf16 |
| Accumulator dtype | f32 (registers) | f32 (Tensor Memory) |
| FP8 in tensor core? | No — dequant in producer WG | No — dequant in WG2 |
| Q@K operands | SMEM/Reg × SMEM | TMEM × SMEM |
| P@V operands | Reg/SMEM × SMEM | SMEM × SMEM |
| Q@K tile | 64×64×16 | 64×128×16 (dual GEMM) |
| P@V tile | 64×256×16 | 64×256×16 |
| Dequant method | `fp8→f32→bf16` via `__float22bfloat162_rn` | `fp8→bf16` via `fp8x2_to_bf16x2_with_scale` with e8m0 scales |

##### Data Type Flow (Common to Both Architectures)

```
KV Cache (FP8 e4m3, per-tile e8m0 scales)
    │
    ▼  Producer/Dequant WG: dequantize fp8 → bf16 (scaled per tile)
    │  SM90: cvt_fp8x8_bf16x8() via __float22bfloat162_rn
    │  SM100: fp8x2_to_bf16x2_with_scale() via __nv_cvt_e8m0x2_to_bf162raw
    ▼
Shared Memory (bf16)    [SM100: Q also copied to TMEM via UTCCP]
    │
    ▼  Q@K:  Q(bf16) × K(bf16) → P(f32)
    ▼  softmax: P(f32) → S(bf16)
    ▼  P@V:  S(bf16) × V(bf16) → O(f32)
    ▼
Output: f32 → bf16  (no-split: direct output; split: f32 to accumulator → combine kernel)
```

Key takeaway: **FP8 never touches tensor cores on either architecture.**
Dequantization always happens before the MMA, ensuring bf16 precision for all
dot-product accumulations.

##### GEMM Precision: Sparse MLA (FlashMLA) vs Indexer MQA (DeepGEMM)

The C4A architecture uses two distinct GEMM pipelines with fundamentally
different precision strategies:

**Sparse MLA — FlashMLA (Attention Kernel)**

The main attention path computes Q@K and P@V using FlashMLA. FP8 is
dequantized *before* the tensor cores see any data:

```
KV Cache: FP8 e4m3 + per-tile e8m0 scales
    │
    ▼  Dequant WG: fp8 → bf16 (in SMEM, before MMA)
    │
    ▼  Q@K:  bf16 × bf16 → f32    (WGMMA on SM90, UMMA on SM100)
    ▼  P@V:  bf16 × bf16 → f32
    │
    ▼  Output: f32 → bf16
```

This ensures maximum numerical precision for the attention dot products.
The dequantization cost is hidden by overlapping it with MMA via warpgroup
specialization (producer/dequant WG runs concurrently with consumer WG).

- Code: `flash_mla_with_kvcache()` in `deepseek_v4_attention.py` line 1689
- Kernel: `FlashMLA/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` (SM90),
  `FlashMLA/csrc/sm100/decode/head64/kernel.cuh` (SM100)

**Indexer MQA — DeepGEMM (Score Computation)**

The indexer computes approximate Q@K scores to select top-k KV blocks.
This is a *single GEMM* (no P@V needed — only logits for selection), and
it operates directly in low-precision formats:

```
Q: FP8 e4m3 (or MXFP4)     K: FP8 e4m3 (or MXFP4)
    │                           │
    └───────────┬───────────────┘
                ▼
    DeepGEMM: fp8 × fp8 → f32   (or fp4 × fp4 → f32)
                │
                ▼
    Logits (f32) → top-k selection → block indices
```

DeepGEMM feeds low-bit operands directly into tensor cores — no
dequantization step. This is acceptable because:

1. **Only scores matter, not values** — the indexer computes Q@K logits
   purely for ranking. Small precision errors don't change which blocks
   are selected.
2. **Single GEMM, not attention** — there's no softmax or P@V pass. The
   output is f32 logits fed into a top-k kernel.
3. **FP4 option halves memory** — `use_fp4_indexer_cache` stores K in
   MXFP4 format, reducing indexer cache by 2× while maintaining selection
   quality.

- Code: `fp8_fp4_mqa_logits()` / `fp8_fp4_paged_mqa_logits()` in
  `vllm/utils/deep_gemm.py` lines 346, 416
- Caller: `SparseAttnIndexer` in
  `vllm/model_executor/layers/sparse_attn_indexer.py` line 221 (prefill),
  line 308 (decode)

**Comparison Table**

| Aspect | Sparse MLA (FlashMLA) | Indexer MQA (DeepGEMM) |
|---|---|---|
| **Purpose** | Full attention (Q@K + softmax + P@V) | Score-only (Q@K logits for top-k) |
| **# GEMMs** | 2 (Q@K, P@V) | 1 (Q@K only) |
| **MMA input dtype** | bf16 × bf16 (dequanted from fp8) | fp8 × fp8 or fp4 × fp4 (direct) |
| **MMA output dtype** | f32 | f32 |
| **Dequant before MMA?** | Yes — fp8→bf16 in producer/dequant WG | No — low-bit goes straight to TC |
| **Output** | Attention values (bf16) + LSE (f32) | Logits (f32) → top-k indices (int32) |
| **Precision trade-off** | None — full bf16 precision | Acceptable — only ranking matters |
| **Kernel backend** | FlashMLA (custom CUTLASS) | DeepGEMM |

##### Indexer MQA: MMA Instructions and Scale Formats

The `fp8_fp4_mqa_logits` function in DeepGEMM dispatches to **three different
kernel paths** depending on the data format and GPU architecture. Despite the
"fp8_fp4" name, it never mixes FP8×FP4 in one MMA — it handles both formats
via separate dispatch.

**Dispatch Logic** (`DeepGEMM/csrc/apis/attention.hpp` lines 232–244):

| Condition | Kernel | MMA Instruction |
|---|---|---|
| FP4 + SM100 (Blackwell) | `sm100_fp4_mqa_logits` | `tcgen05.mma.cta_group::1.kind::mxf4` — native **FP4×FP4→F32** UMMA |
| FP4 + SM120 (fallback) | dequant FP4→BF16→FP8, then `smxx_fp8_mqa_logits` | `MMA_64xNx32_F32E4M3E4M3_SS_TN` — **FP8×FP8→F32** WGMMA |
| FP8 (SM90/100/120) | `smxx_fp8_mqa_logits` | `MMA_64xNx32_F32E4M3E4M3_SS_TN` — **FP8×FP8→F32** WGMMA |

**SM90 FP8 path** — WGMMA with E4M3 operands:

```cpp
// DeepGEMM/deep_gemm/include/deep_gemm/mma/sm90.cuh line 36-51
using WGMMA = FP8MMASelector<BLOCK_Q * kNumHeads>::type;
// → MMA_64xNx32_F32E4M3E4M3_SS_TN  (both A and B are FP8 E4M3)
// PTX: wgmma.mma_async.sync.aligned.m64nNk32.f32.e4m3.e4m3
```

**SM100 FP4 path** — UMMA with MXFP4 block-scaled operands:

```cpp
// DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp4_mqa_logits.cuh line 249
auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
    cutlass::float_e2m1_t,   // A type = FP4 (MXFP4, 1-bit mantissa)
    cutlass::float_e2m1_t,   // B type = FP4 (MXFP4)
    float,                    // accumulator = F32
    cutlass::float_ue8m0_t,  // block scale = UE8M0 (exponent-only, 1 byte)
    UMMA_M, UMMA_N,
    cute::UMMA::Major::K, cute::UMMA::Major::K>();
// PTX: tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32
```

The `kind::mxf4` tells the tensor core to natively consume microscaled FP4
for both operands. The `block_scale.block32` means each group of 32 FP4
values shares one UE8M0 scale factor, applied by the hardware during the MMA.

**Scale Formats — FP8 vs MXFP4**

The two paths use fundamentally different scaling strategies:

| | FP8 Path | MXFP4 Path |
|---|---|---|
| **Q tensor** | `[seq_len, num_heads, head_dim]` fp8_e4m3 | `[seq_len, num_heads, head_dim/2]` uint8 (packed, 2 values/byte) |
| **Q scale** | None (folded into `weights`) | `[seq_len, num_heads]` int32 (ue8m0 scales packed) |
| **KV tensor** | `[seq_len_kv, head_dim]` fp8_e4m3 | `[seq_len_kv, head_dim/2]` uint8 (packed) |
| **KV scale** | `[seq_len_kv]` **float32** — 1 scalar per token | `[seq_len_kv]` **int32** — ue8m0 scales per block-of-32 |
| **Scale granularity** | Per-token (1 FP32 for all 128 dims) | Per-32-elements (4 UE8M0 bytes for 128 dims) |
| **Scale overhead** | 4 bytes/token | 4 bytes/token (same!) |
| **Where applied** | Post-MMA multiply in kernel | Hardware-applied during MMA |

FP8 scale detail — `kv_sf` is a 1-D `float32` tensor of shape `[seq_len_kv]`:

```cpp
// attention.hpp line 191 — FP8 path asserts float32 scale
DG_HOST_ASSERT(kv_sf.scalar_type() == torch::kFloat);
// Shape: [seq_len_kv] — one scalar per compressed KV position
```

MXFP4 scale detail — `kv_sf` is 4 `ue8m0` bytes reinterpreted as `int32`:

```cpp
// attention.hpp line 171 — FP4 path asserts int32 (packed ue8m0)
DG_HOST_ASSERT(kv_sf.scalar_type() == torch::kInt32);
// Shape: [seq_len_kv] — 4 bytes = 4 ue8m0 scales covering 4×32=128 dims
```

The `int32` view trick: DeepGEMM's API accepts scales as `int32` tensors for
both paths. For FP4, the 4 `ue8m0` bytes (one per block of 32 dims) are
reinterpreted as a single int32. For FP8, the 4-byte FP32 scale is
reinterpreted as int32. Same API surface, different semantics.

**KV Cache Layout in vLLM** — from `sparse_attn_indexer.py` lines 43–61:

```python
def _gather_workspace_shapes(total_seq_lens, head_dim, fp8_dtype, use_fp4_cache):
    if use_fp4_cache:
        # MXFP4: packed values + ue8m0 block scales
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),          # 64 bytes for head_dim=128
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),  # 4 bytes (128/32=4 scales)
        )
    # FP8: raw values + per-token fp32 scale
    return (
        ((total_seq_lens, head_dim), fp8_dtype),       # 128 bytes
        ((total_seq_lens, 4), torch.uint8),            # 4 bytes (1 fp32 scale as raw bytes)
    )
```

The paged indexer KV cache (`deepseek_v4_attention.py` line 1976) allocates
`head_dim + head_dim // quant_block_size * 4 = 128 + 4 = 132` bytes per slot
for FP8, or reuses the same 132-byte allocation for FP4 (using only
`head_dim/2 + head_dim/32 = 64 + 4 = 68` bytes, with the rest unused).

**Output Projection: Also DeepGEMM**

After attention, the output projection `O @ Wo_a` also uses DeepGEMM's
`fp8_einsum` (equation `"bhr,hdr->bhd"`), performing an FP8 GEMM with
block-scaled quantization. This is a standard linear-layer GEMM, not
attention, so low-bit precision is standard practice:

```python
# deepseek_v4_attention.py line 537
torch.ops.vllm.deepseek_v4_fp8_einsum(o_fp8, o_scale, wo_a_fp8, wo_a_scale, z, "bhr,hdr->bhd", ...)
```

#### Combine Kernel: Where the Sink Merges In

When a request is split across multiple SM partitions, each partition produces
a partial `(output, lse)` pair. The **combine kernel** merges these splits
and adds the attention sink:

```c++
// From combine.cu (line 101-112)
if (params.attn_sink != nullptr) {
    float attn_sink = __ldg(params.attn_sink + q_head_idx);
    if (global_lse != INFINITY) {
        // Merge sink into global LSE:
        // lse_new = lse + log2(1 + exp2(sink - lse))
        global_lse += log2f(1 + exp2f(attn_sink * CUDART_L2E_F - global_lse));
    } else {
        // No tokens at all — output is purely the sink
        global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink * CUDART_L2E_F;
    }
}
```

The sink is **not** a key-value pair. It's a scalar bias per head that
modifies the softmax denominator, effectively acting as a "virtual token"
with fixed attention weight.

#### Summary: The Full Decode Pipeline

```
                     ┌─────────────────────────────────────┐
                     │  Tile Scheduler                      │
                     │  (get_decoding_sched_meta)           │
                     │                                      │
                     │  Per request, linearize:             │
                     │  [SWA blocks | Compressed blocks]    │
                     │  total = ceil(swa_len/64) +          │
                     │          ceil(compressed_len/64)     │
                     │                                      │
                     │  Distribute across SM partitions     │
                     └──────────────┬──────────────────────┘
                                    │
                     ┌──────────────▼──────────────────────┐
                     │  Main Kernel (per SM partition)      │
                     │                                      │
                     │  Producer WG2:                       │
                     │    for each block in my range:       │
                     │      if block < num_orig_blocks:     │
                     │        load from kv[indices[block]]  │ ← SWA
                     │      else:                           │
                     │        load from extra_kv[           │ ← Compressed
                     │          extra_indices[block]]       │
                     │      dequant FP8 → bf16 → smem      │
                     │                                      │
                     │  Consumer WG0+WG1:                   │
                     │    for each block (same loop):       │
                     │      P = Q @ K^T  (from smem)        │
                     │      online softmax(P) → S           │
                     │      O += S @ V                      │
                     │    write (O_partial, LSE) to accum   │
                     └──────────────┬──────────────────────┘
                                    │
                     ┌──────────────▼──────────────────────┐
                     │  Combine Kernel                      │
                     │                                      │
                     │  For each (request, head):           │
                     │    merge all split (O, LSE) via      │
                     │      log-sum-exp                     │
                     │    add attn_sink to denominator      │
                     │    write final output                │
                     └─────────────────────────────────────┘
```

**Key insight:** Inside the main kernel, SWA and compressed blocks are
processed by the **same consumer loop with identical math**. The only
difference is in the producer — which cache pointer and index array to read
from. Once data is in shared memory, the kernel doesn't know or care where
it came from.

### Prefill: Unified Buffer, Single Pass

During prefill, the distinction is even simpler. vLLM gathers both SWA and
compressed entries into **one bf16 buffer** and merges their index arrays
before calling the kernel:

```python
# Gather compressed KV into workspace buffer at offset 0
dequantize_and_gather_k_cache(workspace[:N], compressed_k_cache, ...)

# Gather SWA KV into the same buffer at offset N
dequantize_and_gather_k_cache(workspace[N:], swa_k_cache, ...)

# Merge index arrays: compressed indices [0..N) + SWA indices [N..)
combined_indices = combine_topk_swa_indices(topk_indices, swa_indices)

# Single attention pass over the unified buffer
flash_mla_sparse_fwd(q, workspace, combined_indices, combined_lens, ...)
```

where `N = ceil(max_model_len / compress_ratio)`.

The prefill kernel (`SparseAttnFwdParams`) has a **single** `kv` pointer —
no `extra_kv` at all. From its perspective, SWA and compressed are just
different regions in the same buffer, indexed by the combined index array.

### Why Different Strategies for Decode vs Prefill?

| | Decode | Prefill |
|---|---|---|
| **Strategy** | Two cache pointers, producer-side dispatch | Single unified buffer |
| **Why** | Avoids gather-copy; producer loads directly from paged caches. With 1 query token, the split-KV overhead is minimal | Many query tokens amortize the gather cost. Single buffer = simpler index math, no producer branching |
| **Sink handling** | Combine kernel (after all splits merge) | Separate `merge_attention_with_sink` call |

### Reference Implementation (for verification)

The reference path (`_forward_sparse_mla_compressed_decode_reference`) makes
the two-group merge explicit in Python:

```python
# Compressed attention (chunked to limit workspace)
for chunk in compressed_chunks:
    comp_scores = accumulate_attention(q, chunk)
comp_output, comp_lse = finish_attention(comp_scores)

# SWA attention
swa_scores = accumulate_attention(q, swa_kv)
swa_output, swa_lse = finish_attention(swa_scores)

# Merge via log-sum-exp + sink bias
_merge_reference_attention_with_sink(
    subset_outputs=[comp_output, swa_output],
    subset_lses=[comp_lse, swa_lse],
    output=output,
)
```

Note: the reference treats SWA and compressed as truly separate (two
independent attention computations + explicit LSE merge), while the optimized
FlashMLA kernel linearizes them into one block stream. Both produce
mathematically identical results.

### Input/Output Shape Summary

| Tensor | Shape | Notes |
|--------|-------|-------|
| q (query) | `[b, s_q, h_q, d_qk]` = `[B, 1, 64, 512]` | bf16, per-request decode |
| kv (SWA cache) | `[num_blocks, 64, 1, bytes_per_token]` | FP8 paged, block_size=64 |
| indices (SWA) | `[b, s_q, topk]` = `[B, 1, ≤128]` | int32 slot indices |
| topk_length | `[b]` | SWA valid count per request |
| extra_kv (compressed) | `[num_blocks, 64, 1, bytes_per_token]` | FP8 paged, same format as SWA |
| extra_indices | `[b, s_q, extra_topk]` = `[B, 1, 512]` | int32 slot indices from indexer |
| extra_topk_length | `[b]` | Compressed valid count per request |
| attn_sink | `[h_q]` = `[64]` | float32 per-head bias |
| output | `[b, s_q, h_q, d_v]` = `[B, 1, 64, 512]` | bf16 |
| lse | `[b, s_q, h_q]` = `[B, 1, 64]` | float32, log-sum-exp per head |

The `bytes_per_token` for MODEL1 (DeepSeek V4) layout:
`448 (fp8 NoPE) + 64×2 (bf16 RoPE) + 7 (e8m0 scales) + 1 (padding) = 584 bytes`

### Why SWA and Compressed Must Be Treated Separately: First Principles

At first glance, both SWA k(v) and compressed C_j are just key-value vectors of
the same dimension (512-dim MLA latent). The query computes `q · k` with both.
So why can't we just concatenate them into one pool and run a single attention?

There are **four fundamental differences** that force separate treatment:

#### 1. RoPE Positions Encode Different Things

SWA tokens carry **real token positions**. Each k(v) at position `p` gets
`RoPE(p)` applied:

```
k(v)_5  → RoPE(5)
k(v)_6  → RoPE(6)
k(v)_7  → RoPE(7)
```

Compressed tokens use **anchor positions** — the left edge of the compression
window, always a multiple of `compress_ratio`:

```
C0 (compresses tokens 0-3)  → RoPE(0)    ← anchor = 4×0
C1 (compresses tokens 4-7)  → RoPE(4)    ← anchor = 4×1
C2 (compresses tokens 8-11) → RoPE(8)    ← anchor = 4×2
```

The position formula (from `deepseek_compressor.py` line 339):
```
compressed_rope_pos = (position // compress_ratio) * compress_ratio
```

This means within any SWA window, the relative distances `q_pos - k_pos` are
fundamentally different for SWA vs compressed tokens. SWA tokens have exact
positions; compressed tokens have quantized positions. Mixing them in a single
RoPE-aware attention would not produce wrong results (the math still works),
but the **access patterns** are completely different, which matters for the
next three reasons.

#### 2. Different Caches with Different Access Patterns

SWA and compressed tokens live in **physically separate KV caches**:

```
SWA cache:        [num_blocks, 64, 512]  ← contiguous sliding window
                  Access: gather last ≤128 tokens via block_table

Compressed cache: [num_blocks, 64, 512]  ← sparse, indexed by top-k
                  Access: gather 512 scattered slots via topk_indices
```

SWA access is **nearly contiguous** — you read the last 128 positions, which
span at most 2-3 blocks. Compressed access is **fully scattered** — the 512
top-k entries can come from anywhere in the sequence, touching potentially
hundreds of blocks.

These access patterns demand fundamentally different GPU tiling strategies:
- SWA: sequential block reads, high cache hit rate
- Compressed: scattered gathers, memory-bandwidth-bound

A single kernel trying to serve both patterns needs to branch on which cache
to read from — but as we'll see in the FlashMLA kernel details below, this
branching happens only in the **producer** warpgroup. Once data lands in
shared memory, the consumer sees no difference.

#### 3. Different Cardinalities and Validity Masks

```
SWA:        ≤128 tokens   (or fewer at sequence start)
Compressed: ≤512 entries  (or fewer when sequence is short)
Sink:       1 bias term   (always present)
```

If concatenated into `[128 + 512 + 1]` slots, the kernel would need a
**partition mask** to know which region has how many valid entries — it can't
just use a single `topk_length` scalar. The separate-input design lets each
group carry its own length cleanly.

#### 4. The Log-Sum-Exp Merge Is Mathematically Exact

The key insight that makes separate computation viable: softmax decomposes
perfectly across partitions.

Given two disjoint key sets A (SWA) and B (compressed):

```
softmax(q · [A, B]) = merge(softmax(q · A), softmax(q · B), lse_A, lse_B)
```

where the merge uses:
```
max_score = max(lse_A, lse_B)
output = (exp(lse_A - max_score) · out_A + exp(lse_B - max_score) · out_B)
       / (exp(lse_A - max_score) · denom_A + exp(lse_B - max_score) · denom_B)
```

This is **not an approximation** — it produces bit-identical results to
computing attention over the concatenated set. The reference implementation
(`_forward_sparse_mla_compressed_decode_reference`) proves this by computing
SWA and compressed attention independently, then calling `lse_merge` followed
by `merge_attention_with_sink`.

#### Summary: Why Separate Is Better

```
                    Concatenate & attend          Separate caches, linearized blocks
                    ─────────────────────         ──────────────────────────────────
Math correctness    ✓ identical                   ✓ identical (online softmax)
RoPE handling       Works (positions differ       Same
                    but math is valid)
GPU efficiency      ✗ Must gather into one        ✓ Producer loads directly from
                    buffer first                    paged caches (no gather copy)
Memory              ✗ Extra bf16 buffer for       ✓ Decode: dequant to smem only,
                    all tokens                      one block at a time
Validity masking    ✗ Need partition offsets       ✓ Each group has simple scalar len
Prefill             Actually used! (unified       Decode: linearized blocks is
                    buffer, single pass)          better with few query tokens
```

The prefill path **does** use the concatenation approach — because with many
query tokens the gather cost is amortized and simpler scheduling wins. The
decode path uses separate computation because with 1 query token per request,
avoiding the gather copy and using specialized tiling for each access pattern
dominates.

---

## KV Cache Layout for C4A Layers

Each C4A layer uses **five** KV cache allocations under vLLM's unified block
manager, all sharing a logical block size of 256 token positions:

| Cache | Block size | Content | Dtype |
|-------|-----------|---------|-------|
| Main compressed KV | 256/4 = 64 entries | Compressed MLA latents (512-dim fp8 + scales) | `fp8_ds_mla` |
| SWA KV | 64 entries | Uncompressed tokens (512-dim fp8 + scales) | `uint8` |
| Main compressor state | 4 entries | Rolling `[kv_state, score_state]` (fp32) | `float32` |
| Indexer compressed KV | 256/4 = 64 entries | Compressed indexer keys (128-dim fp8 + scales) | `uint8` |
| Indexer compressor state | 4 entries | Rolling `[kv_state, score_state]` (fp32) | `float32` |

All five fit into **three page-size buckets**, eliminating cross-pool
fragmentation. The compressor state caches are treated as sliding-window KV
under the same abstraction as SWA, so prefix caching, disaggregated prefill,
and CUDA graphs all work without special-casing.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/models/deepseek_v4.py` | Model definition, layer construction, `compress_ratio` routing |
| `vllm/model_executor/layers/deepseek_v4_attention.py` | MLA wrapper, attention impl, indexer, multi-stream orchestration |
| `vllm/model_executor/layers/deepseek_compressor.py` | Compressor module, state cache, Triton kernels |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | Top-k sparse attention indexer (`SparseAttnIndexer`) |
| `vllm/v1/attention/ops/deepseek_v4_ops/` | Fused ops: inv RoPE, Q RoPE quant, compress+norm+RoPE+insert |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | FlashMLA sparse backend for decode and prefill |
| `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | CUDA kernel: fused Q norm + KV RoPE + FP8 quant + SWA insert |

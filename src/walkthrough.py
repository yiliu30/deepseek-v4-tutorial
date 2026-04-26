#!/usr/bin/env python3
"""
DeepSeek V4 Annotated Walkthrough — traces every tensor shape through the full pipeline.

For each layer type (SWA-only, C4A, C128A), prints exact dimensions at every stage:
  HC pre → attn_norm → Q projection → KV projection → Compressor → Indexer →
  sparse_attn → de-RoPE → O projection → HC post → FFN (Gate → MoE → shared) → HC post

Usage:
    python walkthrough.py          # full trace
    python walkthrough.py --quiet  # summary only
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(42)

from model import (
    Transformer, ModelArgs, Attention, Compressor, Indexer, MoE, Block,
    ParallelHead, Gate, Expert, RMSNorm,
    act_quant, fp4_act_quant, sparse_attn, hc_split_sinkhorn,
    apply_rotary_emb, rotate_activation, get_window_topk_idxs, get_compress_topk_idxs,
    linear, block_size, fp4_block_size, scale_fmt, scale_dtype,
)

QUIET = "--quiet" in sys.argv
STEP = 0  # global step counter for indentation

def S(t):
    """Format tensor shape compactly."""
    if t is None:
        return "None"
    if isinstance(t, tuple):
        return str(tuple(ti.shape for ti in t))
    return str(tuple(t.shape))

def T(t):
    """Format tensor shape + dtype."""
    if t is None:
        return "None"
    return f"{tuple(t.shape)} {t.dtype}"

def log(msg, indent=0):
    if not QUIET:
        print("  " * indent + msg)


# ════════════════════════════════════════════════════════════════════════════════
# Monkey-patch every component for deep tracing
# ════════════════════════════════════════════════════════════════════════════════

# ── Block.forward: HC pre/post + attn + ffn ────────────────────────────────────
_orig_block_forward = Block.forward

def _traced_block_forward(self, x, start_pos, input_ids):
    layer_id = self.layer_id
    ratio = self.attn.compress_ratio
    layer_type = "SWA-only" if ratio == 0 else f"C{ratio}A"
    components = [layer_type]
    if ratio == 4:
        components.append("Compressor(overlap) + Indexer")
    elif ratio > 0:
        components.append("Compressor(no-overlap)")

    log(f"\n{'─'*70}", 0)
    log(f"LAYER {layer_id} [{' + '.join(components)}]", 0)
    log(f"  input x: {T(x)}  start_pos={start_pos}", 0)

    # ── HC pre (attention) ──
    residual = x
    log(f"  ── HC pre (attn) ──", 0)
    log(f"    x (hc copies): {T(x)}  [B, S, hc={self.hc_mult}, D={x.shape[-1]}]", 0)

    # Inline hc_pre to trace intermediates
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(2).float()
    log(f"    flatten(2): {T(x_flat)}  [B, S, hc*D]", 0)
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
    log(f"    rsqrt: {T(rsqrt)}", 0)
    from torch.nn.functional import linear as f_linear
    mixes = f_linear(x_flat, self.hc_attn_fn) * rsqrt
    log(f"    hc_attn_fn: {T(self.hc_attn_fn)}  [mix_hc, hc*D]", 0)
    log(f"    mixes = x_flat @ hc_attn_fn^T * rsqrt: {T(mixes)}  [B, S, mix_hc={(2+self.hc_mult)*self.hc_mult}]", 0)
    pre, post, comb = hc_split_sinkhorn(mixes, self.hc_attn_scale, self.hc_attn_base,
                                         self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
    log(f"    hc_split_sinkhorn →", 0)
    log(f"      pre:  {T(pre)}   [B, S, hc] — weights for reducing hc→1", 0)
    log(f"      post: {T(post)}  [B, S, hc] — weights for expanding 1→hc", 0)
    log(f"      comb: {T(comb)}  [B, S, hc, hc] — doubly-stochastic combination", 0)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    log(f"    y = sum(pre * x, dim=hc): {T(y)}  [B, S, D] — single hidden state", 0)
    x_for_attn = y.to(dtype)
    attn_post, attn_comb = post, comb

    # ── attn_norm ──
    log(f"  ── attn_norm ──", 0)
    x_normed = self.attn_norm(x_for_attn)
    log(f"    RMSNorm({x_for_attn.shape[-1]}): {T(x_normed)}", 0)

    # ── Attention ──
    log(f"  ── Attention ──", 0)
    attn_out = self.attn(x_normed, start_pos)
    log(f"    attn output: {T(attn_out)}", 0)

    # ── HC post (attention) ──
    log(f"  ── HC post (attn) ──", 0)
    x = attn_post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.sum(attn_comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    x = x.type_as(attn_out)
    log(f"    post*attn + comb*residual: {T(x)}  [B, S, hc, D]", 0)

    # ── HC pre (FFN) ──
    residual = x
    log(f"  ── HC pre (ffn) ──", 0)
    shape2, dtype2 = x.size(), x.dtype
    x_flat2 = x.flatten(2).float()
    rsqrt2 = torch.rsqrt(x_flat2.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes2 = f_linear(x_flat2, self.hc_ffn_fn) * rsqrt2
    log(f"    mixes (ffn): {T(mixes2)}", 0)
    pre2, post2, comb2 = hc_split_sinkhorn(mixes2, self.hc_ffn_scale, self.hc_ffn_base,
                                             self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
    log(f"    pre: {T(pre2)}, post: {T(post2)}, comb: {T(comb2)}", 0)
    y2 = torch.sum(pre2.unsqueeze(-1) * x.view(shape2), dim=2).to(dtype2)
    log(f"    y (single state): {T(y2)}", 0)

    # ── ffn_norm ──
    log(f"  ── ffn_norm ──", 0)
    x_ffn = self.ffn_norm(y2)
    log(f"    RMSNorm: {T(x_ffn)}", 0)

    # ── MoE ──
    log(f"  ── MoE ──", 0)
    ffn_out = self.ffn(x_ffn, input_ids)
    log(f"    MoE output: {T(ffn_out)}", 0)

    # ── HC post (FFN) ──
    log(f"  ── HC post (ffn) ──", 0)
    x = post2.unsqueeze(-1) * ffn_out.unsqueeze(-2) + torch.sum(comb2.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    x = x.type_as(ffn_out)
    log(f"    final: {T(x)}  [B, S, hc, D]", 0)
    return x

Block.forward = _traced_block_forward


# ── Attention.forward: Q/KV projections, compressor, indexer, sparse_attn ──────
_orig_attn_forward = Attention.forward

def _traced_attn_forward(self, x, start_pos):
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
    win = self.window_size
    ratio = self.compress_ratio
    rd = self.rope_head_dim

    if self.compress_ratio and self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache[:, win:]
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    # ── Q projection (low-rank MLA) ──
    log(f"    ── Q projection (MLA low-rank) ──", 0)
    qr = q = self.q_norm(self.wq_a(x))
    log(f"      wq_a({x.shape[-1]}→{self.q_lora_rank}): {T(q)}  [B, S, q_lora_rank]", 0)
    q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
    log(f"      wq_b({self.q_lora_rank}→{self.n_local_heads}*{self.head_dim}): {T(q)}  [B, S, n_heads, head_dim]", 0)
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    log(f"      q_norm (per-head RMS): {T(q)}", 0)
    apply_rotary_emb(q[..., -rd:], freqs_cis)
    log(f"      RoPE on q[..., -{rd}:] (last {rd} dims): {T(q)}", 0)

    # ── KV projection ──
    log(f"    ── KV projection ──", 0)
    kv = self.wkv(x)
    log(f"      wkv({x.shape[-1]}→{self.head_dim}): {T(kv)}  [B, S, head_dim]", 0)
    kv = self.kv_norm(kv)
    log(f"      kv_norm: {T(kv)}", 0)
    apply_rotary_emb(kv[..., -rd:], freqs_cis)
    log(f"      RoPE on kv[..., -{rd}:]: {T(kv)}", 0)
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
    log(f"      FP8-sim on kv[..., :-{rd}] (nope dims): no-op in stub", 0)

    # ── Window topk indices ──
    topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
    log(f"    ── Window indices ──", 0)
    log(f"      window_topk_idxs: {S(topk_idxs)}  [B, S, win={win}]", 0)

    # ── Compressed KV + Indexer (if applicable) ──
    if self.compress_ratio:
        offset = kv.size(1) if start_pos == 0 else win
        log(f"    ── Compressed KV (ratio={ratio}) ──", 0)
        if self.indexer is not None:
            log(f"    ── Indexer (Lightning, topk={self.indexer.index_topk}) ──", 0)
            compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            log(f"      indexer topk_idxs: {S(compress_topk_idxs)}", 0)
        else:
            compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            log(f"      static compress_topk_idxs: {S(compress_topk_idxs)}", 0)
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        log(f"      combined topk_idxs (window+compress): {S(topk_idxs)}", 0)
    topk_idxs = topk_idxs.int()

    # ── Compress KV & Attention ──
    if start_pos == 0:
        if seqlen <= win:
            self.kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
        if self.compress_ratio:
            log(f"    ── Compressor ──", 0)
            kv_compress = self.compressor(x, start_pos)
            if kv_compress is not None:
                log(f"      compressed KV: {T(kv_compress)}", 0)
                kv = torch.cat([kv, kv_compress], dim=1)
                log(f"      kv (window + compressed): {T(kv)}", 0)
        log(f"    ── sparse_attn (prefill) ──", 0)
        log(f"      q: {S(q)}, kv: {S(kv)}, topk_idxs: {S(topk_idxs)}", 0)
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    else:
        self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        if self.compress_ratio:
            self.compressor(x, start_pos)
        log(f"    ── sparse_attn (decode) ──", 0)
        log(f"      q: {S(q)}, kv_cache: {S(self.kv_cache[:bsz])}, topk_idxs: {S(topk_idxs)}", 0)
        o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
    log(f"      attn output: {T(o)}  [B, S, H, D]", 0)

    # ── De-RoPE ──
    apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    log(f"    ── de-RoPE on o[..., -{rd}:] ──", 0)

    # ── O projection (grouped low-rank) ──
    log(f"    ── O projection (grouped low-rank) ──", 0)
    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    log(f"      reshape to groups: {T(o)}  [B, S, n_groups={self.n_local_groups}, heads_per_group*D]", 0)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    log(f"      wo_a: {T(wo_a)}  [n_groups, o_lora_rank={self.o_lora_rank}, heads_per_group*D]", 0)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    log(f"      einsum(o, wo_a): {T(o)}  [B, S, n_groups, o_lora_rank]", 0)
    x = self.wo_b(o.flatten(2))
    log(f"      wo_b({o.flatten(2).shape[-1]}→{self.dim}): {T(x)}  [B, S, D]", 0)
    return x

Attention.forward = _traced_attn_forward


# ── Compressor.forward ─────────────────────────────────────────────────────────
_orig_compressor_forward = Compressor.forward

def _traced_compressor_forward(self, x, start_pos):
    bsz, seqlen, _ = x.size()
    ratio = self.compress_ratio
    rd = self.rope_head_dim
    log(f"      Compressor(ratio={ratio}, overlap={self.overlap}, head_dim={self.head_dim}, rotate={self.rotate})", 0)
    log(f"        input: {T(x)}", 0)

    result = _orig_compressor_forward(self, x, start_pos)

    if result is not None:
        log(f"        output (compressed KV): {T(result)}", 0)
        log(f"        → {result.shape[1]} compressed positions (from {seqlen} tokens at ratio={ratio})", 0)
    else:
        log(f"        → accumulating state (no output yet, pos%ratio != 0)", 0)
    return result

Compressor.forward = _traced_compressor_forward


# ── Indexer.forward ────────────────────────────────────────────────────────────
_orig_indexer_forward = Indexer.forward

def _traced_indexer_forward(self, x, qr, start_pos, offset):
    bsz, seqlen, _ = x.size()
    ratio = self.compress_ratio
    rd = self.rope_head_dim
    end_pos = start_pos + seqlen

    log(f"      Indexer(topk={self.index_topk}, n_heads={self.n_local_heads}, head_dim={self.head_dim})", 0)
    log(f"        input x: {T(x)}, qr: {T(qr)}", 0)

    # Trace Q projection inside indexer
    q = self.wq_b(qr)
    log(f"        wq_b({qr.shape[-1]}→{self.n_local_heads*self.head_dim}): {T(q)}", 0)
    q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
    log(f"        unflatten: {T(q)}  [B, S, n_heads={self.n_local_heads}, head_dim={self.head_dim}]", 0)

    # The actual forward handles the rest
    result = _orig_indexer_forward(self, x, qr, start_pos, offset)
    log(f"        topk_idxs: {T(result)}  [B, S, topk={result.shape[-1]}]", 0)
    return result

Indexer.forward = _traced_indexer_forward


# ── MoE.forward ────────────────────────────────────────────────────────────────
_orig_moe_forward = MoE.forward

def _traced_moe_forward(self, x, input_ids):
    shape = x.size()
    x_flat = x.view(-1, self.dim)
    log(f"    MoE: input {T(x)} → flatten to {T(x_flat)}", 0)

    # Gate
    weights, indices = self.gate(x_flat, input_ids.flatten())
    log(f"    Gate({self.gate.score_func}): weights={T(weights)}, indices={T(indices)}", 0)
    log(f"      top-{self.n_activated_experts} experts selected per token", 0)

    result = _orig_moe_forward(self, x, input_ids)
    log(f"    MoE output: {T(result)}", 0)
    return result

MoE.forward = _traced_moe_forward


# ── Transformer.forward ───────────────────────────────────────────────────────
_orig_transformer_forward = Transformer.forward

def _traced_transformer_forward(self, input_ids, start_pos=0):
    bsz, seqlen = input_ids.shape
    log(f"\n{'='*70}", 0)
    log(f"Transformer.forward(input_ids={S(input_ids)}, start_pos={start_pos})", 0)
    log(f"{'='*70}", 0)

    h = self.embed(input_ids)
    log(f"Embed: {T(h)}  [B, S, D]", 0)

    h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
    log(f"HC expand (×{self.hc_mult}): {T(h)}  [B, S, hc, D]", 0)

    for layer in self.layers:
        h = layer(h, start_pos, input_ids)

    log(f"\n{'─'*70}", 0)
    log(f"All layers done. h: {T(h)}", 0)
    log(f"── ParallelHead (HC head → norm → logits) ──", 0)

    logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
    log(f"Logits: {T(logits)}", 0)
    return logits

Transformer.forward = _traced_transformer_forward


# ════════════════════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("DeepSeek V4 Annotated Walkthrough")
print("=" * 70)

args = ModelArgs(n_hash_layers=0)
print(f"\nModelArgs:")
print(f"  dim={args.dim}, n_layers={args.n_layers}, n_heads={args.n_heads}")
print(f"  head_dim={args.head_dim}, rope_head_dim={args.rope_head_dim}")
print(f"  q_lora_rank={args.q_lora_rank}, o_lora_rank={args.o_lora_rank}, o_groups={args.o_groups}")
print(f"  compress_ratios={args.compress_ratios[:args.n_layers]}")
print(f"  n_routed_experts={args.n_routed_experts}, n_activated={args.n_activated_experts}")
print(f"  hc_mult={args.hc_mult}, window_size={args.window_size}")
print(f"  index_topk={args.index_topk}, index_head_dim={args.index_head_dim}")

print("\nLayer map:")
for i, ratio in enumerate(args.compress_ratios[:args.n_layers]):
    if ratio == 0:
        print(f"  Layer {i}: SWA-only (sliding window attention, no compression)")
    elif ratio == 4:
        print(f"  Layer {i}: C4A (compressor+overlap + Lightning Indexer)")
    elif ratio == 128:
        print(f"  Layer {i}: C128A (compressor, no overlap, static indices)")

print("\n--- Building model ---")
import time
t0 = time.time()
model = Transformer(args)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model built in {time.time()-t0:.1f}s, {n_params:.1f}M params")

# ── Prefill ──────────────────────────────────────────────────────────────────
bsz, seq_len = 2, 128
x = torch.randint(0, args.vocab_size, (bsz, seq_len))

print(f"\n{'#'*70}")
print(f"# PREFILL: batch={bsz}, seq_len={seq_len}")
print(f"{'#'*70}")

t0 = time.time()
logits = model(x)
print(f"\nPrefill done in {time.time()-t0:.2f}s, logits: {T(logits)}")

# ── Decode (1 step only for clarity) ─────────────────────────────────────────
print(f"\n{'#'*70}")
print(f"# DECODE: 1 step (position {seq_len})")
print(f"{'#'*70}")

t0 = time.time()
logits = model(x[:, 0:1], seq_len)
print(f"\nDecode done in {time.time()-t0:.2f}s, logits: {T(logits)}")

print(f"\n{'='*70}")
print("Walkthrough complete!")
print(f"{'='*70}")

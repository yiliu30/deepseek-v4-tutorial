#!/usr/bin/env python3
"""
Tiny DeepSeek V4 walkthrough — exercises ALL components:
  - SWA-only layers (compress_ratio=0)
  - C4A layers (compress_ratio=4, with overlapping Compressor + Lightning Indexer)
  - C128A layers (compress_ratio=128, Compressor only)
  - MoE with Gate routing
  - Hyper-Connections (Sinkhorn-normalized)

Uses the original tilelang kernel.py on RTX 5090 (SM100).
"""
import sys
import os
import time

# Ensure our local fast_hadamard_transform stub is found first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(42)

from model import Transformer, ModelArgs, Attention, Compressor, Indexer, MoE, Block


# ── Monkey-patch for verbose tracing ──────────────────────────────────────────

_orig_attn_forward = Attention.forward
_orig_compressor_forward = Compressor.forward
_orig_indexer_forward = Indexer.forward

TRACE = True


def _traced_attn_forward(self, x, start_pos):
    ratio = self.compress_ratio
    layer_type = "SWA-only" if ratio == 0 else f"C{ratio}A"
    has_indexer = hasattr(self, 'indexer') and self.indexer is not None
    if TRACE:
        parts = [layer_type]
        if ratio > 0:
            parts.append(f"Compressor(overlap={'yes' if ratio==4 else 'no'})")
        if has_indexer:
            parts.append("Indexer(FP4+Hadamard)")
        print(f"  Layer {self.layer_id:2d} [{' + '.join(parts)}]  x={tuple(x.shape)}  start_pos={start_pos}")
    return _orig_attn_forward(self, x, start_pos)


def _traced_compressor_forward(self, x, start_pos):
    if TRACE:
        print(f"    → Compressor(ratio={self.compress_ratio}, overlap={self.overlap}, head_dim={self.head_dim})")
    result = _orig_compressor_forward(self, x, start_pos)
    if result is not None and TRACE:
        print(f"      compressed KV: {tuple(result.shape)}")
    elif result is None and TRACE:
        print(f"      (no compression this step — accumulating state)")
    return result


def _traced_indexer_forward(self, x, qr, start_pos, offset):
    if TRACE:
        print(f"    → Indexer(topk={self.index_topk}, heads={self.n_local_heads}, head_dim={self.head_dim})")
    result = _orig_indexer_forward(self, x, qr, start_pos, offset)
    if TRACE:
        print(f"      topk_idxs: {tuple(result.shape)}")
    return result


Attention.forward = _traced_attn_forward
Compressor.forward = _traced_compressor_forward
Indexer.forward = _traced_indexer_forward


# ── Build tiny model ──────────────────────────────────────────────────────────

print("=" * 70)
print("DeepSeek V4 Tiny Walkthrough on RTX 5090")
print("=" * 70)

args = ModelArgs(n_hash_layers=0)
print(f"\nModelArgs: dim={args.dim}, n_layers={args.n_layers}, n_heads={args.n_heads}")
print(f"  head_dim={args.head_dim}, rope_head_dim={args.rope_head_dim}")
print(f"  compress_ratios={args.compress_ratios}")
print(f"  n_routed_experts={args.n_routed_experts}, n_activated={args.n_activated_experts}")
print(f"  hc_mult={args.hc_mult}, window_size={args.window_size}")
print(f"  index_topk={args.index_topk}, index_head_dim={args.index_head_dim}")

print("\n--- Layer type map ---")
for i, ratio in enumerate(args.compress_ratios[:args.n_layers]):
    if ratio == 0:
        print(f"  Layer {i}: SWA-only")
    elif ratio == 4:
        print(f"  Layer {i}: C4A (compressor+overlap + indexer)")
    elif ratio == 128:
        print(f"  Layer {i}: C128A (compressor, no overlap)")
    else:
        print(f"  Layer {i}: C{ratio}A")

print("\n--- Building model ---")
t0 = time.time()
model = Transformer(args)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model built in {time.time()-t0:.1f}s, {n_params:.1f}M params")

# ── Prefill ───────────────────────────────────────────────────────────────────

bsz, seq_len = 2, 128
x = torch.randint(0, args.vocab_size, (bsz, seq_len))

print(f"\n{'='*70}")
print(f"PREFILL: batch={bsz}, seq_len={seq_len}")
print(f"{'='*70}")

t0 = time.time()
logits = model(x)
print(f"\nPrefill done in {time.time()-t0:.2f}s")
print(f"Logits shape: {tuple(logits.shape)}")

# ── Decode ────────────────────────────────────────────────────────────────────

n_decode = 20
print(f"\n{'='*70}")
print(f"DECODE: {n_decode} steps (positions {seq_len} to {seq_len+n_decode-1})")
print(f"{'='*70}")

TRACE = False  # reduce noise — only trace first and last decode step
for i in range(n_decode):
    pos = seq_len + i
    if i == 0 or i == n_decode - 1:
        TRACE = True
        print(f"\n--- Decode step {i} (pos={pos}) ---")
    else:
        TRACE = False

    logits = model(x[:, 0:1], pos)

    if i == 0 or i == n_decode - 1:
        print(f"  Logits: {tuple(logits.shape)}")

TRACE = False
print(f"\n{'='*70}")
print("All components exercised successfully!")
print(f"{'='*70}")

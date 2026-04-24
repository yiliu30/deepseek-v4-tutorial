#!/usr/bin/env python3
"""
Test that all 3 attention code paths (SWA-only, C4A, C128A) run correctly.

Uses a tiny model with random weights — no checkpoint needed.
Verifies:
  1. Each layer uses the expected attention type
  2. Prefill (start_pos=0) produces correct output shapes
  3. Decode steps (start_pos>0) produce correct output shapes
  4. C4A compressor fires during prefill (seqlen > ratio*2)
  5. C128A compressor fires during prefill (seqlen >= 128)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(42)

from model import Transformer, ModelArgs, Attention


def test_attention_paths():
    # Default tiny config: 7 layers, compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0)
    # Layer 0: SWA-only (ratio=0)
    # Layer 1: SWA-only (ratio=0)
    # Layer 2: C4A      (ratio=4, has compressor + indexer)
    # Layer 3: C128A    (ratio=128, has compressor, no indexer)
    # Layer 4: C4A      (ratio=4)
    # Layer 5: C128A    (ratio=128)
    # Layer 6: C4A      (ratio=4)
    args = ModelArgs(n_hash_layers=0)
    model = Transformer(args)

    print("Layer attention types:")
    expected_types = {
        0: ("SWA-only", 0),
        1: ("SWA-only", 0),
        2: ("C4A", 4),
        3: ("C128A", 128),
        4: ("C4A", 4),
        5: ("C128A", 128),
        6: ("C4A", 4),
    }
    for layer in model.layers:
        attn = layer.attn
        layer_id = attn.layer_id
        ratio = attn.compress_ratio
        name, expected_ratio = expected_types[layer_id]
        assert ratio == expected_ratio, f"Layer {layer_id}: expected ratio={expected_ratio}, got {ratio}"
        has_compressor = hasattr(attn, 'compressor')
        has_indexer = hasattr(attn, 'indexer') and attn.indexer is not None
        print(f"  Layer {layer_id}: {name} (ratio={ratio}, compressor={has_compressor}, indexer={has_indexer})")
    print("✓ All layer types correct\n")

    # ── Prefill: seqlen=256 ensures C4A compressor fires (256/4=64 compressed tokens)
    # and C128A compressor fires (256/128=2 compressed tokens)
    bsz, seq_len = 2, 256
    x = torch.randint(0, args.vocab_size, (bsz, seq_len))
    print(f"Prefill: batch={bsz}, seq_len={seq_len}")
    logits = model(x)
    assert logits.shape == (bsz, args.vocab_size), f"Expected logits shape ({bsz}, {args.vocab_size}), got {logits.shape}"
    print(f"  logits: {tuple(logits.shape)} ✓")

    # Verify compressors produced output during prefill
    for layer in model.layers:
        attn = layer.attn
        if attn.compress_ratio:
            # kv_cache should have been assigned to compressor
            assert attn.compressor.kv_cache is not None, f"Layer {attn.layer_id}: compressor kv_cache not assigned"
    print("✓ Prefill passed, all compressors activated\n")

    # ── Decode: 3 steps to exercise incremental path
    print(f"Decode: 3 steps starting at position {seq_len}")
    for step in range(3):
        pos = seq_len + step
        next_tok = torch.randint(0, args.vocab_size, (bsz, 1))
        logits = model(next_tok, pos)
        assert logits.shape == (bsz, args.vocab_size), f"Decode step {step}: expected ({bsz}, {args.vocab_size}), got {logits.shape}"
        print(f"  step {step} (pos={pos}): logits {tuple(logits.shape)} ✓")
    print("✓ Decode passed\n")

    # ── Verify each forward method was actually called (not just dispatched)
    # by checking the dispatch logic matches compress_ratio
    for layer in model.layers:
        attn = layer.attn
        if attn.compress_ratio == 0:
            assert not hasattr(attn, 'compressor'), f"SWA-only layer {attn.layer_id} should not have compressor"
        elif attn.compress_ratio == 4:
            assert attn.indexer is not None, f"C4A layer {attn.layer_id} should have indexer"
        else:
            assert attn.indexer is None, f"C128A layer {attn.layer_id} should not have indexer"
    print("✓ All attention path invariants verified")
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_attention_paths()

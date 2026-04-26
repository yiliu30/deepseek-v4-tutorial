# DeepSeek V4 Tutorial

Study and run DeepSeek V4 (Flash) on consumer GPUs (8× RTX 5090 D).

```bash
I'm DeepSeek 👋
>>> The capital of France is
The capital of France is Paris.<｜end▁of▁sentence｜>
```

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


## More Analysis
### KV Cache
- [DeepSeek V4 KV Design](./docs/ds_v4_kv.md)
[csa](./ds_v4_csa_pipeline.svg)

### Modelling
- [Run DeepSeek V4 Flash on consumer GPUs](./docs/run_ds_flash.md)
- [DeepSeek V4: Pro vs Flash Architecture Comparison](./docs/v4_pro_vs_flash_arch_and_kvcache.md)
- [DeepSeek V4 KV Cache Primitives(SW/Token Compression/Indexer)](./docs/kv_cache_analysis.md)


### Architecture Walkthrough (tiny model)

Run the full model with random weights to trace tensor shapes through all components:

```bash
python walkthrough.py
```

This exercises all V4-specific components: SWA-only, C4A (compressor+indexer), C128A, MoE, Hyper-Connections.



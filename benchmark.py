"""Benchmark DeepSeek V4-Flash on 8x RTX 5090 D.
Measures prefill latency, decode throughput, and GPU memory usage."""
import os
import json
import sys
import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def benchmark(model, prompt_lens, max_new_tokens, n_runs=3):
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    results = []
    for prompt_len in prompt_lens:
        tokens = torch.randint(1, 1000, (1, prompt_len), device="cuda", dtype=torch.long)

        # Warmup
        model.forward(tokens, 0)
        for i in range(min(3, max_new_tokens)):
            model.forward(tokens[:, 0:1], prompt_len + i)
        torch.cuda.synchronize()

        for run in range(n_runs):
            # Prefill
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward(tokens, 0)
            torch.cuda.synchronize()
            t_prefill = time.perf_counter() - t0

            # Decode
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for i in range(max_new_tokens):
                model.forward(tokens[:, 0:1], prompt_len + i)
            torch.cuda.synchronize()
            t_decode = time.perf_counter() - t1

            # Memory
            mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
            mem_reserved = torch.cuda.max_memory_reserved() / 1024**3

            if rank == 0:
                print(f"  Run {run+1}: prefill={t_prefill*1000:.0f}ms, "
                      f"decode={max_new_tokens}/{t_decode:.2f}s = {max_new_tokens/t_decode:.1f} tok/s, "
                      f"mem_alloc={mem_alloc:.1f}GB, mem_reserved={mem_reserved:.1f}GB")

            results.append({
                "prompt_len": prompt_len,
                "run": run,
                "prefill_ms": t_prefill * 1000,
                "decode_tokens": max_new_tokens,
                "decode_s": t_decode,
                "decode_tok_s": max_new_tokens / t_decode,
                "mem_alloc_gb": mem_alloc,
                "mem_reserved_gb": mem_reserved,
            })

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt-lens", type=str, default="128,256,512")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    if rank != 0:
        global print
        print = lambda *_, **__: None

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(42)

    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))
    model_args.max_batch_size = 1
    print(f"Model: {model_args.n_layers} layers, {model_args.n_routed_experts} experts, "
          f"dim={model_args.dim}, expert_dtype={model_args.expert_dtype}")

    with torch.device("cuda"):
        model = Transformer(model_args)
    load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    print("Model loaded")

    # Memory after load
    mem_after_load = torch.cuda.max_memory_allocated() / 1024**3
    print(f"GPU memory after load (rank {rank}): {mem_after_load:.1f}GB")
    torch.cuda.reset_peak_memory_stats()

    prompt_lens = [int(x) for x in args.prompt_lens.split(",")]
    print(f"\nBenchmark: prompt_lens={prompt_lens}, decode_tokens={args.max_new_tokens}, runs={args.n_runs}")
    print("=" * 80)

    for pl in prompt_lens:
        print(f"\nPrompt length: {pl}")
        benchmark(model, [pl], args.max_new_tokens, args.n_runs)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

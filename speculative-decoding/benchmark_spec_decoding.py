#!/usr/bin/env python3
"""
Speculative Decoding Benchmark
Main model : Qwen/Qwen2.5-72B-Instruct  (4-bit NF4, ~36 GB VRAM)  — ungated
Draft model: Qwen/Qwen2.5-7B-Instruct   (~14 GB bf16)              — ungated, same family
Hardware   : 1x H100 80 GB  (~50 GB peak VRAM)
"""

import os, time, json, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAIN_MODEL     = "Qwen/Qwen2.5-72B-Instruct"
DRAFT_MODEL    = "Qwen/Qwen2.5-7B-Instruct"   # same Qwen2.5 family → identical tokenizer
HF_TOKEN       = os.environ.get("HF_TOKEN")    # optional for ungated models; set if you hit rate limits
MAX_NEW_TOKENS = 256
NUM_WARMUP     = 1
NUM_RUNS       = 3

PROMPTS = [
    "Explain how transformers work in deep learning, focusing on the self-attention mechanism.",
    "Write a Python implementation of a red-black tree with insert and search operations.",
    "What are the philosophical implications of the Turing test for machine consciousness?",
    "Describe the step-by-step process of mRNA vaccine development and how immune response is triggered.",
    "Analyse the macroeconomic causes and effects of the 2008 global financial crisis.",
]


def load_models():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Both models share the Qwen2.5 tokenizer — load once from the main model.
    print(f"Loading tokenizer   : {MAIN_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL, token=HF_TOKEN)

    print(f"Loading main model  : {MAIN_MODEL}")
    main_model = AutoModelForCausalLM.from_pretrained(
        MAIN_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    main_model.eval()
    print(f"  VRAM after main model : {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print(f"\nLoading draft model : {DRAFT_MODEL}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    draft_model.eval()
    print(f"  VRAM after draft model: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Sanity-check: both models must have identical vocab sizes for speculative decoding.
    main_vocab  = main_model.config.vocab_size
    draft_vocab = draft_model.config.vocab_size
    assert main_vocab == draft_vocab, (
        f"Vocab size mismatch — main={main_vocab}, draft={draft_vocab}. "
        "Speculative decoding requires identical tokenizers."
    )
    print(f"\n  Vocab size check passed: {main_vocab} tokens (both models)")

    return tokenizer, main_model, draft_model


def run_inference(model, tokenizer, prompt, draft_model=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        if draft_model is not None:
            out = model.generate(
                **inputs,
                assistant_model=draft_model,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        else:
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_new = out.shape[1] - input_len
    return elapsed, n_new


def benchmark(model, tokenizer, prompts, draft_model=None):
    tag = "SPECULATIVE DECODING (main + draft)" if draft_model else "BASELINE (main model only)"
    print(f"\n{'='*70}")
    print(f"  {tag}")
    print(f"{'='*70}")

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n  Prompt {i+1}: {prompt[:60]}...")

        for _ in range(NUM_WARMUP):
            run_inference(model, tokenizer, prompt, draft_model)

        latencies, tps_list, tok_counts = [], [], []
        for r in range(NUM_RUNS):
            lat, n_tok = run_inference(model, tokenizer, prompt, draft_model)
            latencies.append(lat)
            tps_list.append(n_tok / lat)
            tok_counts.append(n_tok)
            print(f"    run {r+1}: {n_tok} tokens in {lat:.2f}s  ({n_tok/lat:.1f} tok/s)")

        result = {
            "prompt_snippet": prompt[:70],
            "avg_latency_s": round(statistics.mean(latencies), 3),
            "p50_latency_s": round(statistics.median(latencies), 3),
            "avg_tokens_per_sec": round(statistics.mean(tps_list), 2),
            "tokens_generated": int(statistics.mean(tok_counts)),
        }
        results.append(result)

    return results


def main():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")
    print(f"PyTorch : {torch.__version__}\n")

    tokenizer, main_model, draft_model = load_models()

    baseline_results = benchmark(main_model, tokenizer, PROMPTS)
    spec_results     = benchmark(main_model, tokenizer, PROMPTS, draft_model=draft_model)

    print(f"\n\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  Main model : {MAIN_MODEL}")
    print(f"  Draft model: {DRAFT_MODEL}  (same Qwen2.5 family, shared tokenizer)")
    print(f"  Max tokens : {MAX_NEW_TOKENS}  |  Runs per prompt: {NUM_RUNS}")
    print(f"{'='*70}")
    print(f"  {'Prompt':<50} {'Base':>8} {'Spec':>8} {'Speedup':>8}")
    print(f"  {'':50} {'tok/s':>8} {'tok/s':>8} {'':>8}")
    print(f"  {'-'*76}")

    speedups = []
    for b, s in zip(baseline_results, spec_results):
        sp = s["avg_tokens_per_sec"] / b["avg_tokens_per_sec"]
        speedups.append(sp)
        print(f"  {b['prompt_snippet'][:50]:<50} {b['avg_tokens_per_sec']:>8.1f} "
              f"{s['avg_tokens_per_sec']:>8.1f} {sp:>7.2f}x")

    avg_sp = statistics.mean(speedups)
    print(f"  {'-'*76}")
    print(f"  {'AVERAGE SPEEDUP':<50} {'':>8} {'':>8} {avg_sp:>7.2f}x")
    print(f"{'='*70}\n")

    output = {
        "gpu": torch.cuda.get_device_name(0),
        "main_model": MAIN_MODEL,
        "draft_model": DRAFT_MODEL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_runs": NUM_RUNS,
        "baseline": baseline_results,
        "speculative_decoding": spec_results,
        "avg_speedup_x": round(avg_sp, 3),
    }

    out_path = os.path.expanduser("~/benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()

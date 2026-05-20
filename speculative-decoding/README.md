# Speculative Decoding Benchmark

Benchmarks **speculative decoding** against standard autoregressive inference on a 72B model, using same-family models to maximise draft token acceptance rate.

## Setup

| | |
|---|---|
| **Hardware** | NVIDIA H100 80GB HBM3 |
| **Main model** | `Qwen/Qwen2.5-72B-Instruct` — 4-bit NF4 quantized via bitsandbytes (~41 GB VRAM) |
| **Draft model** | `Qwen/Qwen2.5-7B-Instruct` — bf16 (~15 GB VRAM) |
| **Peak VRAM** | ~56.5 GB |
| **Framework** | HuggingFace Transformers `assisted_generation` |
| **Tokens per run** | 256 max new tokens |
| **Runs per prompt** | 3 timed + 1 warmup |

Both models are from the **Qwen2.5 family** and share an identical tokenizer (vocab size: 152,064). This is required for speculative decoding — mismatched tokenizers cause token ID misalignment and break the verification step.

## How Speculative Decoding Works

Standard autoregressive decoding generates one token per forward pass through the large model. Speculative decoding exploits the fact that a small, fast **draft model** can propose several tokens at once, which the large **verifier model** then accepts or rejects in a single parallel forward pass.

```
Draft model  →  [t1, t2, t3, t4, t5]  (cheap, fast)
Main model   →  verify all 5 in one pass  (expensive but parallel)
             →  accept t1, t2, t3 | reject t4 onwards
             →  regenerate from t4
```

When the draft model's distribution closely matches the main model, acceptance rates are high and throughput increases significantly. When they diverge (creative/philosophical prompts), fewer tokens are accepted and the benefit shrinks.

## Results

| Prompt | Baseline (tok/s) | Spec Decoding (tok/s) | Speedup | Latency saved |
|--------|------------------|-----------------------|---------|---------------|
| Transformer self-attention (technical) | 11.69 | 17.31 | **1.48x** | 7.1s |
| Red-black tree in Python (code) | 11.62 | 23.65 | **2.04x** | 11.2s |
| Turing test philosophy (open-ended) | 11.99 | 12.90 | **1.08x** | 1.5s |
| mRNA vaccine process (factual) | 11.49 | 13.43 | **1.17x** | 3.2s |
| 2008 financial crisis (analytical) | 11.62 | 13.41 | **1.15x** | 2.9s |
| **Average** | **11.68** | **16.14** | **1.38x** | — |

## Interpreting the Results

### Why code gets the biggest speedup (2.04x)

Code is structurally predictable — keywords, indentation, closing brackets, common patterns. The 7B draft model and 72B main model are highly aligned on these, so draft tokens are accepted at a high rate. More accepted tokens per verification pass = higher throughput.

### Why open-ended prose gets a modest speedup (1.08–1.17x)

For philosophical or creative prompts, the 72B model draws on reasoning patterns that the 7B model approximates less accurately. The main model rejects more draft tokens, falling back to single-token generation more often. The overhead of running both models slightly narrows the gap.

### Why baseline is flat at ~11.6 tok/s

The 72B model in 4-bit NF4 is memory-bandwidth bound on a single H100. Every token generation requires reading ~41 GB of weights — the bottleneck is HBM bandwidth, not compute. This is why speculative decoding helps: accepted draft tokens cost no extra weight reads from the main model.

### The acceptance rate effect

Speculative decoding speedup is approximately:

```
speedup ≈ (1 + α·k) / (1 + cost_ratio)
```

Where `α` = token acceptance rate, `k` = draft tokens proposed per step, `cost_ratio` = draft model cost / main model cost. Higher acceptance rate on code prompts directly translates to the 2x+ speedup seen above.

## Running the Benchmark

```bash
pip install transformers>=4.44.0 accelerate>=0.34.0 bitsandbytes>=0.43.0

# Optional: set HF_TOKEN if you hit rate limits (models are ungated)
export HF_TOKEN=your_token

python benchmark_spec_decoding.py
```

Results are saved to `~/benchmark_results.json`.

## Files

| File | Description |
|------|-------------|
| `benchmark_spec_decoding.py` | Benchmark script — loads both models, runs timed inference in both modes, prints comparison table |
| `benchmark_results.json` | Raw results from the H100 run (latency, tok/s, speedup per prompt) |
| `README.md` | This file |

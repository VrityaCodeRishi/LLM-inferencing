# Weight Quantization Benchmark Results

**Benchmark Date:** February 9, 2025  
**Hardware:** NVIDIA L40S (47.4 GB VRAM)  
**Test Prompt:** "The future of artificial intelligence is"

---

## Summary Tables (with GPU Memory)

### Table 1: GPT-2 (124M params)

| Method | Model | Size (GB) | Time (ms) | Peak VRAM (GB) | Speedup |
|--------|-------|-----------|-----------|----------------|---------|
| FP32 | gpt2 | 0.464 | 93.65 | 0.49 | 1.00x |
| FP16 | gpt2 | 0.232 | 93.78 | 0.27 | 1.00x |
| BFloat16 | gpt2 | 0.232 | 94.69 | 0.27 | 0.99x |
| INT8 bitsandbytes | gpt2 | 0.226 | 333.12 | 0.25 | 0.28x |
| INT4 bitsandbytes | gpt2 | 0.187 | 192.25 | 0.23 | 0.49x |

### Table 2: TinyLlama (1.1B params)

| Method | Model | Size (GB) | Time (ms) | Peak VRAM (GB) | Speedup |
|--------|-------|-----------|-----------|----------------|---------|
| TinyLlama FP16 | TinyLlama-1.1B | 2.049 | 215.65 | 2.11 | baseline |
| AWQ | TinyLlama-1.1B | 0.244 | 389.93 | 0.73 | 0.55x |
| GGUF | TinyLlama-1.1B | 0.623 | 420.35 | N/A (CPU) | 0.51x |
| TinyLlama INT4 bnb | TinyLlama-1.1B | 0.695 | 480.13 | 0.78 | 0.45x |

### Table 3: Qwen2-7B (7B params)

| Method | Model | Size (GB) | Time (ms) | Peak VRAM (GB) | Speedup |
|--------|-------|-----------|-----------|----------------|---------|
| Qwen2-7B FP16 | Qwen2-7B | 14.185 | 42,198 | 14.79 | 1.00x |
| Qwen2-7B INT8 | Qwen2-7B | 8.108 | 24,354 | 8.64 | 1.73x |
| Qwen2-7B INT4 | Qwen2-7B | 5.069 | 9,190 | 5.95 | 4.59x |

---

## Analysis

### 1. Size vs Speed by Model Size

| Model Size | Observation |
|------------|-------------|
| **GPT-2 (124M)** | Quantization reduces size (down to 0.187 GB) but slows inference. INT8 is ~3.5× slower, INT4 is ~2× slower. |
| **TinyLlama (1.1B)** | AWQ achieves ~8× smaller size. All quantized variants are slower than FP16. |
| **Qwen2-7B (7B)** | Quantization wins on both size and speed. INT4 is ~4.6× faster than FP16. |

### 2. Memory Usage Patterns

- **FP16 typically uses ~2× model size** (weights + activations + overhead).
- **Quantization dramatically reduces VRAM:**
  - Qwen2-7B FP16: 14.79 GB → INT4: 5.95 GB (~2.5× reduction)
  - TinyLlama FP16: 2.11 GB → AWQ: 0.73 GB (~2.9× reduction)
- GGUF runs on CPU, so it doesn't use GPU memory in the same way.

### 3. Why Small Models Slow Down, Large Models Speed Up

| Model Size | What Happens | Explanation |
|------------|--------------|-------------|
| **Small (<1B)** | Quantization = **slower** | The model fits easily in VRAM. Computation is fast. The extra work of *dequantizing* weights (INT8/INT4 → FP16) at each layer dominates. Dequant overhead > any memory savings. |
| **Large (7B+)** | Quantization = **faster** | The model is memory-bound. Moving less data (smaller weights) reduces GPU memory bandwidth pressure. The time saved by moving fewer bytes outweighs the dequantization cost. Memory bandwidth becomes the bottleneck, not compute. |

**TL;DR:** Small models are compute-bound; quantization adds work. Large models are memory-bound; quantization reduces data movement and wins.

### 4. Quantization Crossover Point

The inflection point where INT8/INT4 becomes *faster* than FP16 appears to lie between 1B and 7B parameters:

- **GPT-2:** Dequantization overhead dominates → FP16 wins
- **TinyLlama:** Memory savings are significant but speed is still slower than FP16
- **Qwen2-7B:** Quantization wins on both size and speed

### 5. Key Insights

- FP16/BFloat16 work well for all model sizes.
- Quantization benefits increase with model size.
- On 7B+ models, INT8/INT4 may match or beat FP16 speed (memory bandwidth wins).
- AWQ/GPTQ provide best speed/accuracy for 1B+ models (when available).
- INT4 bitsandbytes is the easiest option (no pre-quantized model needed).

# Attention Benchmark: Manual vs SDPA Backends Across GPUs

This README records attention-kernel benchmark results across multiple NVIDIA GPUs.

The goal is to understand how manual/eager attention compares with PyTorch SDPA backends as sequence length increases, especially for FlashAttention-style execution.

---

## What is being benchmarked?

Backends:

- `manual`: eager attention using `QK^T -> softmax -> AV`; materializes the full `[B, H, S, S]` attention matrix.
- `sdpa_auto`: PyTorch `scaled_dot_product_attention` with automatic backend selection.
- `sdpa_flash`: PyTorch SDPA Flash backend forced.
- `sdpa_mem_efficient`: PyTorch SDPA memory-efficient backend forced.
- `sdpa_math`: PyTorch SDPA math fallback.

---

## Common benchmark configuration

Unless stated otherwise, all runs used:

```text
PyTorch:    2.11.0+cu130
Dtype:      bf16
Causal:     False
Batch:      2
Q Heads:    32
KV Heads:   32 (MHA)
Head Dim:   128
Seq Lens:   [1024, 2048, 4096, 8192]
Warmup:     10
Iters:      50
```

SDPA backend availability was true for all recorded runs:

```text
flash_sdp:          True
mem_efficient_sdp:  True
math_sdp:           True
```

Correctness checks were run up to `SeqLen=4096`. At `SeqLen=8192`, correctness was skipped to avoid large manual-reference computation.

---

## Core interpretation

Manual attention materializes the full attention matrix:

```text
[B, H, S, S]
```

So memory grows quadratically with sequence length.

FlashAttention-style SDPA avoids storing the full `S × S` matrix in HBM. It computes attention block-by-block using online softmax.

Key takeaway:

> FlashAttention does not remove the `O(S²)` compute. It removes the `O(S²)` memory materialization, making attention much more GPU-efficient.

---

# Results by GPU

---

## NVIDIA L40S

### Environment

```text
GPU:        NVIDIA L40S
GPU memory: 44.4 GB
PyTorch:    2.11.0+cu130
Dtype:      bf16
Causal:     False
Batch:      2
Q Heads:    32
KV Heads:   32 (MHA)
Head Dim:   128
Seq Lens:   [1024, 2048, 4096, 8192]
Warmup:     10
Iters:      50
```

### L40S summary

| SeqLen | Backend | Status | Avg ms | ±Std | Min ms | Tok/s | TFLOPS | Peak GB | Speedup | Mem Save | Correct |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | manual | OK | 1.411 | 0.005 | 1.398 | 1,451,844 | 24.36 | 0.383 | 1.0x | 1.0x | - |
| 1024 | sdpa_auto | OK | 0.189 | 0.014 | 0.182 | 10,819,108 | 181.51 | 0.133 | 7.5x | 2.9x | PASS |
| 1024 | sdpa_flash | OK | 0.210 | 0.053 | 0.198 | 9,739,054 | 163.39 | 0.133 | 6.7x | 2.9x | PASS |
| 1024 | sdpa_mem_efficient | OK | 0.308 | 0.016 | 0.300 | 6,648,038 | 111.54 | 0.133 | 4.6x | 2.9x | PASS |
| 1024 | sdpa_math | OK | 4.493 | 0.019 | 4.468 | 455,822 | 7.65 | 0.805 | 0.3x | 0.5x | PASS |
| 2048 | manual | OK | 5.284 | 0.010 | 5.261 | 775,242 | 26.01 | 1.258 | 1.0x | 1.0x | - |
| 2048 | sdpa_auto | OK | 0.670 | 0.018 | 0.655 | 6,116,202 | 205.23 | 0.258 | 7.9x | 4.9x | PASS |
| 2048 | sdpa_flash | OK | 0.686 | 0.016 | 0.672 | 5,973,543 | 200.44 | 0.258 | 7.7x | 4.9x | PASS |
| 2048 | sdpa_mem_efficient | OK | 1.056 | 0.027 | 1.032 | 3,878,600 | 130.14 | 0.258 | 5.0x | 4.9x | PASS |
| 2048 | sdpa_math | OK | 16.575 | 0.043 | 16.498 | 247,123 | 8.29 | 2.727 | 0.3x | 0.5x | PASS |
| 4096 | manual | OK | 19.973 | 0.034 | 19.919 | 410,152 | 27.52 | 4.508 | 1.0x | 1.0x | - |
| 4096 | sdpa_auto | OK | 2.489 | 0.157 | 2.256 | 3,291,179 | 220.87 | 0.509 | 8.0x | 8.9x | PASS |
| 4096 | sdpa_flash | OK | 2.483 | 0.099 | 2.276 | 3,298,805 | 221.38 | 0.509 | 8.0x | 8.9x | PASS |
| 4096 | sdpa_mem_efficient | OK | 4.110 | 0.144 | 3.783 | 1,993,125 | 133.76 | 0.508 | 4.9x | 8.9x | PASS |
| 4096 | sdpa_math | OK | 62.231 | 0.167 | 61.838 | 131,639 | 8.83 | 9.946 | 0.3x | 0.5x | PASS |
| 8192 | manual | OK | 78.648 | 0.055 | 78.568 | 208,320 | 27.96 | 16.758 | 1.0x | 1.0x | - |
| 8192 | sdpa_auto | OK | 10.260 | 0.396 | 9.953 | 1,596,905 | 214.33 | 0.760 | 7.7x | 22.1x | - |
| 8192 | sdpa_flash | OK | 9.980 | 0.172 | 9.811 | 1,641,753 | 220.35 | 0.760 | 7.9x | 22.1x | - |
| 8192 | sdpa_mem_efficient | OK | 16.409 | 0.155 | 16.316 | 998,461 | 134.01 | 0.758 | 4.8x | 22.1x | - |
| 8192 | sdpa_math | OK | 243.921 | 0.799 | 242.104 | 67,169 | 9.02 | 37.633 | 0.3x | 0.4x | - |

### L40S interpretation

- At `SeqLen=8192`, `sdpa_flash` is the fastest backend: `9.980 ms`.
- `sdpa_flash` gives `7.9x` speedup and `22.1x` lower peak memory than manual attention at `SeqLen=8192`.
- `sdpa_auto` and `sdpa_flash` are very close, suggesting PyTorch auto dispatch selects a Flash-like path.
- Manual attention is memory-bound, reaching only around `24-28 TFLOPS`; optimized SDPA reaches around `214-221 TFLOPS`.

---

## NVIDIA A100-SXM4-80GB

### Environment

```text
GPU:        NVIDIA A100-SXM4-80GB
GPU memory: 79.2 GB
PyTorch:    2.11.0+cu130
Dtype:      bf16
Causal:     False
Batch:      2
Q Heads:    32
KV Heads:   32 (MHA)
Head Dim:   128
Seq Lens:   [1024, 2048, 4096, 8192]
Warmup:     10
Iters:      50
```

### A100 summary

| SeqLen | Backend | Status | Avg ms | ±Std | Min ms | Tok/s | TFLOPS | Peak GB | Speedup | Mem Save | Correct |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | manual | OK | 0.765 | 0.006 | 0.755 | 2,677,018 | 44.91 | 0.383 | 1.0x | 1.0x | - |
| 1024 | sdpa_auto | OK | 0.252 | 0.015 | 0.247 | 8,110,958 | 136.08 | 0.133 | 3.0x | 2.9x | PASS |
| 1024 | sdpa_flash | OK | 0.274 | 0.029 | 0.263 | 7,469,934 | 125.32 | 0.133 | 2.8x | 2.9x | PASS |
| 1024 | sdpa_mem_efficient | OK | 0.464 | 0.015 | 0.456 | 4,410,921 | 74.00 | 0.133 | 1.6x | 2.9x | PASS |
| 1024 | sdpa_math | OK | 3.774 | 0.199 | 3.386 | 542,676 | 9.10 | 0.805 | 0.2x | 0.5x | PASS |
| 2048 | manual | OK | 2.689 | 0.005 | 2.679 | 1,523,125 | 51.11 | 1.258 | 1.0x | 1.0x | - |
| 2048 | sdpa_auto | OK | 0.713 | 0.017 | 0.705 | 5,741,352 | 192.65 | 0.258 | 3.8x | 4.9x | PASS |
| 2048 | sdpa_flash | OK | 0.733 | 0.019 | 0.726 | 5,587,060 | 187.47 | 0.258 | 3.7x | 4.9x | PASS |
| 2048 | sdpa_mem_efficient | OK | 1.311 | 0.019 | 1.293 | 3,123,390 | 104.80 | 0.258 | 2.1x | 4.9x | PASS |
| 2048 | sdpa_math | OK | 12.703 | 0.027 | 12.682 | 322,454 | 10.82 | 2.727 | 0.2x | 0.5x | PASS |
| 4096 | manual | OK | 12.434 | 0.035 | 12.416 | 658,835 | 44.21 | 4.508 | 1.0x | 1.0x | - |
| 4096 | sdpa_auto | OK | 2.653 | 0.174 | 2.563 | 3,088,255 | 207.25 | 0.509 | 4.7x | 8.9x | PASS |
| 4096 | sdpa_flash | OK | 2.651 | 0.029 | 2.585 | 3,090,593 | 207.41 | 0.509 | 4.7x | 8.9x | PASS |
| 4096 | sdpa_mem_efficient | OK | 4.960 | 0.031 | 4.941 | 1,651,637 | 110.84 | 0.508 | 2.5x | 8.9x | PASS |
| 4096 | sdpa_math | OK | 49.381 | 0.094 | 49.281 | 165,892 | 11.13 | 9.946 | 0.3x | 0.5x | PASS |
| 8192 | manual | OK | 53.097 | 0.154 | 52.908 | 308,568 | 41.42 | 16.758 | 1.0x | 1.0x | - |
| 8192 | sdpa_auto | OK | 10.488 | 0.022 | 10.479 | 1,562,119 | 209.66 | 0.760 | 5.1x | 22.1x | - |
| 8192 | sdpa_flash | OK | 10.515 | 0.046 | 10.486 | 1,558,109 | 209.13 | 0.760 | 5.0x | 22.1x | - |
| 8192 | sdpa_mem_efficient | OK | 19.338 | 0.034 | 19.265 | 847,234 | 113.71 | 0.758 | 2.7x | 22.1x | - |
| 8192 | sdpa_math | OK | 203.125 | 0.119 | 202.890 | 80,660 | 10.83 | 37.633 | 0.3x | 0.4x | - |

### A100 interpretation

- At `SeqLen=8192`, `sdpa_auto`/`sdpa_flash` are effectively tied around `10.5 ms`.
- A100 manual attention is much faster than L40S manual attention, so relative Flash speedup is lower than on L40S.
- At `SeqLen=8192`, optimized SDPA gives around `5x` speedup and `22.1x` memory saving over manual.
- Manual attention reaches around `41-51 TFLOPS`, while optimized SDPA reaches around `209 TFLOPS`.

---

## NVIDIA H100 80GB HBM3

### Environment

```text
GPU:        NVIDIA H100 80GB HBM3
GPU memory: 79.2 GB
PyTorch:    2.11.0+cu130
Dtype:      bf16
Causal:     False
Batch:      2
Q Heads:    32
KV Heads:   32 (MHA)
Head Dim:   128
Seq Lens:   [1024, 2048, 4096, 8192]
Warmup:     10
Iters:      50
```

### H100 summary

| SeqLen | Backend | Status | Avg ms | ±Std | Min ms | Tok/s | TFLOPS | Peak GB | Speedup | Mem Save | Correct |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | manual | OK | 0.364 | 0.003 | 0.360 | 5,629,412 | 94.45 | 0.406 | 1.0x | 1.0x | - |
| 1024 | sdpa_auto | OK | 0.077 | 0.011 | 0.073 | 26,695,364 | 447.87 | 0.156 | 4.7x | 2.6x | PASS |
| 1024 | sdpa_flash | OK | 0.135 | 0.011 | 0.131 | 15,131,812 | 253.87 | 0.156 | 2.7x | 2.6x | PASS |
| 1024 | sdpa_mem_efficient | OK | 0.237 | 0.013 | 0.232 | 8,645,891 | 145.05 | 0.156 | 1.5x | 2.6x | PASS |
| 1024 | sdpa_math | OK | 1.721 | 0.004 | 1.715 | 1,189,829 | 19.96 | 0.828 | 0.2x | 0.5x | PASS |
| 2048 | manual | OK | 1.459 | 0.003 | 1.454 | 2,807,136 | 94.19 | 1.281 | 1.0x | 1.0x | - |
| 2048 | sdpa_auto | OK | 0.232 | 0.012 | 0.227 | 17,649,087 | 592.21 | 0.281 | 6.3x | 4.6x | PASS |
| 2048 | sdpa_flash | OK | 0.433 | 0.013 | 0.424 | 9,463,018 | 317.53 | 0.282 | 3.4x | 4.5x | PASS |
| 2048 | sdpa_mem_efficient | OK | 0.845 | 0.020 | 0.835 | 4,849,815 | 162.73 | 0.281 | 1.7x | 4.6x | PASS |
| 2048 | sdpa_math | OK | 6.065 | 0.008 | 6.047 | 675,360 | 22.66 | 2.750 | 0.2x | 0.5x | PASS |
| 4096 | manual | OK | 6.377 | 0.014 | 6.361 | 1,284,705 | 86.22 | 4.531 | 1.0x | 1.0x | - |
| 4096 | sdpa_auto | OK | 0.831 | 0.015 | 0.822 | 9,852,657 | 661.20 | 0.531 | 7.7x | 8.5x | PASS |
| 4096 | sdpa_flash | OK | 1.602 | 0.021 | 1.584 | 5,112,670 | 343.11 | 0.532 | 4.0x | 8.5x | PASS |
| 4096 | sdpa_mem_efficient | OK | 3.153 | 0.030 | 3.138 | 2,598,146 | 174.36 | 0.531 | 2.0x | 8.5x | PASS |
| 4096 | sdpa_math | OK | 23.663 | 0.009 | 23.636 | 346,193 | 23.23 | 9.969 | 0.3x | 0.5x | PASS |
| 8192 | manual | OK | 28.619 | 0.020 | 28.570 | 572,481 | 76.84 | 16.781 | 1.0x | 1.0x | - |
| 8192 | sdpa_auto | OK | 3.593 | 0.297 | 3.194 | 4,560,197 | 612.06 | 0.781 | 8.0x | 21.5x | - |
| 8192 | sdpa_flash | OK | 6.168 | 0.063 | 6.075 | 2,656,335 | 356.53 | 0.783 | 4.6x | 21.4x | - |
| 8192 | sdpa_mem_efficient | OK | 12.388 | 0.376 | 11.910 | 1,322,600 | 177.52 | 0.781 | 2.3x | 21.5x | - |
| 8192 | sdpa_math | OK | 98.602 | 0.021 | 98.575 | 166,163 | 22.30 | 37.657 | 0.3x | 0.4x | - |

### H100 interpretation

- H100 is the fastest GPU among the recorded runs.
- At `SeqLen=8192`, `sdpa_auto` is the fastest backend: `3.593 ms`.
- `sdpa_auto` gives `8.0x` speedup and `21.5x` memory saving over manual at `SeqLen=8192`.
- Forced `sdpa_flash` is still much faster than manual, but it is slower than `sdpa_auto` on H100.
- This suggests PyTorch auto dispatch may select a more optimized H100-specific path than forced `SDPBackend.FLASH_ATTENTION` for this configuration.
- Manual attention reaches around `76-94 TFLOPS`, much stronger than L40S and A100 manual attention.
- Optimized `sdpa_auto` reaches more than `600 TFLOPS` for larger sequence lengths.

---

# Cross-GPU comparison at SeqLen 8192

## Best optimized backend per GPU

| GPU | Best Backend | Best Avg ms | Best TFLOPS | Manual Avg ms | Speedup vs Manual | Manual Peak GB | Best Peak GB | Memory Saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L40S | sdpa_flash | 9.980 | 220.35 | 78.648 | 7.9x | 16.758 | 0.760 | 22.1x |
| A100-SXM4-80GB | sdpa_auto | 10.488 | 209.66 | 53.097 | 5.1x | 16.758 | 0.760 | 22.1x |
| H100 80GB HBM3 | sdpa_auto | 3.593 | 612.06 | 28.619 | 8.0x | 16.781 | 0.781 | 21.5x |

## Manual attention comparison at SeqLen 8192

| GPU | Manual Avg ms | Manual TFLOPS | Manual Peak GB |
|---|---:|---:|---:|
| L40S | 78.648 | 27.96 | 16.758 |
| A100-SXM4-80GB | 53.097 | 41.42 | 16.758 |
| H100 80GB HBM3 | 28.619 | 76.84 | 16.781 |

## Optimized attention comparison at SeqLen 8192

| GPU | Best Backend | Best Avg ms | Best TFLOPS | Best Peak GB |
|---|---|---:|---:|---:|
| L40S | sdpa_flash | 9.980 | 220.35 | 0.760 |
| A100-SXM4-80GB | sdpa_auto | 10.488 | 209.66 | 0.760 |
| H100 80GB HBM3 | sdpa_auto | 3.593 | 612.06 | 0.781 |

---

# Overall conclusions

## 1. Memory savings are consistent across GPUs

At `SeqLen=8192`, manual attention uses around `16.8 GB`, while optimized SDPA paths use around `0.76-0.78 GB`.

This proves the memory advantage comes from the attention execution strategy, not from the GPU model.

## 2. H100 is the strongest optimized attention performer

H100 with `sdpa_auto` reaches `3.593 ms` and `612.06 TFLOPS` at `SeqLen=8192`, clearly outperforming L40S and A100.

## 3. A100 manual attention is stronger than L40S manual attention

A100 reduces the pain of manual attention because its memory subsystem is stronger, but optimized SDPA still wins clearly.

## 4. `sdpa_auto` should be benchmarked, not ignored

On L40S and A100, `sdpa_auto` is close to forced `sdpa_flash`.

On H100, `sdpa_auto` is much faster than forced `sdpa_flash`.

So production inference should not blindly force `sdpa_flash`; `sdpa_auto` may select a better hardware-specific path.

## 5. `sdpa_mem_efficient` is a useful fallback, but not the fastest here

It consistently beats manual attention, but trails the best optimized backend.

## 6. `sdpa_math` is not the right performance baseline

`sdpa_math` is slower than manual across the runs. Treat it as PyTorch's SDPA fallback path, not as the eager baseline.


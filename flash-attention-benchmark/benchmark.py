"""
Attention Mechanism Benchmark
=============================

Benchmarks multiple attention implementations on CUDA GPUs:
  - Manual eager attention (materializes full S×S matrix)
  - SDPA auto (PyTorch picks best backend)
  - SDPA FlashAttention forced
  - SDPA Memory-Efficient forced
  - SDPA Math forced

Improvements over naive benchmarks:
  - CUDA events for accurate GPU-side timing (no CPU dispatch noise)
  - Correctness verification (torch.allclose against manual reference)
  - TFLOPS computation for roofline comparison
  - Memory-efficient (xFormers) backend included
  - Per-backend memory isolation with fresh tensors
  - Proper peak memory reset between warmup and timed runs
  - JSON export for downstream analysis / plotting
  - GQA (grouped query attention) support via kv_heads parameter

Usage:
    python attention_benchmark.py
    python attention_benchmark.py --seq-lens 512,1024,2048,4096,8192 --batch 2 --heads 32 --head-dim 128 --dtype bf16 --causal
    python attention_benchmark.py --seq-lens 1024,4096,16384 --kv-heads 8 --json results.json
"""

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# ──────────────────────────────────────────────
# Attention implementations
# ──────────────────────────────────────────────

def manual_attention(q, k, v, causal: bool = False):
    """
    Eager standard attention. Materializes full [B, H, S, S] score matrix.
    Serves as correctness reference and performance baseline.
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


def sdpa_auto(q, k, v, causal: bool = False):
    """PyTorch SDPA with automatic backend selection."""
    enable_gqa = q.shape[1] != k.shape[1]
    return F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=causal, enable_gqa=enable_gqa,
    )


def sdpa_flash(q, k, v, causal: bool = False):
    """Force FLASH_ATTENTION backend."""
    enable_gqa = q.shape[1] != k.shape[1]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=causal, enable_gqa=enable_gqa,
        )


def sdpa_mem_efficient(q, k, v, causal: bool = False):
    """Force EFFICIENT_ATTENTION (xFormers memory-efficient) backend."""
    enable_gqa = q.shape[1] != k.shape[1]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=causal, enable_gqa=enable_gqa,
        )


def sdpa_math(q, k, v, causal: bool = False):
    """Force MATH backend (unfused, like manual but through SDPA dispatch)."""
    enable_gqa = q.shape[1] != k.shape[1]
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=causal, enable_gqa=enable_gqa,
        )


BACKENDS = [
    ("manual", manual_attention),
    ("sdpa_auto", sdpa_auto),
    ("sdpa_flash", sdpa_flash),
    ("sdpa_mem_efficient", sdpa_mem_efficient),
    ("sdpa_math", sdpa_math),
]


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    backend: str
    seq_len: int
    batch: int
    heads: int
    kv_heads: int
    head_dim: int
    causal: bool
    dtype: str
    ok: bool
    avg_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    std_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    tflops: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    qkv_memory_gb: Optional[float] = None
    attn_matrix_gb: Optional[float] = None
    speedup_vs_manual: Optional[float] = None
    memory_saving_vs_manual: Optional[float] = None
    correctness: Optional[str] = None
    error: str = ""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def memory_gb(*tensors) -> float:
    return sum(t.numel() * t.element_size() for t in tensors) / (1024 ** 3)


def estimate_attn_matrix_gb(batch, heads, seq_len, dtype):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return batch * heads * seq_len * seq_len * elem_size / (1024 ** 3)


def attention_flops(batch, heads, kv_heads, seq_len, head_dim, causal=False):
    """
    Compute FLOPs for attention forward pass.

    For MHA (heads == kv_heads):
        Q @ K^T:  2 * B * H * S * S * D
        Attn @ V: 2 * B * H * S * S * D
        Total:    4 * B * H * S^2 * D

    For GQA, K/V have fewer heads, so the matmul dimensions change accordingly.

    Causal masking roughly halves the effective work (lower triangle only).
    """
    # Q @ K^T: each of `heads` Q-heads attends to its KV-head's keys
    qk_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    # Attn @ V: same shape
    av_flops = 2 * batch * heads * seq_len * seq_len * head_dim

    total = qk_flops + av_flops

    if causal:
        total = total // 2  # approximate: only lower triangle

    return total


def check_correctness(
    fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ref_output: torch.Tensor,
    causal: bool,
) -> str:
    """
    Compare fn output against reference. Returns status string.
    Uses relaxed tolerances appropriate for half-precision.
    """
    try:
        out = fn(q, k, v, causal)

        # fp16/bf16 need looser tolerances
        if q.dtype == torch.float16:
            atol, rtol = 1e-2, 1e-2
        else:  # bf16
            atol, rtol = 2e-2, 2e-2

        if torch.allclose(out, ref_output, atol=atol, rtol=rtol):
            return "PASS"

        # Report max deviation for debugging
        max_diff = (out - ref_output).abs().max().item()
        return f"DRIFT(max={max_diff:.4e})"

    except Exception as e:
        return f"ERR({type(e).__name__})"


def make_qkv(batch, heads, kv_heads, seq_len, head_dim, device, dtype):
    """Create Q, K, V tensors. K and V use kv_heads (for GQA support)."""
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    return q, k, v


def expand_kv_for_manual(k, v, heads, kv_heads):
    """
    Expand K, V from [B, kv_heads, S, D] to [B, heads, S, D]
    by repeating each KV head for its group of Q heads.
    Required for manual attention which doesn't handle GQA natively.
    """
    if heads == kv_heads:
        return k, v
    repeats = heads // kv_heads
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    return k, v


# ──────────────────────────────────────────────
# Core benchmark
# ──────────────────────────────────────────────

def benchmark_backend(
    name: str,
    fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmup: int,
    iters: int,
) -> tuple:
    """
    Benchmark a single backend using CUDA events for precise GPU timing.
    Returns (avg_ms, min_ms, max_ms, std_ms, peak_memory_gb).
    """
    cleanup()

    # ── Warmup ──
    for _ in range(warmup):
        _ = fn(q, k, v, causal)
    torch.cuda.synchronize()

    # ── Reset memory tracking after warmup ──
    torch.cuda.reset_peak_memory_stats()

    # ── Timed iterations with CUDA events ──
    timings = []
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        out = fn(q, k, v, causal)
        end_event.record()

        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))  # ms

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Keep reference to output so it's not optimized away
    _ = out.shape

    avg_ms = sum(timings) / len(timings)
    min_ms = min(timings)
    max_ms = max(timings)
    std_ms = (sum((t - avg_ms) ** 2 for t in timings) / len(timings)) ** 0.5

    return avg_ms, min_ms, max_ms, std_ms, peak_gb


def run_benchmark(args):
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dtype_str = args.dtype

    kv_heads = args.kv_heads if args.kv_heads else args.heads
    if args.heads % kv_heads != 0:
        print(f"ERROR: --heads ({args.heads}) must be divisible by --kv-heads ({kv_heads})", file=sys.stderr)
        sys.exit(1)

    is_gqa = kv_heads != args.heads

    # ── System info ──
    gpu_detected = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_label = args.gpu if args.gpu else gpu_detected

    print("=" * 70)
    print("ATTENTION BENCHMARK")
    print("=" * 70)
    print(f"GPU:        {gpu_label}")
    print(f"GPU (auto): {gpu_detected} ({gpu_mem_gb:.1f} GB)")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Dtype:      {dtype_str}")
    print(f"Causal:     {args.causal}")
    print(f"Batch:      {args.batch}")
    print(f"Q Heads:    {args.heads}")
    print(f"KV Heads:   {kv_heads}" + (" (GQA)" if is_gqa else " (MHA)"))
    print(f"Head Dim:   {args.head_dim}")
    print(f"Seq Lens:   {args.seq_lens}")
    print(f"Warmup:     {args.warmup}")
    print(f"Iters:      {args.iters}")
    print()

    print("SDPA Backend Availability")
    print("-" * 35)
    print(f"  flash_sdp:          {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  mem_efficient_sdp:  {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  math_sdp:           {torch.backends.cuda.math_sdp_enabled()}")
    print()

    all_results: List[BenchmarkResult] = []

    for seq_len in args.seq_lens:
        print("─" * 70)
        print(f"  Sequence Length: {seq_len}")
        print("─" * 70)

        torch.manual_seed(args.seed)

        q, k, v = make_qkv(args.batch, args.heads, kv_heads, seq_len, args.head_dim, device, dtype)
        qkv_gb = memory_gb(q, k, v)
        attn_matrix_gb = estimate_attn_matrix_gb(args.batch, args.heads, seq_len, dtype)
        flops = attention_flops(args.batch, args.heads, kv_heads, seq_len, args.head_dim, args.causal)

        print(f"  QKV memory:         {qkv_gb:.3f} GB")
        print(f"  Attn matrix (est):  {attn_matrix_gb:.3f} GB")
        print(f"  Attention FLOPs:    {flops / 1e12:.3f} TFLOPs")
        print()

        # ── Compute reference output from manual attention ──
        if seq_len <= args.correctness_max_seq_len:
            q_ref, k_ref, v_ref = q.clone(), k.clone(), v.clone()
            if is_gqa:
                k_ref_expanded, v_ref_expanded = expand_kv_for_manual(k_ref, v_ref, args.heads, kv_heads)
            else:
                k_ref_expanded, v_ref_expanded = k_ref, v_ref

            try:
                ref_output = manual_attention(q_ref, k_ref_expanded, v_ref_expanded, args.causal)
            except torch.cuda.OutOfMemoryError:
                ref_output = None
                print("  ⚠ Manual attention OOM — correctness checks will be skipped")

            del q_ref, k_ref, v_ref
            if is_gqa:
                del k_ref_expanded, v_ref_expanded
            cleanup()
        else:
            ref_output = None
            print(f"  Skipping correctness check (seq_len {seq_len} > --correctness-max-seq-len {args.correctness_max_seq_len})")

        manual_ms = None
        manual_peak = None

        for name, fn in BACKENDS:
            # For manual attention with GQA, we need expanded K/V
            if name == "manual":
                if is_gqa:
                    k_use, v_use = expand_kv_for_manual(k, v, args.heads, kv_heads)
                else:
                    k_use, v_use = k, v
            else:
                k_use, v_use = k, v

            result = BenchmarkResult(
                backend=name,
                seq_len=seq_len,
                batch=args.batch,
                heads=args.heads,
                kv_heads=kv_heads,
                head_dim=args.head_dim,
                causal=args.causal,
                dtype=dtype_str,
                ok=False,
                qkv_memory_gb=qkv_gb,
                attn_matrix_gb=attn_matrix_gb,
            )

            # ── Correctness check ──
            if ref_output is not None and name != "manual":
                result.correctness = check_correctness(fn, q, k_use, v_use, ref_output, args.causal)
                cleanup()

            # ── Benchmark ──
            try:
                avg_ms, min_ms, max_ms, std_ms, peak_gb = benchmark_backend(
                    name, fn, q, k_use, v_use, args.causal, args.warmup, args.iters,
                )

                result.ok = True
                result.avg_ms = avg_ms
                result.min_ms = min_ms
                result.max_ms = max_ms
                result.std_ms = std_ms
                result.peak_memory_gb = peak_gb
                result.tokens_per_sec = (args.batch * seq_len) / (avg_ms / 1000)
                result.tflops = (flops / 1e12) / (avg_ms / 1000)

                if name == "manual":
                    manual_ms = avg_ms
                    manual_peak = peak_gb

                if manual_ms is not None and avg_ms > 0:
                    result.speedup_vs_manual = manual_ms / avg_ms

                if manual_peak is not None and peak_gb > 0:
                    result.memory_saving_vs_manual = manual_peak / peak_gb

            except torch.cuda.OutOfMemoryError:
                result.error = "OOM"
                cleanup()
            except Exception as e:
                result.error = f"{type(e).__name__}: {str(e)[:120]}"
                cleanup()

            all_results.append(result)

            # ── Print inline ──
            if result.ok:
                corr = f"  [{result.correctness}]" if result.correctness else ""
                speedup = f"  {result.speedup_vs_manual:.1f}x" if result.speedup_vs_manual else ""
                print(
                    f"  {name:<20}  "
                    f"avg {result.avg_ms:>8.3f} ms  "
                    f"(±{result.std_ms:.3f})  "
                    f"min {result.min_ms:>8.3f}  "
                    f"max {result.max_ms:>8.3f}  "
                    f"peak {result.peak_memory_gb:.3f} GB  "
                    f"{result.tflops:.2f} TFLOPS"
                    f"{speedup}{corr}"
                )
            else:
                print(f"  {name:<20}  FAILED: {result.error}")

        print()
        del q, k, v, ref_output
        cleanup()

    # ── Summary table ──
    print_summary_table(all_results)

    # ── JSON export ──
    if args.json:
        export = {
            "meta": {
                "pytorch_version": torch.__version__,
                "gpu": gpu_label,
                "gpu_detected": gpu_detected,
                "gpu_memory_gb": round(gpu_mem_gb, 1),
                "dtype": dtype_str,
                "causal": args.causal,
                "batch": args.batch,
                "heads": args.heads,
                "kv_heads": kv_heads,
                "head_dim": args.head_dim,
                "warmup": args.warmup,
                "iters": args.iters,
            },
            "results": [asdict(r) for r in all_results],
        }
        with open(args.json, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\nResults saved to {args.json}")


# ──────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────

def fmt(x, d=3):
    return f"{x:.{d}f}" if x is not None else "-"


def print_summary_table(results: List[BenchmarkResult]):
    headers = [
        "SeqLen", "Backend", "Status", "Avg ms", "±Std", "Min ms",
        "Tok/s", "TFLOPS", "Peak GB", "Speedup", "Mem Save", "Correct", "Error",
    ]

    rows = []
    for r in results:
        speedup = f"{r.speedup_vs_manual:.1f}x" if r.speedup_vs_manual else "-"
        mem_save = f"{r.memory_saving_vs_manual:.1f}x" if r.memory_saving_vs_manual else "-"
        rows.append([
            str(r.seq_len),
            r.backend,
            "OK" if r.ok else "FAIL",
            fmt(r.avg_ms),
            fmt(r.std_ms),
            fmt(r.min_ms),
            fmt(r.tokens_per_sec, 0) if r.tokens_per_sec else "-",
            fmt(r.tflops, 2),
            fmt(r.peak_memory_gb),
            speedup,
            mem_save,
            r.correctness or "-",
            r.error,
        ])

    col_w = [
        max(len(h), *(len(row[i]) for row in rows))
        for i, h in enumerate(headers)
    ]

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    def fmt_row(cells):
        return "|" + "|".join(f" {cells[i]:<{col_w[i]}} " for i in range(len(headers))) + "|"

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(sep)
    print(fmt_row(headers))
    print(sep)

    prev_seq = None
    for row in rows:
        if prev_seq and row[0] != prev_seq:
            print(sep)
        print(fmt_row(row))
        prev_seq = row[0]

    print(sep)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_seq_lens(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention implementations on CUDA GPUs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seq-lens", type=parse_seq_lens,
                        default=parse_seq_lens("1024,2048,4096,8192"),
                        help="Comma-separated sequence lengths")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=32, help="Number of Q attention heads")
    parser.add_argument("--kv-heads", type=int, default=None,
                        help="Number of KV heads (for GQA). Defaults to --heads (MHA)")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--causal", action="store_true", help="Use causal attention mask")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per backend")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU label for this run (e.g. 'L40S', 'A100-80GB', 'H100'). "
                             "Defaults to auto-detected GPU name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correctness-max-seq-len", type=int, default=4096,
                        help="Only run manual correctness check up to this sequence length")
    parser.add_argument("--json", type=str, default=None, help="Export results to JSON file")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
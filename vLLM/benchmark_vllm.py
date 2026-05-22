import time
import json
import asyncio
import statistics
from dataclasses import dataclass, field, asdict

import httpx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

BASE_URL = "http://localhost:8000"
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_TOKENS = 256

PROMPTS = {
    "tiny": "What is 2+2?",
    "short": "Explain the concept of gravitational waves in simple terms. How were they first detected and why is this discovery important for physics?",
    "medium": (
        "You are an expert systems architect. A startup is building a real-time collaborative "
        "document editor similar to Google Docs. They need to support 10,000 concurrent users "
        "editing the same document with sub-100ms latency for character-level updates. "
        "The system must handle conflict resolution, offline editing with sync, version history, "
        "and rich text formatting. They want to deploy on AWS with a budget of $5,000/month. "
        "Design the complete system architecture including: data model, conflict resolution strategy "
        "(OT vs CRDT), real-time communication layer, storage layer, caching strategy, "
        "and deployment topology. Justify every technical decision with trade-offs."
    ),
    "long": (
        "You are a senior machine learning engineer tasked with designing a production ML pipeline. "
        "The company has 500TB of raw clickstream data from an e-commerce platform with 50 million "
        "monthly active users. The goal is to build a real-time recommendation engine that: "
        "1) Processes user behavior streams with sub-second latency for instant recommendations, "
        "2) Handles cold-start problems for new users and new products using hybrid collaborative "
        "and content-based filtering, 3) Incorporates contextual features like time of day, device "
        "type, location, and browsing session patterns, 4) Supports A/B testing with automatic "
        "traffic allocation and statistical significance detection, 5) Provides explainable "
        "recommendations that comply with GDPR transparency requirements. "
        "The tech stack includes Kafka for streaming, Spark for batch processing, and Kubernetes "
        "for orchestration. The team has experience with PyTorch but is open to other frameworks. "
        "Current infrastructure runs on GCP with a monthly budget of $50,000 for the ML pipeline. "
        "Design the end-to-end architecture covering: data ingestion and feature engineering pipeline, "
        "model architecture (consider transformers, two-tower models, graph neural networks), "
        "training infrastructure with distributed training strategy, feature store design, "
        "model serving with canary deployments and automatic rollback, monitoring and alerting "
        "for data drift and model degradation, and a retraining schedule with CI/CD for ML. "
        "For each component, discuss at least two alternatives you considered and why you chose "
        "the recommended approach. Include estimated latency budgets for each stage of the "
        "inference pipeline."
    ),
}

CONCURRENCY_LEVELS = [1, 2, 4]


@dataclass
class RequestResult:
    prompt_label: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_time_ms: float
    tpot_ms: float
    tokens_per_sec: float


@dataclass
class ConcurrencyResult:
    concurrency: int
    prompt_label: str
    avg_ttft_ms: float
    avg_tpot_ms: float
    avg_total_ms: float
    total_tokens: int
    wall_time_ms: float
    throughput_tps: float


async def send_streaming_request(client: httpx.AsyncClient, prompt: str, label: str) -> RequestResult:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    token_times = []
    first_token_time = None
    start = time.perf_counter()
    completion_tokens = 0
    prompt_tokens = 0

    async with client.stream("POST", f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                completion_tokens += 1
            if "usage" in chunk and chunk["usage"]:
                prompt_tokens = chunk["usage"].get("prompt_tokens", 0)

    end = time.perf_counter()

    if not prompt_tokens:
        prompt_tokens = len(prompt.split()) + 10  # rough estimate

    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    total_ms = (end - start) * 1000

    if len(token_times) > 1:
        inter_token_deltas = [
            (token_times[i] - token_times[i - 1]) * 1000
            for i in range(1, len(token_times))
        ]
        tpot_ms = statistics.mean(inter_token_deltas)
    else:
        tpot_ms = 0

    tps = (completion_tokens / (end - start)) if (end - start) > 0 else 0

    return RequestResult(
        prompt_label=label,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=round(ttft_ms, 2),
        total_time_ms=round(total_ms, 2),
        tpot_ms=round(tpot_ms, 2),
        tokens_per_sec=round(tps, 2),
    )


async def benchmark_single_requests() -> list[RequestResult]:
    print("=" * 60)
    print("Phase 1: Single-request latency by prompt size")
    print("=" * 60)
    results = []
    async with httpx.AsyncClient() as client:
        for label, prompt in PROMPTS.items():
            runs = []
            for i in range(3):
                r = await send_streaming_request(client, prompt, label)
                runs.append(r)
                print(f"  [{label}] run {i+1}/3 — TTFT: {r.ttft_ms:.0f}ms  TPOT: {r.tpot_ms:.1f}ms  TPS: {r.tokens_per_sec:.1f}  tokens: {r.completion_tokens}")

            avg = RequestResult(
                prompt_label=label,
                prompt_tokens=round(statistics.mean(r.prompt_tokens for r in runs)),
                completion_tokens=round(statistics.mean(r.completion_tokens for r in runs)),
                ttft_ms=round(statistics.mean(r.ttft_ms for r in runs), 2),
                total_time_ms=round(statistics.mean(r.total_time_ms for r in runs), 2),
                tpot_ms=round(statistics.mean(r.tpot_ms for r in runs), 2),
                tokens_per_sec=round(statistics.mean(r.tokens_per_sec for r in runs), 2),
            )
            results.append(avg)
            print(f"  [{label}] AVG — TTFT: {avg.ttft_ms:.0f}ms  TPOT: {avg.tpot_ms:.1f}ms  TPS: {avg.tokens_per_sec:.1f}")
            print()
    return results


async def benchmark_concurrency() -> list[ConcurrencyResult]:
    print("=" * 60)
    print("Phase 2: Throughput under concurrency")
    print("=" * 60)
    results = []
    prompt_label = "short"
    prompt = PROMPTS[prompt_label]

    async with httpx.AsyncClient() as client:
        for conc in CONCURRENCY_LEVELS:
            wall_start = time.perf_counter()
            tasks = [send_streaming_request(client, prompt, prompt_label) for _ in range(conc)]
            runs = await asyncio.gather(*tasks)
            wall_end = time.perf_counter()

            wall_ms = (wall_end - wall_start) * 1000
            total_tokens = sum(r.completion_tokens for r in runs)
            throughput = total_tokens / (wall_ms / 1000)

            cr = ConcurrencyResult(
                concurrency=conc,
                prompt_label=prompt_label,
                avg_ttft_ms=round(statistics.mean(r.ttft_ms for r in runs), 2),
                avg_tpot_ms=round(statistics.mean(r.tpot_ms for r in runs), 2),
                avg_total_ms=round(statistics.mean(r.total_time_ms for r in runs), 2),
                total_tokens=total_tokens,
                wall_time_ms=round(wall_ms, 2),
                throughput_tps=round(throughput, 2),
            )
            results.append(cr)
            print(f"  concurrency={conc} — TTFT: {cr.avg_ttft_ms:.0f}ms  TPOT: {cr.avg_tpot_ms:.1f}ms  throughput: {cr.throughput_tps:.1f} tok/s  wall: {cr.wall_time_ms:.0f}ms")
    print()
    return results


def fetch_server_metrics() -> dict:
    resp = httpx.get(f"{BASE_URL}/metrics", timeout=10)
    metrics = {}
    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            metrics[parts[0]] = float(parts[1])
    return metrics


def plot_results(single: list[RequestResult], conc: list[ConcurrencyResult], server_metrics: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("vLLM Benchmark — Mistral-7B-Instruct-v0.2 on NVIDIA L4 (24GB)", fontsize=14, fontweight="bold")

    # 1) TTFT by prompt size
    ax = axes[0, 0]
    labels = [r.prompt_label for r in single]
    ttfts = [r.ttft_ms for r in single]
    bars = ax.bar(labels, ttfts, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token by Prompt Size")
    for bar, val in zip(bars, ttfts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    # 2) TPOT and TPS by prompt size (dual y-axis)
    ax = axes[0, 1]
    x = range(len(labels))
    tpots = [r.tpot_ms for r in single]
    tps_vals = [r.tokens_per_sec for r in single]
    bar_width = 0.35
    bars1 = ax.bar([i - bar_width / 2 for i in x], tpots, bar_width, label="TPOT (ms)", color="#9C27B0")
    ax.set_ylabel("TPOT (ms)", color="#9C27B0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("TPOT & Tokens/sec by Prompt Size")
    ax2 = ax.twinx()
    bars2 = ax2.bar([i + bar_width / 2 for i in x], tps_vals, bar_width, label="Tokens/sec", color="#009688")
    ax2.set_ylabel("Tokens/sec", color="#009688")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # 3) Throughput under concurrency
    ax = axes[1, 0]
    conc_levels = [c.concurrency for c in conc]
    throughputs = [c.throughput_tps for c in conc]
    conc_ttfts = [c.avg_ttft_ms for c in conc]
    bars = ax.bar(conc_levels, throughputs, color="#E91E63", width=0.6)
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Aggregate Throughput (tok/s)")
    ax.set_title("Throughput Scaling with Concurrency")
    ax.set_xticks(conc_levels)
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    # 4) Summary table
    ax = axes[1, 1]
    ax.axis("off")
    kv_usage = server_metrics.get('vllm:kv_cache_usage_perc{engine="0",model_name="mistralai/Mistral-7B-Instruct-v0.2"}', 0)
    prefix_queries = server_metrics.get('vllm:prefix_cache_queries_total{engine="0",model_name="mistralai/Mistral-7B-Instruct-v0.2"}', 0)
    prefix_hits = server_metrics.get('vllm:prefix_cache_hits_total{engine="0",model_name="mistralai/Mistral-7B-Instruct-v0.2"}', 0)
    gen_tokens = server_metrics.get('vllm:generation_tokens_total{engine="0",model_name="mistralai/Mistral-7B-Instruct-v0.2"}', 0)
    prompt_tokens = server_metrics.get('vllm:prompt_tokens_total{engine="0",model_name="mistralai/Mistral-7B-Instruct-v0.2"}', 0)
    preemptions = server_metrics.get('vllm:num_preemptions_total{engine="0",model_name="mistralai/Mistral-7B-Instruct-v0.2"}', 0)

    hit_rate = (prefix_hits / prefix_queries * 100) if prefix_queries > 0 else 0

    table_data = [
        ["KV Cache Usage", f"{kv_usage * 100:.1f}%"],
        ["Prefix Cache Hit Rate", f"{hit_rate:.1f}%"],
        ["Total Prompt Tokens", f"{int(prompt_tokens):,}"],
        ["Total Generated Tokens", f"{int(gen_tokens):,}"],
        ["Preemptions", f"{int(preemptions)}"],
        ["Best TTFT (single)", f"{min(r.ttft_ms for r in single):.0f} ms"],
        ["Best TPOT (single)", f"{min(r.tpot_ms for r in single):.1f} ms"],
        ["Peak Throughput", f"{max(throughputs):.0f} tok/s"],
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECEFF1")
    ax.set_title("Server Metrics Summary", pad=20)

    plt.tight_layout()
    out_path = "/Users/curious_techie/Desktop/llm-inferencing/vLLM/benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    return out_path


async def main():
    print("Warming up with a single request...")
    async with httpx.AsyncClient() as client:
        await send_streaming_request(client, "Hi", "warmup")
    print("Warmup done.\n")

    single_results = await benchmark_single_requests()
    conc_results = await benchmark_concurrency()

    print("Fetching server-side metrics...")
    server_metrics = fetch_server_metrics()

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for r in single_results:
        print(f"  {r.prompt_label:8s} | TTFT: {r.ttft_ms:7.0f}ms | TPOT: {r.tpot_ms:6.1f}ms | TPS: {r.tokens_per_sec:6.1f} | tokens: {r.completion_tokens}")
    print()
    for c in conc_results:
        print(f"  conc={c.concurrency} | TTFT: {c.avg_ttft_ms:7.0f}ms | TPOT: {c.avg_tpot_ms:6.1f}ms | throughput: {c.throughput_tps:6.1f} tok/s")

    results_data = {
        "model": MODEL,
        "gpu": "NVIDIA L4 (24GB)",
        "vllm_version": "0.21.0",
        "max_tokens": MAX_TOKENS,
        "single_request": [asdict(r) for r in single_results],
        "concurrency": [asdict(c) for c in conc_results],
    }
    json_path = "/Users/curious_techie/Desktop/llm-inferencing/vLLM/benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nJSON results saved to {json_path}")

    plot_results(single_results, conc_results, server_metrics)


if __name__ == "__main__":
    asyncio.run(main())

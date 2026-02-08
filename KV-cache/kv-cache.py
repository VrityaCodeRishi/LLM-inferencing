import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
PROMPT = (
    "How learning LLM inferencing can help you as an engineer?\n"
    "Answer clearly with structured bullet points."
)

MAX_NEW_TOKENS = 256
WARMUP = 2
ITERS = 5


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_inputs(tokenizer, model, prompt):
    msgs = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    return inputs


@torch.no_grad()
def bench_generate(gen_fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        _ = gen_fn()
    sync()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for _ in range(iters):
        _ = gen_fn()

    sync()
    t1 = time.time()

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return (t1 - t0) / iters, peak_mb


def load_model(cache_impl: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    model.generation_config.cache_implementation = cache_impl

    print("Cache impl:", model.generation_config.cache_implementation)


    return tokenizer, model


def main():
    assert torch.cuda.is_available(), "Run this on your L40S GPU."

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        do_sample=False,
    )


    tok_dyn, model_dyn = load_model("dynamic")
    inputs_dyn = make_inputs(tok_dyn, model_dyn, PROMPT)

    compiled_dyn = torch.compile(model_dyn, mode="reduce-overhead", fullgraph=True)

    def gen_dynamic():
        return compiled_dyn.generate(**inputs_dyn, **gen_kwargs)

    t_dyn, peak_dyn = bench_generate(gen_dynamic)

    print(f"[dynamic cache + compile] avg latency: {t_dyn:.4f}s")
    print(f"Peak VRAM: {peak_dyn:.0f} MB\n")


    tok_sta, model_sta = load_model("static")
    inputs_sta = make_inputs(tok_sta, model_sta, PROMPT)

    compiled_sta = torch.compile(model_sta, mode="reduce-overhead", fullgraph=True)

    def gen_static():
        return compiled_sta.generate(**inputs_sta, **gen_kwargs)

    t_sta, peak_sta = bench_generate(gen_static)

    print(f"[static cache + compile] avg latency: {t_sta:.4f}s")
    print(f"Peak VRAM: {peak_sta:.0f} MB\n")

    print(f"Speedup (static vs dynamic): {t_dyn / t_sta:.2f}x")
    print(f"VRAM delta (static - dynamic): {peak_sta - peak_dyn:.0f} MB")



if __name__ == "__main__":
    main()
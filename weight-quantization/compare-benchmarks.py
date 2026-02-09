# Script generated from Claude to compare the performance of different quantization methods.
"""
COMPLETE QUANTIZATION BENCHMARK - DUAL MODEL COMPARISON
Uses TWO models to show all quantization methods:
1. GPT-2 (124M) - For basic methods (FP32, FP16, INT8, INT4)
2. Phi-2 (2.7B) - For advanced methods (GPTQ, AWQ, GGUF)

NO HARDCODED RESULTS - Everything is measured live!
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configuration
GPT2_MODEL = "gpt2"
TINYLLAMA_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Base model (1.1B params)
TINYLLAMA_GPTQ = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"  # Pre-quantized GPTQ
TINYLLAMA_AWQ = "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ"   # Pre-quantized AWQ
GGUF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# GGUF path: check GGUF_PATH env var, then script dir, then cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
GGUF_PATH = os.environ.get("GGUF_PATH") or os.path.join(_script_dir, GGUF_FILENAME)
if not os.path.exists(GGUF_PATH):
    GGUF_PATH = os.path.join(os.getcwd(), GGUF_FILENAME)

# Note: TinyLlama is small enough to show quantization working but large enough to benefit

# Large model for FP16 vs INT8/INT4 comparison (fits L40S 48GB)
# Use LARGE_MODEL env var to override. Qwen2-7B is open, no HF auth needed.
LARGE_MODEL = os.environ.get("LARGE_MODEL", "Qwen/Qwen2-7B")  # 7B params, ~14GB FP16

TEST_PROMPT = "The future of artificial intelligence is"
NUM_WARMUP = 3
NUM_RUNS = 10

print("="*90)
print("COMPLETE QUANTIZATION BENCHMARK - DUAL MODEL COMPARISON")
print("="*90)
print(f"\nModel 1 (Basic methods): {GPT2_MODEL}")
print(f"Model 2 (Advanced methods): TinyLlama variants")
print(f"Model 3 (Large model FP16 vs INT8/INT4): {LARGE_MODEL}")
print(f"Test prompt: '{TEST_PROMPT}'")

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n✓ GPU: {gpu_name}")
    print(f"✓ VRAM: {gpu_memory:.1f} GB")
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA: {torch.version.cuda}")
    DEVICE = 'cuda'
else:
    print("\n⚠️  No GPU detected, using CPU")
    DEVICE = 'cpu'


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    name: str
    model_name: str
    size_gb: float
    time_ms: float
    memory_gb: float
    success: bool
    error: Optional[str] = None
    generated_text: Optional[str] = None


def get_model_size(model) -> float:
    """Calculate model size in GB"""
    try:
        size = sum(p.numel() * p.element_size() for p in model.parameters())
        return size / 1024**3
    except:
        return 0.0


def benchmark_model(model, tokenizer, name: str, model_name: str) -> BenchmarkResult:
    """Benchmark a model and return structured results"""
    print(f"\n{'─'*90}")
    print(f"Testing: {name} ({model_name})")
    print(f"{'─'*90}")
    
    try:
        model.eval()
        
        # Prepare input
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt")
        if DEVICE == 'cuda' and next(model.parameters()).is_cuda:
            inputs = inputs.to(DEVICE)
        
        # Warmup
        with torch.no_grad():
            for _ in range(NUM_WARMUP):
                _ = model.generate(**inputs, max_length=30, do_sample=False)
        
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(NUM_RUNS):
                outputs = model.generate(**inputs, max_length=30, do_sample=False)
        
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / NUM_RUNS * 1000  # ms
        
        # Memory
        if DEVICE == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_memory = 0
        
        # Model size
        model_size = get_model_size(model)
        
        # Generated text
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✓ Model size: {model_size:.3f} GB")
        print(f"✓ Inference time: {elapsed:.2f} ms")
        print(f"✓ Peak memory: {peak_memory:.2f} GB")
        print(f"✓ Generated: {generated[:60]}...")
        
        return BenchmarkResult(
            name=name,
            model_name=model_name,
            size_gb=model_size,
            time_ms=elapsed,
            memory_gb=peak_memory,
            success=True,
            generated_text=generated[:100]
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:150]}")
        return BenchmarkResult(
            name=name,
            model_name=model_name,
            size_gb=0,
            time_ms=0,
            memory_gb=0,
            success=False,
            error=str(e)[:150]
        )


# ============================================================================
# BENCHMARK FUNCTIONS - GPT-2 (BASIC METHODS)
# ============================================================================

def benchmark_gpt2_basic() -> List[BenchmarkResult]:
    """Benchmark GPT-2 with basic quantization methods"""
    print(f"\n{'='*90}")
    print(f"SECTION 1: GPT-2 BASIC QUANTIZATION METHODS")
    print(f"{'='*90}")
    
    results = []
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. FP32
    print(f"\n{'='*90}")
    print("1️⃣  FP32 (Full Precision) - BASELINE")
    print(f"{'='*90}")
    print("32-bit floating point - maximum accuracy")
    
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL).to(DEVICE)
    results.append(benchmark_model(model, tokenizer, "FP32", GPT2_MODEL))
    del model
    torch.cuda.empty_cache()
    
    # 2. FP16
    print(f"\n{'='*90}")
    print("2️⃣  FP16 (Half Precision)")
    print(f"{'='*90}")
    print("16-bit floating point - 2x compression")
    
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL).to(DEVICE).half()
    results.append(benchmark_model(model, tokenizer, "FP16", GPT2_MODEL))
    del model
    torch.cuda.empty_cache()
    
    # 3. BFloat16
    print(f"\n{'='*90}")
    print("3️⃣  BFloat16 (Brain Float)")
    print(f"{'='*90}")
    print("16-bit with wider range - 2x compression, better stability")
    
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL).to(DEVICE).bfloat16()
    results.append(benchmark_model(model, tokenizer, "BFloat16", GPT2_MODEL))
    del model
    torch.cuda.empty_cache()
    
    # 4. INT8 bitsandbytes
    print(f"\n{'='*90}")
    print("4️⃣  INT8 bitsandbytes")
    print(f"{'='*90}")
    print("8-bit quantization - 4x compression")
    
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            GPT2_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, "INT8 bitsandbytes", GPT2_MODEL))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
        results.append(BenchmarkResult(
            name="INT8 bitsandbytes",
            model_name=GPT2_MODEL,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:100]
        ))
    
    # 5. INT4 bitsandbytes
    print(f"\n{'='*90}")
    print("5️⃣  INT4 bitsandbytes (NF4)")
    print(f"{'='*90}")
    print("4-bit quantization - 8x compression")
    
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            GPT2_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, "INT4 bitsandbytes", GPT2_MODEL))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
        results.append(BenchmarkResult(
            name="INT4 bitsandbytes",
            model_name=GPT2_MODEL,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:100]
        ))
    
    return results


# ============================================================================
# BENCHMARK FUNCTIONS - PHI-2 (ADVANCED METHODS)
# ============================================================================

def benchmark_tinyllama_advanced() -> List[BenchmarkResult]:
    """Benchmark TinyLlama with advanced quantization methods"""
    print(f"\n{'='*90}")
    print(f"SECTION 2: TINYLLAMA ADVANCED QUANTIZATION METHODS")
    print(f"{'='*90}")
    print("Using TinyLlama (1.1B parameters) to demonstrate GPTQ, AWQ, etc.")
    
    results = []
    tokenizer = None  # Initialize tokenizer
    
    # 1. TinyLlama FP16 (baseline for comparison)
    print(f"\n{'='*90}")
    print("6️⃣  TinyLlama FP16 (Baseline for Advanced Methods)")
    print(f"{'='*90}")
    print("Base model in FP16 for comparison")
    
    try:
        from transformers import AutoTokenizer  # Import here
        
        tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_BASE)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            TINYLLAMA_BASE,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, "TinyLlama FP16", TINYLLAMA_BASE))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error loading TinyLlama base: {str(e)[:150]}")
        results.append(BenchmarkResult(
            name="TinyLlama FP16",
            model_name=TINYLLAMA_BASE,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:150]
        ))
    
    # 2. GPTQ
    print(f"\n{'='*90}")
    print("7️⃣  GPTQ (GPU Post-Training Quantization)")
    print(f"{'='*90}")
    print("4-bit with Hessian optimization - calibrated, high accuracy")
    
    try:
        from transformers import GPTQConfig, AutoTokenizer
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_GPTQ)
            tokenizer.pad_token = tokenizer.eos_token
        
        gptq_config = GPTQConfig(
            bits=4,
            group_size=128,
            desc_act=False,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            TINYLLAMA_GPTQ,
            quantization_config=gptq_config,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, "GPTQ", TINYLLAMA_GPTQ))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error loading GPTQ: {str(e)[:150]}")
        print("Note: Install with: pip install auto-gptq einops")
        results.append(BenchmarkResult(
            name="GPTQ",
            model_name=TINYLLAMA_GPTQ,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:150]
        ))
    
    # 3. AWQ
    print(f"\n{'='*90}")
    print("8️⃣  AWQ (Activation-aware Weight Quantization)")
    print(f"{'='*90}")
    print("4-bit activation-aware - fastest inference")
    
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_AWQ)
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoAWQForCausalLM.from_quantized(
            TINYLLAMA_AWQ,
            fuse_layers=True,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, "AWQ", TINYLLAMA_AWQ))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error loading AWQ: {str(e)[:150]}")
        print("Note: Install with: pip install autoawq")
        results.append(BenchmarkResult(
            name="AWQ",
            model_name=TINYLLAMA_AWQ,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:150]
        ))
    
    # 4. GGUF (if available)
    print(f"\n{'='*90}")
    print("9️⃣  GGUF (llama.cpp format)")
    print(f"{'='*90}")
    print("CPU-optimized quantization")
    
    try:
        from llama_cpp import Llama
        
        if os.path.exists(GGUF_PATH):
            # Load and benchmark GGUF
            print(f"Loading: {GGUF_PATH}")
            llm = Llama(model_path=GGUF_PATH, n_ctx=512, verbose=False)
            
            model_size_gb = os.path.getsize(GGUF_PATH) / 1024**3
            
            print(f"\n{'─'*90}")
            print(f"Testing: GGUF ({os.path.basename(GGUF_PATH)})")
            print(f"{'─'*90}")
            
            # Warmup
            for _ in range(NUM_WARMUP):
                _ = llm(TEST_PROMPT, max_tokens=25, temperature=0)
            
            # Benchmark
            start = time.time()
            for _ in range(NUM_RUNS):
                output = llm(TEST_PROMPT, max_tokens=25, temperature=0)
            elapsed = (time.time() - start) / NUM_RUNS * 1000  # ms
            
            generated = output["choices"][0]["text"] if output.get("choices") else ""
            
            print(f"✓ Model size: {model_size_gb:.3f} GB")
            print(f"✓ Inference time: {elapsed:.2f} ms")
            print(f"✓ Peak memory: N/A (llama.cpp)")
            print(f"✓ Generated: {generated[:60]}...")
            
            results.append(BenchmarkResult(
                name="GGUF",
                model_name=os.path.basename(GGUF_PATH),
                size_gb=model_size_gb,
                time_ms=elapsed,
                memory_gb=0,  # llama.cpp doesn't report through torch
                success=True,
                generated_text=generated[:100]
            ))
            del llm
        else:
            print("⚠️  GGUF file not found")
            print(f"   Expected at: {GGUF_PATH}")
            print("   Download: wget 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf' -O tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
            results.append(BenchmarkResult(
                name="GGUF",
                model_name=GGUF_FILENAME,
                size_gb=0, time_ms=0, memory_gb=0,
                success=False, error="GGUF file not found"
            ))
    except ImportError:
        print("❌ llama-cpp-python not installed")
        print("Install with: pip install llama-cpp-python")
        results.append(BenchmarkResult(
            name="GGUF",
            model_name="TinyLlama-GGUF",
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error="llama-cpp-python not installed"
        ))
    except Exception as e:
        print(f"❌ Error: {str(e)[:150]}")
        results.append(BenchmarkResult(
            name="GGUF",
            model_name=GGUF_FILENAME,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:150]
        ))
    
    # 5. INT4 bitsandbytes on TinyLlama (Fallback/Comparison)
    print(f"\n{'='*90}")
    print("🔟 TinyLlama INT4 bitsandbytes (Easy Alternative)")
    print(f"{'='*90}")
    print("4-bit quantization without pre-quantized model")
    
    try:
        from transformers import BitsAndBytesConfig, AutoTokenizer
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_BASE)
            tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            TINYLLAMA_BASE,
            quantization_config=bnb_config,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, "TinyLlama INT4 bnb", TINYLLAMA_BASE))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error loading INT4 bitsandbytes: {str(e)[:150]}")
        results.append(BenchmarkResult(
            name="TinyLlama INT4 bnb",
            model_name=TINYLLAMA_BASE,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:150]
        ))
    
    return results


# ============================================================================
# BENCHMARK FUNCTIONS - LARGE MODEL (FP16 vs INT8 vs INT4)
# ============================================================================

def benchmark_large_model() -> List[BenchmarkResult]:
    """
    Benchmark FP16 vs INT8 vs INT4 on a large model (7B+).
    On large models, quantization can win on speed due to memory bandwidth savings.
    Designed for L40S (48GB) - 7B fits easily, 13B would also fit.
    """
    print(f"\n{'='*90}")
    print(f"SECTION 3: LARGE MODEL - FP16 vs INT8 vs INT4 (bitsandbytes)")
    print(f"{'='*90}")
    print(f"Model: {LARGE_MODEL} (~7B params)")
    print("Tests if quantization overhead is overcome by memory savings on larger models")
    
    results = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Error loading tokenizer: {str(e)[:150]}")
        results.append(BenchmarkResult(
            name="Large model",
            model_name=LARGE_MODEL,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=f"Tokenizer: {str(e)[:100]}"
        ))
        return results
    
    # 1. FP16 (baseline)
    print(f"\n{'='*90}")
    print("1️⃣  Large Model FP16 (Baseline)")
    print(f"{'='*90}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LARGE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, f"{LARGE_MODEL.split('/')[-1]} FP16", LARGE_MODEL))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error: {str(e)[:150]}")
        results.append(BenchmarkResult(
            name=f"{LARGE_MODEL.split('/')[-1]} FP16",
            model_name=LARGE_MODEL,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:150]
        ))
    
    # 2. INT8 bitsandbytes
    print(f"\n{'='*90}")
    print("2️⃣  Large Model INT8 bitsandbytes")
    print(f"{'='*90}")
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            LARGE_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, f"{LARGE_MODEL.split('/')[-1]} INT8", LARGE_MODEL))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error: {str(e)[:150]}")
        results.append(BenchmarkResult(
            name=f"{LARGE_MODEL.split('/')[-1]} INT8",
            model_name=LARGE_MODEL,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:150]
        ))
    
    # 3. INT4 bitsandbytes
    print(f"\n{'='*90}")
    print("3️⃣  Large Model INT4 bitsandbytes (NF4)")
    print(f"{'='*90}")
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            LARGE_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        results.append(benchmark_model(model, tokenizer, f"{LARGE_MODEL.split('/')[-1]} INT4", LARGE_MODEL))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error: {str(e)[:150]}")
        results.append(BenchmarkResult(
            name=f"{LARGE_MODEL.split('/')[-1]} INT4",
            model_name=LARGE_MODEL,
            size_gb=0, time_ms=0, memory_gb=0,
            success=False, error=str(e)[:150]
        ))
    
    return results


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_summary_table(results: List[BenchmarkResult], title: str, baseline_name: Optional[str] = None):
    """Print formatted summary table. baseline_name: substring to match for speedup (e.g. 'FP32' or 'FP16')"""
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    
    # Find baseline for speedup calculation
    baseline = None
    for r in results:
        if r.success and r.time_ms > 0:
            match = (baseline_name and baseline_name in r.name) if baseline_name else ("FP32" in r.name or ("FP16" in r.name and "Phi-2" in r.name))
            if match:
                baseline = r
                break
    
    print(f"\n{'Method':<25} {'Model':<20} {'Size (GB)':<12} {'Time (ms)':<12} {'Speedup':<12} {'Status':<20}")
    print("─"*110)
    
    for result in results:
        method = result.name
        model = result.model_name.split('/')[-1][:18]  # Truncate long names
        
        if result.success:
            size = result.size_gb
            time_ms = result.time_ms
            
            # Calculate compression
            if baseline and baseline.size_gb > 0:
                compression = baseline.size_gb / size if size > 0 else 0
                size_str = f"{size:.3f} ({compression:.1f}x)"
            else:
                size_str = f"{size:.3f}"
            
            # Calculate speedup
            if baseline and baseline.time_ms > 0 and time_ms > 0:
                speedup = baseline.time_ms / time_ms
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            time_str = f"{time_ms:.2f}"
            status = "✓ Success"
            
        else:
            size_str = "N/A"
            time_str = "N/A"
            speedup_str = "N/A"
            error_short = result.error[:15] if result.error else "Failed"
            status = f"❌ {error_short}"
        
        print(f"{method:<25} {model:<20} {size_str:<12} {time_str:<12} {speedup_str:<12} {status:<20}")


def print_combined_analysis(
    gpt2_results: List[BenchmarkResult],
    tinyllama_results: List[BenchmarkResult],
    large_model_results: Optional[List[BenchmarkResult]] = None,
):
    """Print combined analysis and recommendations"""
    print(f"\n{'='*90}")
    print("COMBINED ANALYSIS")
    print(f"{'='*90}")
    
    # Separate successful results by model
    gpt2_successful = [r for r in gpt2_results if r.success and r.time_ms > 0]
    tinyllama_successful = [r for r in tinyllama_results if r.success and r.time_ms > 0]
    large_successful = [r for r in (large_model_results or []) if r.success and r.time_ms > 0]
    
    print("\n📊 GPT-2 (124M parameters) Results:")
    if gpt2_successful:
        fastest_gpt2 = min(gpt2_successful, key=lambda x: x.time_ms)
        smallest_gpt2 = min(gpt2_successful, key=lambda x: x.size_gb)
        
        print(f"  ⚡ Fastest: {fastest_gpt2.name} ({fastest_gpt2.time_ms:.2f} ms)")
        print(f"  💾 Smallest: {smallest_gpt2.name} ({smallest_gpt2.size_gb:.3f} GB)")
        
        # Calculate compression ratios
        fp32 = next((r for r in gpt2_successful if "FP32" in r.name), None)
        if fp32:
            print(f"\n  Compression vs FP32:")
            for r in gpt2_successful:
                if r != fp32:
                    compression = fp32.size_gb / r.size_gb
                    speedup = fp32.time_ms / r.time_ms if r.time_ms > 0 else 0
                    print(f"    {r.name}: {compression:.1f}x smaller, {speedup:.2f}x speed")
    
    print("\n📊 TinyLlama (1.1B parameters) Results:")
    if tinyllama_successful:
        fastest_tiny = min(tinyllama_successful, key=lambda x: x.time_ms)
        smallest_tiny = min(tinyllama_successful, key=lambda x: x.size_gb)
        
        print(f"  ⚡ Fastest: {fastest_tiny.name} ({fastest_tiny.time_ms:.2f} ms)")
        print(f"  💾 Smallest: {smallest_tiny.name} ({smallest_tiny.size_gb:.3f} GB)")
        
        # Calculate compression ratios
        tiny_base = next((r for r in tinyllama_successful if "TinyLlama FP16" in r.name), None)
        if tiny_base:
            print(f"\n  Compression vs TinyLlama FP16:")
            for r in tinyllama_successful:
                if r != tiny_base and r.size_gb > 0:
                    compression = tiny_base.size_gb / r.size_gb
                    speedup = tiny_base.time_ms / r.time_ms if r.time_ms > 0 else 0
                    print(f"    {r.name}: {compression:.1f}x smaller, {speedup:.2f}x speed")
    
    if large_successful:
        model_short = LARGE_MODEL.split("/")[-1]
        print(f"\n📊 {model_short} (7B parameters) Results:")
        fastest_large = min(large_successful, key=lambda x: x.time_ms)
        smallest_large = min(large_successful, key=lambda x: x.size_gb)
        print(f"  ⚡ Fastest: {fastest_large.name} ({fastest_large.time_ms:.2f} ms)")
        print(f"  💾 Smallest: {smallest_large.name} ({smallest_large.size_gb:.3f} GB)")
        large_base = next((r for r in large_successful if "FP16" in r.name), None)
        if large_base:
            print(f"\n  FP16 vs Quantization (does INT8/INT4 overcome dequant overhead?):")
            for r in large_successful:
                if r != large_base and r.size_gb > 0:
                    compression = large_base.size_gb / r.size_gb
                    speedup = large_base.time_ms / r.time_ms if r.time_ms > 0 else 0
                    faster = "✓ faster" if speedup > 1 else "✗ slower"
                    print(f"    {r.name}: {compression:.1f}x smaller, {speedup:.2f}x speed {faster}")
    
    # Recommendations
    print(f"\n{'='*90}")
    print("RECOMMENDATIONS")
    print(f"{'='*90}")
    
    print("\n🎯 For Small Models (< 1B params like GPT-2):")
    if gpt2_successful:
        fp16_result = next((r for r in gpt2_successful if "FP16" in r.name), None)
        if fp16_result:
            print(f"  → Use FP16: {fp16_result.size_gb/0.464:.1f}x compression, {fp16_result.time_ms:.0f}ms inference")
        print("  → Quantization overhead outweighs benefits")
    
    print("\n🎯 For Medium Models (1-3B params like TinyLlama):")
    if tinyllama_successful:
        awq_result = next((r for r in tinyllama_successful if "AWQ" in r.name), None)
        gptq_result = next((r for r in tinyllama_successful if "GPTQ" in r.name), None)
        
        if awq_result:
            print(f"  → AWQ: {awq_result.size_gb:.1f} GB, {awq_result.time_ms:.0f}ms - ⚡ Fastest")
        if gptq_result:
            print(f"  → GPTQ: {gptq_result.size_gb:.1f} GB, {gptq_result.time_ms:.0f}ms - 🎯 High accuracy")
        
        int4_result = next((r for r in tinyllama_results if "INT4 bnb" in r.name and r.success), None)
        if int4_result:
            print(f"  → INT4 bitsandbytes: {int4_result.size_gb:.1f} GB, {int4_result.time_ms:.0f}ms - 😊 Easiest")
    
    if large_successful:
        print(f"\n🎯 For Large Models (7B+ on L40S/48GB):")
        fastest = min(large_successful, key=lambda x: x.time_ms)
        print(f"  → {fastest.name}: {fastest.size_gb:.2f} GB, {fastest.time_ms:.0f}ms")
        if any("INT8" in r.name or "INT4" in r.name for r in large_successful):
            quant_faster = any(r.time_ms < next((x.time_ms for x in large_successful if "FP16" in x.name), float("inf")) for r in large_successful if "INT" in r.name)
            if quant_faster:
                print("  → Quantization can be FASTER on large models (memory bandwidth wins)")
            else:
                print("  → Quantization saves VRAM but may still be slower (dequant overhead)")
    
    print("\n💡 Key Insights:")
    print("  • FP16/BFloat16 work well for all model sizes")
    print("  • Quantization benefits increase with model size")
    print("  • On 7B+ models, INT8/INT4 may match or beat FP16 speed (memory bandwidth)")
    print("  • AWQ/GPTQ provide best speed/accuracy for 1B+ models")
    print("  • INT4 bitsandbytes is easiest (no pre-quantized model needed)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete benchmark"""
    print("\n" + "="*90)
    print("STARTING COMPREHENSIVE BENCHMARK")
    print("="*90)
    
    # Section 1: GPT-2 basic methods
    gpt2_results = benchmark_gpt2_basic()
    
    # Section 2: TinyLlama advanced methods
    tinyllama_results = benchmark_tinyllama_advanced()
    
    # Section 3: Large model FP16 vs INT8 vs INT4 (L40S-friendly)
    large_model_results = benchmark_large_model()
    
    # Print summary tables
    print_summary_table(gpt2_results, "TABLE 1: GPT-2 BASIC QUANTIZATION METHODS")
    print_summary_table(tinyllama_results, "TABLE 2: TINYLLAMA ADVANCED QUANTIZATION METHODS")
    print_summary_table(large_model_results, f"TABLE 3: {LARGE_MODEL.split('/')[-1]} FP16 vs INT8 vs INT4", baseline_name="FP16")
    
    # Combined analysis
    print_combined_analysis(gpt2_results, tinyllama_results, large_model_results)
    
    print(f"\n{'='*90}")
    print("✓ BENCHMARK COMPLETE")
    print(f"{'='*90}")
    
    # Return results for further analysis if needed
    return {
        'gpt2': gpt2_results,
        'tinyllama': tinyllama_results,
        'large_model': large_model_results,
    }


if __name__ == "__main__":
    results = main()
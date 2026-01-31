#!/usr/bin/env bash

  # Notes to self:
  # - cross_kv_cache_fraction MUST be unset (empty) for decoder-only models (Mistral).
  # - sink_token_length is left unset (empty) to avoid StreamLLM runtime assertion in this stack.
  # - max_attention_window_size set to 8192.
  # - enable_kv_cache_reuse disabled for stable bring-up as there were some problems I am facing; 


set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  mistral7b_triton_inflight_setup.sh [--start] [--skip-convert] [--skip-build] [--skip-repo]

Environment overrides (optional):
  MODEL_DIR              HF model directory (default: /models/Mistral-7B-Instruct-v0.3)
  CKPT_DIR               TRT-LLM checkpoint output dir
  ENGINE_DIR             TRT engine output dir
  MODEL_REPO             Triton model repository output dir
  TRITON_MAX_BATCH_SIZE  (default: 32)
  MAX_INPUT_LEN          (default: 8192)
  TRITON_HTTP_PORT       HTTP port (default: 8000)
  TRITON_GRPC_PORT       GRPC port (default: 8001)
  TRITON_METRICS_PORT    Metrics port (default: 8002)

Examples:
  export MODEL_DIR=/models/Mistral-7B-Instruct-v0.3
  bash /opt/tritonserver/tensorrtllm_backend/scripts/mistral7b_triton_inflight_setup.sh --start
EOF
}

START=0
SKIP_CONVERT=0
SKIP_BUILD=0
SKIP_REPO=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START=1; shift ;;
    --skip-convert) SKIP_CONVERT=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-repo) SKIP_REPO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

BACKEND_ROOT="/opt/tritonserver/tensorrtllm_backend"
FILL="${BACKEND_ROOT}/triton_backend/tools/fill_template.py"
LAUNCH="${BACKEND_ROOT}/triton_backend/scripts/launch_triton_server.py"

MODEL_DIR="${MODEL_DIR:-/models/Mistral-7B-Instruct-v0.3}"
TRITON_MAX_BATCH_SIZE="${TRITON_MAX_BATCH_SIZE:-32}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-8192}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"

CKPT_DIR="${CKPT_DIR:-/opt/tritonserver/tllm_checkpoint_1gpu_mistral_bf16}"
ENGINE_DIR="${ENGINE_DIR:-/opt/tritonserver/trt_engines/mistral7b/bf16/1-gpu}"
MODEL_REPO="${MODEL_REPO:-/opt/tritonserver/model_repo_mistral}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "MODEL_DIR does not exist: $MODEL_DIR" >&2
  exit 1
fi

if [[ ! -f "$FILL" ]]; then
  echo "fill_template.py not found at: $FILL" >&2
  exit 1
fi

echo "== Settings =="
echo "MODEL_DIR=$MODEL_DIR"
echo "CKPT_DIR=$CKPT_DIR"
echo "ENGINE_DIR=$ENGINE_DIR"
echo "MODEL_REPO=$MODEL_REPO"
echo "TRITON_MAX_BATCH_SIZE=$TRITON_MAX_BATCH_SIZE"
echo "MAX_INPUT_LEN=$MAX_INPUT_LEN"
echo "TRITON_HTTP_PORT=$TRITON_HTTP_PORT"
echo "TRITON_GRPC_PORT=$TRITON_GRPC_PORT"
echo "TRITON_METRICS_PORT=$TRITON_METRICS_PORT"
echo

if [[ "$SKIP_CONVERT" -eq 0 ]]; then
  echo "== Converting HF checkpoint -> TRT-LLM checkpoint (BF16) =="
  cd "${BACKEND_ROOT}/examples/models/core/llama"
  python3 convert_checkpoint.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$CKPT_DIR" \
    --dtype bfloat16
  echo
else
  echo "== Skipping convert_checkpoint.py =="
  echo
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "== Building TensorRT engines (BF16) =="
  mkdir -p "$ENGINE_DIR"
  trtllm-build \
    --checkpoint_dir "$CKPT_DIR" \
    --output_dir "$ENGINE_DIR" \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16 \
    --max_input_len "$MAX_INPUT_LEN"
  echo
else
  echo "== Skipping trtllm-build =="
  echo
fi

if [[ "$SKIP_REPO" -eq 0 ]]; then
  echo "== Creating Triton model repo from template =="
  rm -rf "$MODEL_REPO"
  cp -r "${BACKEND_ROOT}/triton_backend/all_models/inflight_batcher_llm" "$MODEL_REPO"

  echo "== Filling template placeholders (config.pbtxt) =="

  python3 "$FILL" -i "${MODEL_REPO}/ensemble/config.pbtxt" \
    "triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:TYPE_FP32"

  python3 "$FILL" -i "${MODEL_REPO}/preprocessing/config.pbtxt" \
    "tokenizer_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:1,max_queue_delay_microseconds:0,max_queue_size:0,add_special_tokens:True,max_num_images:0,multimodal_model_path:,engine_dir:${ENGINE_DIR}"

  python3 "$FILL" -i "${MODEL_REPO}/postprocessing/config.pbtxt" \
    "tokenizer_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:1,skip_special_tokens:True"


  python3 "$FILL" -i "${MODEL_REPO}/tensorrt_llm/config.pbtxt" \
    "triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:false,engine_dir:${ENGINE_DIR},tokenizer_dir:${MODEL_DIR},batching_strategy:inflight_batching,max_beam_width:1,max_queue_delay_microseconds:0,max_queue_size:0,encoder_input_features_data_type:TYPE_FP16,prompt_embedding_table_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,max_tokens_in_paged_kv_cache:${MAX_INPUT_LEN},max_attention_window_size:${MAX_INPUT_LEN},sink_token_length:,batch_scheduler_policy:max_utilization,kv_cache_free_gpu_mem_fraction:0.5,cross_kv_cache_fraction:,kv_cache_host_memory_bytes:0,kv_cache_onboard_blocks:True,exclude_input_in_output:True,enable_kv_cache_reuse:False,normalize_log_probs:False,enable_chunked_context:False,request_stats_max_iterations:10,stats_check_period_ms:1000,cancellation_check_period_ms:1000,iter_stats_max_iterations:0,gpu_device_ids:0,participant_ids:0,num_nodes:1,lora_cache_optimal_adapter_size:0,lora_cache_max_adapter_size:0,lora_cache_gpu_memory_fraction:0,lora_cache_host_memory_bytes:0,lora_prefetch_dir:,decoding_mode:,lookahead_window_size:0,lookahead_ngram_size:0,lookahead_verification_set_size:0,medusa_choices:,eagle_choices:,gpu_weights_percent:100,enable_context_fmha_fp32_acc:False,multi_block_mode:False,cuda_graph_mode:False,cuda_graph_cache_size:0,speculative_decoding_fast_logits:False,guided_decoding_backend:,xgrammar_tokenizer_info_path:,encoder_engine_dir:,enable_trt_overlap:False"

  python3 "$FILL" -i "${MODEL_REPO}/tensorrt_llm_bls/config.pbtxt" \
    "triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:false,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,tensorrt_llm_draft_model_name:,multimodal_encoders_name:,prompt_embedding_table_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"

  echo "== Sanity check: no unfilled template variables remain =="
  if grep -R '\\${' -n "$MODEL_REPO" >/dev/null; then
    printf 'ERROR: Found unfilled template variables in %s (grep '\''\\${'\'').\n' "$MODEL_REPO" >&2
    grep -R '\\${' -n "$MODEL_REPO" | head -50 >&2 || true
    exit 1
  fi
  echo "OK"
  echo
else
  echo "== Skipping model repo generation =="
  echo
fi

echo "== Next steps =="
echo "Start Triton:"
echo " python3 ${LAUNCH} --world_size 1 --no-mpi --force --model_repo ${MODEL_REPO} --http-port ${TRITON_HTTP_PORT} --grpc-port ${TRITON_GRPC_PORT} --metrics-port ${TRITON_METRICS_PORT}"

python3 "${LAUNCH}" --world_size 1 --no-mpi --force --model_repo "${MODEL_REPO}"

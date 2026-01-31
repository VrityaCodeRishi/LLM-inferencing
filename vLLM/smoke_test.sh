#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
MODEL="${VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"

curl -sS "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Say hello world in one short sentence.\"}],
    \"temperature\": 0
  }" | sed 's/\\r//g'

echo

#!/bin/bash
# Evaluate every task in data/ against a self-hosted vLLM server
# (OpenAI-compatible /v1/chat/completions endpoint, e.g. `vllm serve <model>`).
#
# Usage:
#   MODEL=Qwen/Qwen3-8B VLLM_URL=http://localhost:8000 bash scripts/eval_vllm.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:?Set MODEL, e.g. MODEL=Qwen/Qwen3-8B}"
VLLM_URL="${VLLM_URL:?Set VLLM_URL to the vLLM OpenAI-compatible server, e.g. http://localhost:8000}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

echo "Model: $MODEL"
echo "vLLM URL: $VLLM_URL"
echo "Gen kwargs: $GEN_KWARGS"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for jsonl_path in data/*.jsonl; do
    task="$(basename "$jsonl_path" .jsonl)"
    echo "=== $task ==="

    if python evaluation/run.py \
        --model "$MODEL" \
        --model_router vllm \
        --vllm_url "$VLLM_URL" \
        --gen-kwargs "$GEN_KWARGS" \
        --tasks "$task" \
        --async --max-concurrent "$MAX_CONCURRENT"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "Done: $SUCCESS_COUNT succeeded, $FAIL_COUNT failed"
[ "$FAIL_COUNT" -gt 0 ] && exit 1
exit 0

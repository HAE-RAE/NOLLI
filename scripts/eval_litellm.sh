#!/bin/bash
# Evaluate every task in data/data/ against a cloud model via LiteLLM
# (covers OpenAI, Anthropic, Google/Gemini, OpenRouter — anything LiteLLM supports).
#
# Usage:
#   MODEL=anthropic/claude-opus-4-8 bash scripts/eval_litellm.sh
#   MODEL=gemini/gemini-3-flash-preview GEN_KWARGS="temperature=1.0,max_tokens=32768" bash scripts/eval_litellm.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-gemini/gemini-3-flash-preview}"
GEN_KWARGS="${GEN_KWARGS:-temperature=1.0,max_tokens=32768,reasoning_effort=medium}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"

echo "Model: $MODEL"
echo "Mode: liteLLM"
echo "Gen kwargs: $GEN_KWARGS"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for jsonl_path in data/data/*.jsonl; do
    task="$(basename "$jsonl_path" .jsonl)"
    echo "=== $task ==="

    if python evaluation/run.py \
        --model "$MODEL" \
        --model_router litellm \
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

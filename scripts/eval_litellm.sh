#!/bin/bash
# Evaluate every task/difficulty (data auto-downloaded from HAERAE-HUB/NOLLI
# on the HuggingFace hub) against a cloud model via LiteLLM
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

TASKS="$(python -c 'from evaluation.evaluators import list_tasks; print(" ".join(list_tasks()))')"

for base_task in $TASKS; do
  for diff in easy medium hard; do
    task="${base_task}_${diff}"
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
done

echo "Done: $SUCCESS_COUNT succeeded, $FAIL_COUNT failed"
[ "$FAIL_COUNT" -gt 0 ] && exit 1
exit 0

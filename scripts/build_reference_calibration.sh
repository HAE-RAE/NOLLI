#!/usr/bin/env bash
# Rebuild the reference-model re-calibration datasets from their recipe files.
#
#   bash scripts/build_reference_calibration.sh <reference_model> [task ...]
#
# Reads   configs/reference_calibration/<reference_model>/<task>.json
# Writes  data/reference_calibration/<reference_model>/<task>_<tier>.jsonl (+ .config.json)
# With no task arguments, every recipe in the model directory is built (3 tiers each).
#
# Evaluate the result with the local data directory. With --data-dir, task names must
# include the tier (the file read is <data-dir>/<task>.jsonl), e.g.
#   python evaluation/run.py --model openrouter/qwen/qwen3.5-27b --model_router litellm \
#       --gen-kwargs "temperature=0.6,top_p=0.95,top_k=20,max_tokens=32768" \
#       --data-dir data/reference_calibration/qwen3.5-27b \
#       --tasks cipher_en_easy cipher_en_medium cipher_en_hard --async
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "usage: $0 <reference_model> [task ...]" >&2
    exit 2
fi
MODEL="$1"; shift
RECIPE_DIR="configs/reference_calibration/$MODEL"
OUT_DIR="data/reference_calibration/$MODEL"
PYTHON="${PYTHON:-python}"

if [ $# -gt 0 ]; then
    TASKS=("$@")
else
    TASKS=()
    for f in "$RECIPE_DIR"/*.json; do
        [ -e "$f" ] || continue
        TASKS+=("$(basename "$f" .json)")
    done
fi
if [ ${#TASKS[@]} -eq 0 ]; then
    echo "no recipes found in $RECIPE_DIR" >&2
    exit 1
fi

for task in "${TASKS[@]}"; do
    recipe="$RECIPE_DIR/$task.json"
    [ -f "$recipe" ] || { echo "missing recipe: $recipe" >&2; exit 1; }
    for tier in easy medium hard; do
        "$PYTHON" scripts/build_reference_calibration_dataset.py \
            --recipe "$recipe" --tier "$tier" --out-dir "$OUT_DIR"
    done
done

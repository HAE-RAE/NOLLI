#!/usr/bin/env bash
# Regenerate data/data/*.jsonl from the generation/ scripts.
# Each generator writes 100 items/difficulty (300/task) into data/jsonl/,
# except where noted. Copy/move the outputs into data/data/ once verified.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo "NOLLI Dataset Generation (25 tasks)"
echo "============================================"

echo ""; echo "[1/25] array_formula_en..."
python generation/array_formula_en.py --num 100

echo ""; echo "[2/25] array_formula_ko..."
python generation/array_formula_ko.py --num 100

echo ""; echo "[3/25] causal_dag_en..."
python generation/causal_dag_en.py --num 300

echo ""; echo "[4/25] causal_dag_ko..."
python generation/causal_dag_ko.py --num 300

echo ""; echo "[5/25] cipher_en..."
python generation/cipher_en.py --num 100

echo ""; echo "[6/25] cipher_ko..."
python generation/cipher_ko.py --num 100

echo ""; echo "[7/25] cryptarithmetic_en..."
python generation/cryptarithmetic_en.py --num 300

echo ""; echo "[8/25] cryptarithmetic_ko..."
python generation/cryptarithmetic_ko.py --num 300

echo ""; echo "[9/25] inequality_en..."
python generation/inequality_en.py --num 300

echo ""; echo "[10/25] inequality_ko..."
python generation/inequality_ko.py --num 300

# kinship.py also writes a bare kinship_{diff}.jsonl; only the kinship_ko_*
# split is part of the released dataset.
echo ""; echo "[11/25] kinship_ko..."
python generation/kinship.py --num 100

echo ""; echo "[12/25] minesweeper_en..."
python generation/minesweeper_en.py --num 300

echo ""; echo "[13/25] minesweeper_ko..."
python generation/minesweeper_ko.py --num 300

echo ""; echo "[14/25] number_baseball_en..."
python generation/number_baseball_en.py --num 300

echo ""; echo "[15/25] number_baseball_ko..."
python generation/number_baseball_ko.py --num 300

echo ""; echo "[16/25] sat_puzzles_en..."
python generation/sat_puzzle_en.py --num-samples 300

echo ""; echo "[17/25] sat_puzzles_ko..."
python generation/sat_puzzle_ko.py --num-samples 300

echo ""; echo "[18/25] sudoku_en..."
python generation/sudoku_en.py --num 300

echo ""; echo "[19/25] sudoku_ko..."
python generation/sudoku_ko.py --num 300

echo ""; echo "[20/25] yacht_dice_en..."
python generation/yacht_dice_en.py --num 100

echo ""; echo "[21/25] yacht_dice_ko..."
python generation/yacht_dice_ko.py --num 100

# Saju / Four Pillars (Korean manseryeok, KO-only)
echo ""; echo "[22/25] saju_ko..."
python generation/saju_ko.py --num 300

# Jamo Composition (Korean-script structure, KO-only)
echo ""; echo "[23/25] jamo_ko..."
python generation/jamo_ko.py --num 300

# Korean date reasoning (KO-only)
echo ""; echo "[24/25] time_ko..."
python generation/time_ko.py --num 300

# Korean traditional unit conversion (KO-only)
echo ""; echo "[25/25] korean_units_ko..."
python generation/korean_units_ko.py --num 100

# bash scripts/generate_dataset.sh

# NOLLI

Logical puzzle benchmark: dataset generation code, the full dataset, and the LLM evaluation framework used in the paper.

25 tasks (10 EN/KO pairs + 5 KO-only, 3 difficulty tiers each, 100 items/tier -> 300/task, 7,500 items total).

## Puzzle Types

### Array Formula (EN / KO)
Apply row/column aggregation formulas (SUM/MEAN/MAX/MIN, ...) to a 2D array in sequence and track intermediate results to derive the final value.

### Causal DAG (EN / KO)
Infer event-propagation time through a causal graph with per-edge delays. Ground truth via shortest path (Dijkstra) on a generated DAG; unique by construction.

### Cipher (EN / KO)
Decode a stack of ciphers (Substitution, Vigenere, Reverse, Playfair, Transposition). Answers are random strings so linguistic guessing doesn't help; hint count scales with difficulty.

### Cryptarithmetic (EN / KO)
Classic letters-for-digits arithmetic (e.g. SEND+MORE=MONEY). No leading zeros, each letter a unique digit, uniqueness verified by backtracking solver.

### Inequality (EN / KO)
CSP: place numbers 1..N satisfying a set of inequality constraints. Uniqueness verified by backtracking.

### Kinship (KO)
Infer a Korean kinship term from a dialogue describing a chain of family relationships (paternal/maternal/in-law). Accepts synonym answers (e.g. 큰아버지/백부).

### Minesweeper (EN / KO)
CSP-style minesweeper with minimal hints preserving a unique solution (backtracking-verified). Difficulty via grid size (6x6/8x8/10x10). Answers are mine coordinates.

### Number Baseball (EN / KO)
Infer a hidden N-digit number (no repeated digits) from Strike/Ball hints.

### SAT Puzzle (EN / KO)
Boolean satisfiability (CNF) framed as a natural-language scenario (crime, meetings, task assignment, ...). Backward-generated so the intended answer always satisfies every clause.

### Sudoku (EN / KO)
9x9 Sudoku with guaranteed-unique solutions, HMAC-based spot-check cell selection, optional rotation/reflection symmetry, reproducible via fixed seeds.

### Yacht Dice (EN / KO)
Assign 12 dice rolls to 12 scoring categories to maximize total score; ground truth via the Hungarian algorithm over the 12! assignment space.

### Saju — Four Pillars / Manseryeok (KO-only)
Given a birth date/time, compute the four 사주(四柱) pillars (연주/월주/일주/시주). Day/hour pillars require almanac facts (입춘 boundary, 절기, 월두법/시두법) that can't be reconstructed by reasoning alone, which is what pushes the hard tier's difficulty. Ground truth: solar-longitude 절기 via `ephem`, cross-checked against `korean_lunar_calendar`.

### Jamo Composition (KO-only)
Decompose a 한글 syllable into 초성/중성/종성, shift 초성 by a fixed offset, and recompose. Difficulty comes from 받침 structure (겹받침 for hard), not text length — has no English equivalent. Ground truth is pure Unicode composition arithmetic.

### Time — Korean Calendar Reasoning (KO-only)
From a Korean holiday anchor (새해 첫날, 어린이날, ...) and a relative-day expression (금일/명일/모레, ...), compute an offset date (or its 60갑자 일진). Pure calendar computation.

### Korean Units (KO-only)
Convert mixed traditional Korean units (area/volume/length/weight — 평·마지기·되·자·냥, ...) using a conversion table given in the prompt (rates are randomized per problem, so memorized rates don't help), then compute a signed weighted sum.

## Installation

```bash
git clone https://github.com/HAE-RAE/NOLLI.git
cd NOLLI
pip install -r requirements.txt
```

To evaluate against a self-hosted open model, separately install and run [vLLM](https://github.com/vllm-project/vllm) (not a Python dependency of this repo — it serves an OpenAI-compatible HTTP endpoint that `evaluation/model/vllm.py` calls):

```bash
pip install vllm
vllm serve <model_name_or_path> --port 8000
```

## Environment Setup

```bash
cp .env.example .env
# fill in the keys for whichever providers you evaluate:
#   OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
```

## Model Calling

Two backends, selected via `--model_router`:

- **`litellm`** — cloud APIs (OpenAI, Anthropic, Google/Gemini, OpenRouter, ...) through a single interface (`evaluation/model/litellm.py`). Model name uses LiteLLM's provider-prefixed format: `openai/gpt-...`, `anthropic/claude-...`, `gemini/gemini-...`, `openrouter/<provider>/<model>`.
- **`vllm`** — a self-hosted vLLM server (or anything exposing an OpenAI-compatible `/v1/chat/completions`), called directly over HTTP/SSE (`evaluation/model/vllm.py`), no LiteLLM involved.

```bash
# LiteLLM (cloud API)
python evaluation/run.py \
    --model anthropic/claude-opus-4-8 \
    --model_router litellm \
    --gen-kwargs "max_tokens=32768,reasoning_effort=medium" \
    --tasks sudoku_en_easy --async

# vLLM (self-hosted)
python evaluation/run.py \
    --model Qwen/Qwen3-8B \
    --model_router vllm \
    --vllm_url "http://localhost:8000" \
    --gen-kwargs "temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on" \
    --tasks sudoku_en_easy --async
```

Or run every task in `data/` at once:

```bash
MODEL=anthropic/claude-opus-4-8 bash scripts/eval_litellm.sh
MODEL=Qwen/Qwen3-8B VLLM_URL=http://localhost:8000 bash scripts/eval_vllm.sh
```

Results are written to `results/{model}/{task}_{difficulty}/`.

## Regenerating the Dataset

```bash
bash scripts/generate_dataset.sh
```

Each `generation/*.py` is independently runnable (`python generation/sudoku_en.py --num 300`) and writes JSONL split by difficulty — but into `data/jsonl/` (and, for a couple of tasks, `data/csv/`), **not** `data/`. `data/` is the curated, QA'd snapshot actually used for evaluation; after regenerating and spot-checking, promote the files you want to release:

```bash
cp data/jsonl/sudoku_en_*.jsonl data/
```

## Data Format

- `data/{task}_{difficulty}.jsonl` — one file per task x difficulty (`easy`/`medium`/`hard`), 100 items each.
- Common fields: `id`, `question`, `answer`, `solution` (step-by-step reasoning), `difficulty`, plus task-specific metadata.

## Project Structure

```
NOLLI/
├── data/                    # 25 tasks x 3 difficulties = 75 JSONL files (7,500 items)
├── generation/              # one generator per task/locale
├── validators/              # dataset QA (solution-uniqueness audits)
├── evaluation/
│   ├── core/                # BaseEvaluator, ResultHandler
│   ├── evaluators/           # per-task grading logic
│   ├── model/                # LiteLLMClient / VLLMClient (LiteLLM + vLLM)
│   ├── run.py                # CLI entry point
│   └── task_names.py
├── scripts/
│   ├── generate_dataset.sh
│   ├── eval_litellm.sh
│   └── eval_vllm.sh
├── requirements.txt
└── .env.example
```

## License

MIT License — see [LICENSE](LICENSE).

# NOLLI

**NOLLI** blends Korean *놀이* (play) and *논리* (logic), naming both the reasoning the suite targets and the puzzle form in which it is tested. It is a difficulty-calibrated English–Korean benchmark for measuring where models fail across languages.

It contains **15 puzzle types** (**25 tasks**; **7,500 items**). Every item is procedurally generated, seed-regenerable, and verified to have a unique solution. Difficulty is calibrated to target accuracy bands on a fixed reference model rather than by problem size alone.

## Puzzle Types

Tasks follow a three-level cross-lingual spectrum: **direct translations**, **script adaptations**, and **Korean-only** tasks.

### 1. Direct Translations (8 types; EN + KO)

Same generators and parameters in both languages, so gaps isolate presentation language.

| task | description | en (easy/med/hard) | ko (easy/med/hard) | total |
|---|---|---|---|---|
| `array_formula` | Apply row/column aggregation formulas (SUM/MEAN/MAX/...) to a 2D array in sequence | 100 / 100 / 100 | 100 / 100 / 100 | 600 |
| `causal_dag` | Infer event-propagation time through a causal graph with per-edge delays | 100 / 100 / 100 | 100 / 100 / 100 | 600 |
| `inequality` | CSP: place numbers 1..N satisfying inequality constraints (unique solution) | 100 / 100 / 100 | 100 / 100 / 100 | 600 |
| `minesweeper` | Minesweeper with minimal hints preserving a unique solution | 100 / 100 / 100 | 100 / 100 / 100 | 600 |
| `number_baseball` | Infer a hidden N-digit number from Strike/Ball hints | 100 / 100 / 100 | 100 / 100 / 100 | 600 |
| `sat_puzzles` | Boolean satisfiability (CNF) framed as a natural-language scenario | 100 / 100 / 100 | 100 / 100 / 100 | 600 |
| `sudoku` | 9x9 Sudoku with guaranteed-unique solutions | 100 / 100 / 100 | 100 / 100 / 100 | 600 |
| `yacht_dice` | Assign 12 dice rolls to 12 scoring categories to maximize total score | 100 / 100 / 100 | 100 / 100 / 100 | 600 |

### 2. Script Adaptations (2 types; EN + KO)

Not translation-equivalent: English uses Roman letters; Korean uses Hangul *jamo*. Calibrated independently per language.

| task | description | en (easy/med/hard) | ko (easy/med/hard) | total |
|---|---|---|---|---|
| `cipher` | Decode stacked ciphers (Substitution, Vigenere, Reverse, Playfair, Transposition) | 100 / 100 / 100 | 100 / 100 / 100 | 600 |
| `cryptarithmetic` | Letters-for-digits arithmetic (SEND+MORE=MONEY style); Korean maps *jamo* groups | 100 / 100 / 100 | 100 / 100 / 100 | 600 |

### 3. Korean-Only (5 types; KO)

No English counterpart — built on structures specific to the Korean language and culture
(kinship terminology, the sexagenary calendar, traditional units, Hangul orthography).

| task | description | ko (easy/med/hard) | total |
|---|---|---|---|
| `jamo` | Decompose a Hangul syllable into 초성/중성/종성, shift 초성, recompose | 100 / 100 / 100 | 300 |
| `kinship` | Infer a Korean kinship term from a chain of family relationships (26-way multiple choice) | 100 / 100 / 100 | 300 |
| `korean_units` | Convert traditional Korean units (평·마지기·되·자·냥, ...) with randomized rates | 100 / 100 / 100 | 300 |
| `saju` | Compute the four 사주(四柱) pillars from a birth date/time | 100 / 100 / 100 | 300 |
| `time` | Korean calendar reasoning: holiday anchor + relative-day expression → date / 60갑자 일진 | 100 / 100 / 100 | 300 |

## Installation

```bash
git clone https://github.com/HAE-RAE/NOLLI.git
cd NOLLI
```

### Option A: .venv

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: conda

```bash
conda create -n nolli python=3.11 -y
conda activate nolli
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

## Evaluation Data

Evaluation loads puzzles directly from the HuggingFace hub
([HAERAE-HUB/NOLLI](https://huggingface.co/datasets/HAERAE-HUB/NOLLI)) — no local data files needed.
Task names follow `{task}_{lang}[_{difficulty}]`: `sudoku_ko` runs all three tiers,
`sudoku_ko_easy` runs one. To use locally generated jsonl files instead, pass `--data-dir data`.

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

Or run every task at once:

```bash
MODEL=anthropic/claude-opus-4-8 bash scripts/eval_litellm.sh
MODEL=Qwen/Qwen3-8B VLLM_URL=http://localhost:8000 bash scripts/eval_vllm.sh
```

Results are written to `results/{model}/{task}_{difficulty}/`.

## Regenerating the Dataset

```bash
bash scripts/generate_dataset.sh
```

Each `generation/*.py` is independently runnable (e.g. `python generation/sudoku_en.py --num 300`) and writes:

```
data/{task}_{easy|medium|hard}.jsonl
```

## Data Format

- `data/{task}_{difficulty}.jsonl` — one file per task × difficulty (`easy` / `medium` / `hard`), 100 items each.
- Common fields: `id`, `question`, `answer`, `solution` (step-by-step reasoning), `difficulty`, plus task-specific metadata.

## Project Structure

```
NOLLI/
├── data/                    # (generated locally; distributed via HAERAE-HUB/NOLLI on the HF hub)
├── generation/              # one generator per task/locale
├── validators/              # uniqueness audits
├── evaluation/
│   ├── core/                # BaseEvaluator, ResultHandler
│   ├── evaluators/           # per-task grading logic
│   ├── model/                # LiteLLMClient / VLLMClient
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

## Citation

```bibtex
@misc{choi2026nolli,
      title={NOLLI: A Difficulty-Calibrated Puzzle Benchmark for Diagnosing the English-Korean Performance Gap},
      author={Dasol Choi and Joonyong Park and Daegon Yu and Soo Yong Kim and Youngsook Song and Seunghyeok Hong},
      year={2026},
      eprint={2608.04397},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.04397},
}
```

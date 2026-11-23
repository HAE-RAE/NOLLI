#!/usr/bin/env python3
"""Build one task x tier dataset with (possibly overridden) difficulty-lever values and a chosen
seed range -- for re-calibrating the difficulty tiers against another reference model.

Only difficulty-lever *values* are changed; questions, answers and solutions come from the repo
generators:
  - cipher_en     : generation/cipher_en.py      (override keys of DIFFICULTY_CONFIG[LEVEL_x])
  - cipher_ko     : generation/cipher_ko.py      (override cipher_stack / keyword_logic / hint_count,
                    or --mix to combine several override recipes within one tier; the answer word
                    pool stays the tier's pool, built from DIFFICULTY_CONFIG when the generator is
                    created, so answer_length is not an override lever)
  - time_ko       : generation/time_ko.py        (override {"mix": {output_type: probability}})
  - causal_dag_en : generation/causal_dag_en.py  (override keys of the tier config; num_events
                    is given as an inclusive [min, max] range)

Seed of item i = seed_base + i (i from --start, default 0). With the original seed bases
(cipher_en 4000/4100/4200, cipher_ko 5000/5100/5200, time_ko 1000000/2000000/3000000,
causal_dag_en 0/1000/2000) and no override, the output equals what the generator script writes
to data/<task>_<tier>.jsonl. Generate those files first with the release sizes
(`python generation/cipher_en.py --num 100`, `cipher_ko.py --num 100` -- per tier;
`time_ko.py --num 300`, `causal_dag_en.py --num 300` -- total over the three tiers),
then run with --check-original.

Settings can be given on the command line (--n/--seed-base/--override/--mix) or read from a
recipe file (--recipe configs/reference_calibration/<model>/<task>.json), whose
tiers.<tier> entry holds n, seed_base and override or mix.

Output: <out-dir>/<task>_<tier>.jsonl         (evaluation/run.py --data-dir compatible)
        <out-dir>/<task>_<tier>.config.json   (resolved config, seeds, overlap check, generator hash)

Usage:
  python scripts/build_reference_calibration_dataset.py \\
      --recipe configs/reference_calibration/qwen3.5-27b/cipher_en.json --tier easy \\
      --out-dir data/reference_calibration/qwen3.5-27b
  python scripts/build_reference_calibration_dataset.py \\
      --task cipher_en --tier easy --n 50 --seed-base 7100000 \\
      --override '{"answer_length": [6, 10]}' --out-dir /tmp/probe
"""
import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "generation"))

TIERS = ["easy", "medium", "hard"]
ORIGINAL_SEED_BASE = {
    "cipher_en": {"easy": 4000, "medium": 4100, "hard": 4200},
    "cipher_ko": {"easy": 5000, "medium": 5100, "hard": 5200},
    "time_ko": {"easy": 1_000_000, "medium": 2_000_000, "hard": 3_000_000},
    "causal_dag_en": {"easy": 0, "medium": 1000, "hard": 2000},
}
CIPHER_LEVEL = {"easy": "LEVEL_0", "medium": "LEVEL_1", "hard": "LEVEL_2"}
CIPHER_KO_LEVERS = ("cipher_stack", "keyword_logic", "hint_count")


def _resolve_config(task, tier, override):
    if task == "cipher_en":
        import cipher_en
        cfg = copy.deepcopy(cipher_en.DIFFICULTY_CONFIG[CIPHER_LEVEL[tier]])
        for k, v in (override or {}).items():
            if k not in cfg:
                raise KeyError(f"cipher_en: unknown lever '{k}' (allowed: {sorted(cfg)})")
            cfg[k] = tuple(v) if k == "answer_length" else v
        cfg["name"] = tier
        return cfg
    if task == "cipher_ko":
        import cipher_ko
        cfg = copy.deepcopy(cipher_ko.DIFFICULTY_CONFIG[CIPHER_LEVEL[tier]])
        for k, v in (override or {}).items():
            if k not in CIPHER_KO_LEVERS:
                raise KeyError(f"cipher_ko: lever '{k}' not allowed (allowed: {list(CIPHER_KO_LEVERS)})")
            cfg[k] = v
        cfg["name"] = tier  # the answer word pool stays tied to the tier
        return cfg
    if task == "time_ko":
        import time_ko
        cfg = copy.deepcopy(time_ko.DIFFICULTY_CONFIGS[tier])
        for k, v in (override or {}).items():
            if k != "mix":
                raise KeyError("time_ko: only 'mix' can be overridden")
            if abs(sum(v.values()) - 1.0) > 1e-9:
                raise ValueError(f"time_ko: mix must sum to 1, got {sum(v.values())}")
            cfg["mix"] = v
        return cfg
    if task == "causal_dag_en":
        # the tier config lives inside generate_puzzle(); overrides are passed through as config_override
        return {"tier_config": f"causal_dag_en.CausalPuzzleGenerator.generate_puzzle['{tier}']",
                "config_override": override or None}
    raise ValueError(f"unknown task {task}")


def _resolve_mix(task, tier, mix, seed_base):
    """Recipe mix within a tier: mix = {"total": N, "parts": [{"label", "count", "override"}]}.
    Exact counts per label, stratified per 50-item block; the label of item i is fixed by a
    seed_base-seeded shuffle, so a dataset built in two halves (--start 0 / --start 50) gets the
    same assignment as one built at once. Only cipher_ko supports it."""
    if task != "cipher_ko":
        raise ValueError("--mix is only implemented for cipher_ko")
    total = int(mix["total"])
    if sum(int(p["count"]) for p in mix["parts"]) != total:
        raise ValueError(f"mix counts do not sum to total {total}")
    # blocks of 50: each block gets the proportional count by largest-remainder allocation;
    # the last block takes what remains.
    block, labels = 50, []
    rng = random.Random(f"mix-{seed_base}")
    remaining = {p["label"]: int(p["count"]) for p in mix["parts"]}
    order = [p["label"] for p in mix["parts"]]
    n_blocks = -(-total // block)
    for b in range(n_blocks):
        size = min(block, total - b * block)
        left = sum(remaining.values())
        if b == n_blocks - 1:
            share = dict(remaining)
        else:
            quota = {k: remaining[k] * size / left for k in order}
            share = {k: int(quota[k]) for k in order}
            for k in sorted(order, key=lambda k: (-(quota[k] - share[k]), order.index(k)))[:size - sum(share.values())]:
                share[k] += 1
        blk = [k for k in order for _ in range(share[k])]
        if len(blk) != size:
            raise ValueError(f"mix block {b}: {len(blk)} labels for {size} items")
        for k in order:
            remaining[k] -= share[k]
        rng.shuffle(blk)
        labels += blk
    cfgs = {part["label"]: _resolve_config(task, tier, part["override"] or None) for part in mix["parts"]}
    return {"__mix__": {"labels": labels, "cfgs": cfgs}}


def _generate(task, tier, cfg, n, seed_base, start=0):
    items = []
    if task == "cipher_en":
        import cipher_en
        gen = cipher_en.SelfContainedCipherGenerator()
        for i in range(start, start + n):
            p = gen.generate_problem(cfg, seed_base + i)
            items.append({
                "id": f"cipher_en_{tier}_{i:04d}",
                "question": p["problem"],
                "answer": p["answer"],
                "solution": p["solution"],
                "difficulty": tier,
            })
    elif task == "cipher_ko":
        # same as generation/cipher_ko.py create_hangul_dataset: per-tier answer de-dup with
        # seed = seed_base + i + attempt*10000. Items 0..start-1 are regenerated (not written) so
        # that the de-dup state is identical when extending a dataset.
        import cipher_ko
        gen = cipher_ko.HangulCipherGenerator()
        used = set()
        mix = cfg.get("__mix__")
        if mix is not None and len(mix["labels"]) < start + n:
            raise ValueError("mix total is smaller than start + n")
        for i in range(start + n):
            cfg_i = mix["cfgs"][mix["labels"][i]] if mix is not None else cfg
            prob = None
            for attempt in range(1000):
                cand = gen.generate_problem(cfg_i, seed_base + i + attempt * 10000)
                if cand["answer"] not in used or len(used) >= len(gen.answer_pools[tier]):
                    prob = cand
                    break
            if prob is None:
                raise RuntimeError("cipher_ko: could not generate a de-duplicated problem")
            used.add(prob["answer"])
            if i >= start:
                items.append({
                    "id": f"cipher_ko_{tier}_{i:04d}",
                    "question": prob["problem"],
                    "answer": prob["answer"],
                    "solution": prob["solution"],
                    "difficulty": tier,
                })
    elif task == "time_ko":
        import time_ko
        for i in range(start, start + n):
            items.append(time_ko.generate_problem(tier, i, seed=seed_base + i, mix=cfg["mix"]))
    elif task == "causal_dag_en":
        import causal_dag_en
        gen = causal_dag_en.CausalPuzzleGenerator()
        for i in range(start, start + n):
            p = gen.generate_puzzle(tier, seed=seed_base + i, config_override=cfg["config_override"])
            items.append({
                "id": f"causal_dag_en_{tier}_{i:04d}",
                "question": causal_dag_en.create_question(p),
                "answer": str(p.answer),
                "solution": causal_dag_en._build_causal_dag_solution_en(p),
                "difficulty": tier,
            })
    return items


def _overlap_with_original(task, items, original_dir):
    """Count generated questions that already appear in <original_dir>/<task>_<tier>.jsonl (any tier).
    Returns None when none of the original files exist."""
    seen, found = set(), 0
    for t in TIERS:
        path = original_dir / f"{task}_{t}.jsonl"
        if path.exists():
            found += 1
            with open(path, encoding="utf-8") as f:
                seen.update(json.loads(line)["question"] for line in f)
    if found == 0:
        return None
    return sum(it["question"] in seen for it in items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default=None,
                    help="JSON recipe file; tiers.<tier> supplies n, seed_base and override or mix")
    ap.add_argument("--task", default=None, choices=sorted(ORIGINAL_SEED_BASE))
    ap.add_argument("--tier", required=True, choices=TIERS)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--start", type=int, default=0,
                    help="first item index (item i uses seed seed_base+i and id suffix i); to extend a dataset")
    ap.add_argument("--override", default=None, help="JSON dict of difficulty-lever overrides")
    ap.add_argument("--mix", default=None,
                    help='cipher_ko only: JSON {"total":100,"parts":[{"label":"rev","count":80,"override":{...}},...]}')
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--original-dir", default=str(PROJECT_ROOT / "data"),
                    help="directory with the original <task>_<tier>.jsonl (for --check-original and the overlap count)")
    ap.add_argument("--check-original", action="store_true",
                    help="regression: regenerate with the original seeds and no override, compare to --original-dir")
    args = ap.parse_args()

    override = json.loads(args.override) if args.override else None
    mix = json.loads(args.mix) if args.mix else None
    if args.recipe is not None:
        if args.check_original or override is not None or mix is not None or args.n is not None \
                or args.seed_base is not None:
            ap.error("--recipe cannot be combined with --check-original/--override/--mix/--n/--seed-base")
        recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
        if args.task is not None and args.task != recipe["task"]:
            ap.error(f"--task {args.task} does not match recipe task {recipe['task']}")
        args.task = recipe["task"]
        entry = recipe["tiers"][args.tier]
        args.n, args.seed_base = int(entry["n"]), int(entry["seed_base"])
        override, mix = entry.get("override"), entry.get("mix")
    if args.task is None:
        ap.error("--task (or --recipe) is required")
    if args.task not in ORIGINAL_SEED_BASE:
        ap.error(f"unsupported task {args.task}")
    if mix is not None and override:
        ap.error("use either override or mix")
    original_dir = Path(args.original_dir)

    if args.check_original:
        if override or mix is not None:
            ap.error("--check-original uses the original config; drop --override/--mix")
        cfg = _resolve_config(args.task, args.tier, None)
        orig_path = original_dir / f"{args.task}_{args.tier}.jsonl"
        orig = [json.loads(line) for line in open(orig_path, encoding="utf-8")]
        new = _generate(args.task, args.tier, cfg, len(orig), ORIGINAL_SEED_BASE[args.task][args.tier])
        same = sum(a == b for a, b in zip(orig, new))
        print(f"[check-original] {args.task}_{args.tier}: {same}/{len(orig)} identical")
        sys.exit(0 if same == len(orig) else 1)

    if args.out_dir is None:
        ap.error("--out-dir is required unless --check-original")
    if args.n is None or args.seed_base is None:
        ap.error("--n and --seed-base are required without --recipe")
    cfg = (_resolve_mix(args.task, args.tier, mix, args.seed_base) if mix is not None
           else _resolve_config(args.task, args.tier, override))
    items = _generate(args.task, args.tier, cfg, args.n, args.seed_base, args.start)
    n_unique = len({it["question"] for it in items})
    overlap = _overlap_with_original(args.task, items, original_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_{args.tier}"
    with open(out_dir / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "generation/"], cwd=PROJECT_ROOT,
                                        text=True).strip()
    except Exception:
        commit, dirty = None, None
    gen_file = PROJECT_ROOT / "generation" / f"{args.task}.py"
    meta = {
        "task": args.task, "tier": args.tier, "n": args.n,
        "seed_base": args.seed_base, "start": args.start,
        "seeds": [args.seed_base + args.start, args.seed_base + args.start + args.n - 1],
        "recipe": args.recipe, "override": override, "mix": mix,
        "item_labels": cfg["__mix__"]["labels"][args.start:args.start + args.n] if mix is not None else None,
        "resolved_config": cfg,
        "unique_questions": n_unique,
        "overlap_with_original_data": overlap,
        "original_dir": str(original_dir),
        "git_commit": commit, "generation_dir_uncommitted_changes": dirty,
        "generator_file": str(gen_file.relative_to(PROJECT_ROOT)),
        "generator_sha256": hashlib.sha256(gen_file.read_bytes()).hexdigest(),
    }
    with open(out_dir / f"{stem}.config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=list)
    print(f"wrote {out_dir / (stem + '.jsonl')} n={args.n} unique={n_unique} "
          f"overlap_with_original={overlap if overlap is not None else 'n/a (no original files)'}")


if __name__ == "__main__":
    main()

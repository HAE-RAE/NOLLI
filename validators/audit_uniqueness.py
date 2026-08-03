#!/usr/bin/env python3
"""
A3: Verify that puzzles claiming a UNIQUE solution actually have one, by running
each task's real solver on the PUBLISHED dataset (data/).

Motivation: the CSP tasks call real solvers in generation, but the published
dataset can be stale; so verify directly against the shipped JSONL.

Covers (parseable + reusable solver): sat, cryptarithmetic, number_baseball, sudoku.
For each: parse the puzzle from the dataset, count solutions (cap 2), report %unique.
Also records solver work (intrinsic difficulty) where available.
"""
import json, re, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TIERS = ["easy", "medium", "hard"]


def load_mod(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jl(task, tier):
    p = ROOT / f"data/{task}_{tier}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def report(task, per_tier):
    print(f"\n### {task}")
    print(f"{'tier':8}{'N':>5}{'unique%':>9}{'multi%':>8}{'0해%':>7}{'median solver work':>20}")
    for t in TIERS:
        d = per_tier.get(t)
        if not d:
            print(f"{t:8}{'-':>5}")
            continue
        n, u, m, z, work = d
        wk = f"{int(np.median(work))}" if work else "-"
        print(f"{t:8}{n:>5}{u/n*100:>8.1f}%{m/n*100:>7.1f}%{z/n*100:>6.1f}%{wk:>20}")


# --------------------------------------------------------------------------- #
# SAT: count satisfying assignments (published JSONL has no variables/clauses;
# parse Variables from question + CNF from solution [SEG] lines)
# --------------------------------------------------------------------------- #
def _parse_sat(r):
    """Return (variables, clauses) where clauses are [[(var, polarity), ...], ...]."""
    if r.get("variables") and r.get("clauses"):
        vars = list(r["variables"])
        clauses = [[(var, pol) for var, pol in cl] for cl in r["clauses"]]
        return vars, clauses

    vm = re.search(r"\*\*Variables:\*\*\s*(.+)", r["question"])
    if not vm:
        return None, None
    vars = [x.strip() for x in vm.group(1).split(",") if x.strip()]

    clauses = []
    for body in re.findall(r"\[SEG \d+\] clause \((.+?)\) ->", r.get("solution", "")):
        cl = []
        for tok in body.split(" OR "):
            tok = tok.strip()
            if tok.startswith("NOT "):
                cl.append((tok[4:].strip(), False))
            else:
                cl.append((tok, True))
        if cl:
            clauses.append(cl)
    if not clauses:
        return None, None
    return vars, clauses


def audit_sat():
    out = {}
    for t in TIERS:
        rows = jl("sat_puzzles_en", t)
        if not rows:
            continue
        u = m = z = 0
        work = []
        for r in rows:
            vars, clauses = _parse_sat(r)
            if not vars or not clauses:
                continue
            cnt = 0
            for i in range(2 ** len(vars)):
                a = {v: bool((i >> j) & 1) for j, v in enumerate(vars)}
                if all(any(a[var] == pol for var, pol in cl) for cl in clauses):
                    cnt += 1
                    if cnt >= 2:
                        break
            work.append(2 ** len(vars))
            if cnt == 0:
                z += 1
            elif cnt == 1:
                u += 1
            else:
                m += 1
        out[t] = (len(work), u, m, z, work)
    return out


# --------------------------------------------------------------------------- #
# Cryptarithmetic: parse equation -> find_solutions (reuse generator solver)
# --------------------------------------------------------------------------- #
def _parse_crypt_en(r, cr):
    """(단어 리스트, 공개키 token->digit) 반환. 2026-07 재작성판: 유일성은
    '공개된 부분키' 조건부라 반드시 revealed 를 fixed= 로 넘겨야 한다."""
    q = r["question"]
    words = [mm.group(1) for line in q.splitlines()
             if (mm := re.match(r"^[+=]?\s*([A-Z]{2,})\s*$", line.strip()))]
    revealed = {}
    for m in re.finditer(r"\[group [A-Z]-[A-Z]\]\s*(.+)", q):
        for pair in m.group(1).split(","):
            mm = re.match(r"([A-Z])=(\d)", pair.strip())
            if mm:
                ch, d = mm.group(1), int(mm.group(2))
                revealed[f"{cr._group_of(ch)}{ch}"] = d
    return words, revealed


def audit_crypt():
    # 2026-07 재작성(3독립표): 공개키 없이 find_solutions 를 부르면 비유일로
    # 잘못 보고된다 → 반드시 revealed 를 fixed= 로 넘겨 조건부 유일성을 잰다.
    cr = load_mod("cr", "generation/cryptarithmetic_en.py")
    out = {}
    for t in TIERS:
        rows = jl("cryptarithmetic_en", t)
        if not rows:
            continue
        u = m = z = 0
        work = []
        for r in rows:
            words, revealed = _parse_crypt_en(r, cr)
            if len(words) < 3:
                continue
            stt = {}
            sols = cr.find_solutions(tuple(words), max_count=2,
                                     fixed=dict(revealed), _stats=stt)
            work.append(stt.get("nodes", 0))
            ans = str(r.get("answer", "")).strip()
            n_uni = len(sols)
            if n_uni == 0:
                z += 1
            elif n_uni == 1 and sols[0][0] == ans:
                u += 1
            else:
                m += 1
        out[t] = (len(work), u, m, z, work)
    return out


# --------------------------------------------------------------------------- #
# Number baseball: parse hints -> find_all_solutions (reuse)
# --------------------------------------------------------------------------- #
def audit_nb():
    nb = load_mod("nb", "generation/number_baseball_en.py")
    out = {}
    for t in TIERS:
        rows = jl("number_baseball_en", t)
        if not rows:
            continue
        u = m = z = 0
        for r in rows:
            n = int(re.search(r"has (\d+) digits", r["question"]).group(1))
            hints = [nb.Hint(g, int(s), int(b)) for g, s, b in
                     re.findall(r"Guess:\s*(\d+)\s*->\s*(\d+) Strike\(s\),\s*(\d+) Ball", r["question"])]
            sols = nb.BullsAndCows(n).find_all_solutions(hints, max_count=2)
            if len(sols) == 0:
                z += 1
            elif len(sols) == 1:
                u += 1
            else:
                m += 1
        out[t] = (len(rows), u, m, z, [])
    return out


# --------------------------------------------------------------------------- #
# Sudoku: parse 9x9 grid -> count_solutions(limit=2) (reuse)
# --------------------------------------------------------------------------- #
def audit_sudoku():
    sd = load_mod("sd", "generation/sudoku_en.py")
    out = {}
    for t in TIERS:
        rows = jl("sudoku_en", t)
        if not rows:
            continue
        u = m = z = 0
        for r in rows:
            q = r["question"]
            seg = q.split("Puzzle grid:")[-1]
            grid = []
            for line in seg.splitlines():
                toks = line.strip().split()
                if len(toks) == 9 and all(tk == "." or tk.isdigit() for tk in toks):
                    grid.append([0 if tk == "." else int(tk) for tk in toks])
                if len(grid) == 9:
                    break
            if len(grid) != 9:
                continue
            cnt = sd.count_solutions(grid, limit=2)
            if cnt == 0:
                z += 1
            elif cnt == 1:
                u += 1
            else:
                m += 1
        out[t] = (u + m + z, u, m, z, [])
    return out


# --------------------------------------------------------------------------- #
# Minesweeper: parse 'rN c c c ...' grid (# = hidden) -> solve_puzzle(max=2) (reuse)
# --------------------------------------------------------------------------- #
def audit_minesweeper():
    ms = load_mod("ms", "generation/minesweeper_en.py")
    out = {}
    for t in TIERS:
        rows = jl("minesweeper_en", t)
        if not rows:
            continue
        u = m = z = 0
        work = []
        for r in rows:
            q = r["question"]
            mines = int(re.search(r"Total mines:\s*(\d+)", q).group(1))
            grid = []
            for line in q.splitlines():
                mm = re.match(r"^r\d+\s+(.+)$", line.strip())
                if not mm:
                    continue
                toks = mm.group(1).split()
                if all(tk == "#" or tk.isdigit() for tk in toks) and len(toks) >= 5:
                    grid.append([None if tk == "#" else int(tk) for tk in toks])
            if not grid:
                continue
            R, C = len(grid), len(grid[0])
            stt = {}
            sols = ms.solve_puzzle(grid, R, C, max_solutions=2, total_mines=mines, _stats=stt)
            work.append(stt.get("nodes", 0))
            if len(sols) == 0:
                z += 1
            elif len(sols) == 1:
                u += 1
            else:
                m += 1
        out[t] = (u + m + z, u, m, z, work)
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("A3 — 공개 데이터셋 유일해 전수 검증 (task별 실제 솔버)")
    print("=" * 70)
    report("sat_puzzles_en", audit_sat())
    report("cryptarithmetic_en", audit_crypt())
    report("number_baseball_en", audit_nb())
    report("sudoku_en", audit_sudoku())
    report("minesweeper_en", audit_minesweeper())

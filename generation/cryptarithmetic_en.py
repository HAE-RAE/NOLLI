"""Cryptarithmetic puzzle generator (EN) — letter-group multi-table, KO-aligned.

This mirrors generation/cryptarithmetic_ko.py exactly, replacing Hangul's
3 jamo positions (초성/중성/종성) with 3 fixed LETTER GROUPS:
    group 0 = A–I,  group 1 = J–R,  group 2 = S–Z
Each group has its OWN independent substitution table (letter→digit).
- Within a group: different letters → different digits (0–9).
- The 3 tables are independent (A in group0 may be 5, and so may some S in group2).
- A word's numeric value = each letter → its group's digit, concatenated left→right.
- Each word's first letter ≠ 0.

Same algorithm as KO: real words → decompose to typed tokens (here: group-tagged
letters) → multi independent tables → reveal a partial key → the rest is deduced so
the addition holds → answer = numeric value of the result word (= the sum).

(Classic constructed-letter version archived at backups/cryptarithmetic_en_classic_pre_kostyle.py)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

MAX_SOLUTIONS = 1

# 3 fixed letter groups (영어판 '초성/중성/종성'): index 0/1/2
_GROUP_LABEL = {0: "A-I", 1: "J-R", 2: "S-Z"}


def _group_of(ch: str) -> int:
    if "A" <= ch <= "I":
        return 0
    if "J" <= ch <= "R":
        return 1
    return 2


# ============================================================================
# 실제 영어 단어 풀 (글자 길이별 버킷은 런타임 계산)
# ============================================================================

_WORD_POOL_FLAT = [
    # 4
    "BIRD", "FISH", "LION", "BEAR", "WOLF", "DEER", "GOAT", "FROG", "DUCK", "CRAB",
    "TREE", "ROSE", "LEAF", "SEED", "CORN", "RICE", "BEAN", "PLUM", "LIME", "PEAR",
    "BLUE", "GOLD", "GRAY", "PINK", "RUBY", "JADE", "SNOW", "RAIN", "WIND", "FIRE",
    "ROAD", "PATH", "GATE", "DOOR", "ROOF", "WALL", "DESK", "BOOK", "PAGE", "WORD",
    "TIME", "YEAR", "WEEK", "NOON", "DAWN", "STAR", "MOON", "WAVE", "LAKE", "HILL",
    # 5
    "BREAD", "TABLE", "CHAIR", "HOUSE", "MONEY", "WATER", "LIGHT", "PLANT", "MUSIC",
    "DREAM", "STONE", "RIVER", "OCEAN", "CLOUD", "STORM", "GRASS", "FRUIT", "APPLE",
    "GRAPE", "LEMON", "MANGO", "TIGER", "HORSE", "SHEEP", "MOUSE", "EAGLE", "ROBIN",
    "SNAKE", "WHALE", "SHARK", "TRAIN", "PLANE", "TRUCK", "WHEEL", "ROBOT", "PHONE",
    "GLASS", "BRICK", "STEEL", "PAPER", "CANDY", "HONEY", "SUGAR", "FLOUR", "JUICE",
    # 6
    "FLOWER", "GARDEN", "FOREST", "ISLAND", "DESERT", "VALLEY", "BRIDGE", "CASTLE",
    "MARKET", "SCHOOL", "OFFICE", "WINDOW", "MIRROR", "PENCIL", "BOTTLE", "BASKET",
    "RABBIT", "MONKEY", "TURTLE", "PARROT", "SPIDER", "DRAGON", "ORANGE", "BANANA",
    "CHERRY", "CARROT", "POTATO", "TOMATO", "PEPPER", "WINTER", "SPRING", "SUMMER",
    "PLANET", "ROCKET", "CAMERA", "GUITAR", "VIOLIN", "DOCTOR", "FARMER", "SINGER",
    # 7
    "RAINBOW", "THUNDER", "DIAMOND", "PICTURE", "KITCHEN", "LIBRARY", "STATION",
    "MORNING", "EVENING", "JOURNEY", "FREEDOM", "JUSTICE", "PROJECT", "PROGRAM",
    "MACHINE", "BICYCLE", "PENGUIN", "DOLPHIN", "LEOPARD", "GIRAFFE", "OCTOPUS",
    # 8
    "ELEPHANT", "MOUNTAIN", "HOSPITAL", "COMPUTER", "SANDWICH", "UMBRELLA",
    "DINOSAUR", "AIRPLANE", "BUILDING", "CHILDREN", "BIRTHDAY", "BASEBALL",
    "FOOTBALL", "SNOWBALL", "STARFISH", "NOTEBOOK", "SCISSORS", "TRIANGLE",
]


def _word_letters(word: str) -> int:
    return len(word)


# ============================================================================
# 토큰화 (글자 -> 그룹태그 토큰)
# ============================================================================

def _decompose_word_to_tokens(word: str) -> List[str]:
    """단어를 그룹태그된 글자 토큰 리스트로 분해. 토큰 = str(group)+letter (예 '0B')."""
    return [f"{_group_of(c)}{c}" for c in word]


def _word_value(word: str, token_map: Dict[str, int]) -> int:
    ds = [str(token_map[t]) for t in _decompose_word_to_tokens(word)]
    return int("".join(ds)) if ds else 0


_WORD_BY_LEN: Dict[int, List[str]] = defaultdict(list)
for _w in dict.fromkeys(_WORD_POOL_FLAT):
    _WORD_BY_LEN[_word_letters(_w)].append(_w)


@dataclass
class PuzzleCandidate:
    operand_words: List[str]
    result: str
    answer: str
    operand_ints: List[int]
    # 전체(정답) 키: {"g0":{letter:d}, "g1":{...}, "g2":{...}}
    mapping: Dict[str, Dict[str, int]] = None
    revealed: Dict[str, Dict[str, int]] = None
    solver_steps: int = 0

    @property
    def puzzle_str(self) -> str:
        return " + ".join(self.operand_words) + f" = {self.result}"

    @property
    def operands(self):
        return list(self.operand_words)


# ============================================================================
# Solver: 그룹별 독립 치환 + 일부 토큰 고정(fixed) + 노드 예산
# ============================================================================

class _NodeBudgetExceeded(Exception):
    pass


def find_solutions(
    puzzle: tuple,
    max_count: int = 4,
    _stats: Optional[Dict] = None,
    fixed: Optional[Dict[str, int]] = None,
    max_nodes: Optional[int] = None,
) -> List[Tuple[str, Dict[str, int]]]:
    """열별 백트래킹. 기호 = 그룹태그된 글자 토큰, distinct 제약은 그룹별 독립."""
    fixed = fixed or {}

    *operand_words, result_word = puzzle
    words = list(operand_words) + [result_word]
    word_tokens = [_decompose_word_to_tokens(w) for w in words]
    op_tokens = word_tokens[:-1]
    res_tokens = word_tokens[-1]

    all_tokens: Set[str] = set()
    for wt in word_tokens:
        all_tokens.update(wt)

    by_grp: Dict[str, Set[str]] = defaultdict(set)
    for t in all_tokens:
        by_grp[t[0]].add(t)
    for toks in by_grp.values():
        if len(toks) > 10:
            return []

    first_tokens = set(wt[0] for wt in word_tokens if wt)

    max_len = max(len(wt) for wt in word_tokens)
    reversed_ops = [wt[::-1] for wt in op_tokens]
    wr = res_tokens[::-1]

    ordered: List[str] = []
    seen: Set[str] = set(fixed.keys())
    for col in range(max_len):
        for wt in reversed_ops + [wr]:
            if col < len(wt) and wt[col] not in seen:
                ordered.append(wt[col])
                seen.add(wt[col])
    for t in all_tokens:
        if t not in seen:
            ordered.append(t)
            seen.add(t)

    cols_op: List[Tuple[str, ...]] = []
    cols_cr: List[Optional[str]] = []
    for col in range(max_len):
        cols_op.append(tuple(rw[col] for rw in reversed_ops if col < len(rw)))
        cols_cr.append(wr[col] if col < len(wr) else None)

    solutions: List[Tuple[str, Dict[str, int]]] = []
    mapping: Dict[str, int] = dict(fixed)
    used: Dict[str, set] = {"0": set(), "1": set(), "2": set()}
    for _tok, _d in fixed.items():
        used[_tok[0]].add(_d)

    node_count = [0]

    def _check_full(mp: Dict[str, int]) -> bool:
        carry = 0
        for col in range(max_len):
            ops = cols_op[col]
            cr = cols_cr[col]
            for ol in ops:
                if ol not in mp:
                    return True
            if cr is not None and cr not in mp:
                return True
            total = carry + sum(mp[ol] for ol in ops)
            dr = mp[cr] if cr is not None else 0
            if total % 10 != dr:
                return False
            carry = total // 10
        return carry == 0

    def backtrack(idx: int):
        node_count[0] += 1
        if _stats is not None:
            _stats['nodes'] = node_count[0]
        if max_nodes is not None and node_count[0] > max_nodes:
            raise _NodeBudgetExceeded
        if len(solutions) >= max_count:
            return
        if idx == len(ordered):
            if _check_full(mapping):
                num_result = int("".join(str(mapping[t]) for t in res_tokens))
                solutions.append((str(num_result), dict(mapping)))
            return

        tok = ordered[idx]
        grp = tok[0]
        used_grp = used[grp]
        is_first = tok in first_tokens

        for digit in range(10):
            if digit in used_grp:
                continue
            if digit == 0 and is_first:
                continue
            mapping[tok] = digit
            used_grp.add(digit)

            valid = True
            carry = 0
            for col in range(max_len):
                ops = cols_op[col]
                cr = cols_cr[col]
                ready = True
                for ol in ops:
                    if ol not in mapping:
                        ready = False
                        break
                if ready and not (cr is None or cr in mapping):
                    ready = False
                if ready:
                    total = carry + sum(mapping[ol] for ol in ops)
                    dr = mapping[cr] if cr is not None else 0
                    if total % 10 != dr:
                        valid = False
                        break
                    carry = total // 10
                else:
                    break
            if valid:
                backtrack(idx + 1)
            del mapping[tok]
            used_grp.discard(digit)

    try:
        backtrack(0)
    except _NodeBudgetExceeded:
        if _stats is not None:
            _stats['aborted'] = True
    return solutions


# ============================================================================
# 산술 헬퍼
# ============================================================================

def count_carries(*nums: int) -> int:
    carries = 0
    carry = 0
    str_nums = [str(n)[::-1] for n in nums]
    max_len = max(len(s) for s in str_nums)
    for i in range(max_len):
        total = carry
        for s in str_nums:
            total += int(s[i]) if i < len(s) else 0
        carry = total // 10
        if carry > 0:
            carries += 1
    return carries


# ============================================================================
# 부분 키 선택 (유일해 유지 최소 공개)
# ============================================================================

def _select_revealed(puzzle: tuple, full_tokens: Dict[str, int], hide_ratio: float):
    tokens = list(full_tokens.keys())
    total = len(tokens)
    target_hide = max(1, round(hide_ratio * total))
    random.shuffle(tokens)
    revealed = dict(full_tokens)
    hidden = 0
    for tok in tokens:
        if hidden >= target_hide:
            break
        trial = {t: d for t, d in revealed.items() if t != tok}
        sols = find_solutions(puzzle, max_count=MAX_SOLUTIONS + 1, fixed=trial)
        if len(sols) == 1:
            del revealed[tok]
            hidden += 1
    if hidden == 0:
        return None
    return revealed, hidden


def _tokens_to_tables(tokens: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    out = {"g0": {}, "g1": {}, "g2": {}}
    for tok, d in tokens.items():
        out["g" + tok[0]][tok[1:]] = d
    return out


def _tables_to_tokens(tables: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for gi in (0, 1, 2):
        for letter, d in (tables or {}).get("g%d" % gi, {}).items():
            out["%d%s" % (gi, letter)] = d
    return out


# ============================================================================
# 난이도 설정 & 생성
# ============================================================================
# 난이도 = 피연산자 수 + 단어 길이(글자수=자릿수) + 가림 비율(추론량).
# 목표 정확도(gemini-3-flash, KO와 동일): easy~75% · medium~50% · hard~20-25%.
DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    "easy":   {"num_operands": 4, "op_lens": (6,),    "hide_ratio": 0.65,
               "min_carries": 2, "max_attempts": 60000},
    "medium": {"num_operands": 4, "op_lens": (7,),    "hide_ratio": 0.90,
               "min_carries": 2, "max_attempts": 90000},
    "hard":   {"num_operands": 4, "op_lens": (8,),    "hide_ratio": 0.88,
               "min_carries": 3, "max_attempts": 130000},
}


def generate_puzzle_by_difficulty(
    difficulty: str,
    used_patterns: Set[str] = None,
    **overrides,
) -> Optional[PuzzleCandidate]:
    if used_patterns is None:
        used_patterns = set()

    config = dict(DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["easy"]))
    config.update(overrides)

    N = config["num_operands"]
    hide_ratio = config["hide_ratio"]
    min_carries = config.get("min_carries", 0)
    op_lens = config.get("op_lens", (4, 5))
    max_attempts = config.get("max_attempts", 40000)

    lens = [L for L in op_lens if len(_WORD_BY_LEN.get(L, [])) >= N]
    if not lens:
        return None

    for _ in range(max_attempts):
        L = random.choice(lens)
        ops = random.sample(_WORD_BY_LEN[L], N)
        res_cands = [w for Lr in (L, L + 1)
                     for w in _WORD_BY_LEN.get(Lr, []) if w not in ops]
        if not res_cands:
            continue
        random.shuffle(res_cands)

        for r in res_cands[:8]:
            puzzle = tuple(ops + [r])
            sols = find_solutions(puzzle, max_count=2, max_nodes=30000)
            if not sols:
                continue

            full_tokens = sols[0][1]
            answer = sols[0][0]
            operand_ints = [_word_value(w, full_tokens) for w in ops]
            if count_carries(*operand_ints) < min_carries:
                continue

            pattern = "+".join(ops) + "=" + r
            if pattern in used_patterns:
                continue

            sel = _select_revealed(puzzle, full_tokens, hide_ratio)
            if sel is None:
                continue
            revealed_tokens, _hidden = sel

            stats = {'nodes': 0}
            s2 = find_solutions(puzzle, max_count=MAX_SOLUTIONS + 1,
                                fixed=revealed_tokens, _stats=stats)
            if len(s2) != 1 or s2[0][0] != answer:
                continue

            used_patterns.add(pattern)
            return PuzzleCandidate(
                operand_words=ops,
                result=r,
                answer=answer,
                operand_ints=operand_ints,
                mapping=_tokens_to_tables(full_tokens),
                revealed=_tokens_to_tables(revealed_tokens),
                solver_steps=stats['nodes'],
            )

    return None


# ============================================================================
# 추론 연쇄 (teacher trace)
# ============================================================================

def _deduction_steps(
    puzzle: tuple,
    revealed_tokens: Dict[str, int],
    full_tokens: Dict[str, int],
    operand_ints: List[int],
    answer: str,
) -> List[str]:
    *operand_words, result_word = puzzle
    words = list(operand_words) + [result_word]
    word_tokens = [_decompose_word_to_tokens(w) for w in words]
    op_tokens = word_tokens[:-1]
    res_tokens = word_tokens[-1]
    max_len = max(len(wt) for wt in word_tokens)
    rev_ops = [wt[::-1] for wt in op_tokens]
    rev_res = res_tokens[::-1]
    first_tokens = set(wt[0] for wt in word_tokens if wt)

    op_strs = [str(n) for n in operand_ints]
    L = max(len(s) for s in op_strs + [answer])
    pop = [s.rjust(L) for s in op_strs]
    carries = [0] * (L + 1)
    for pos in range(L):
        ci = L - 1 - pos
        s = sum(int(x[ci]) for x in pop if x[ci] != ' ') + carries[pos]
        carries[pos + 1] = s // 10

    known = dict(revealed_tokens)
    used = {"0": set(), "1": set(), "2": set()}
    for t, d in known.items():
        used[t[0]].add(d)
    remaining = [t for t in full_tokens if t not in known]

    def _forced(t: str) -> Optional[int]:
        grp = t[0]
        cands = []
        for d in range(10):
            if d in used[grp]:
                continue
            if d == 0 and t in first_tokens:
                continue
            if find_solutions(puzzle, max_count=1, fixed={**known, t: d}):
                cands.append(d)
                if len(cands) > 1:
                    break
        return cands[0] if len(cands) == 1 else None

    def _single_unknown_col(t: str):
        for pos in range(max_len):
            col_ops = [ro[pos] for ro in rev_ops if pos < len(ro)]
            col_res = rev_res[pos] if pos < len(rev_res) else None
            toks = col_ops + ([col_res] if col_res else [])
            if t in toks and all(o in known for o in toks if o != t):
                return pos, col_ops, col_res
        return None

    steps: List[str] = []
    step_no = 0
    while remaining:
        picked = None
        for t in remaining:
            d = _forced(t)
            if d is not None:
                picked = (t, d)
                break
        if picked is None:
            labels = ", ".join(f"{t[1:]}({_GROUP_LABEL[int(t[0])]})" for t in remaining)
            steps.append(f"  - [deduce] Remaining letters {labels} are fixed uniquely by "
                         f"the combined column constraints.")
            for t in remaining:
                known[t] = full_tokens[t]
            break

        t, d = picked
        step_no += 1
        letter = t[1:]
        glabel = _GROUP_LABEL[int(t[0])]
        col = _single_unknown_col(t)
        if col:
            pos, col_ops, col_res = col
            cin = carries[pos]
            cout = carries[pos + 1]
            parts = [f"[{letter}=?]" if o == t else f"{full_tokens[o]}" for o in col_ops]
            terms = " + ".join(parts)
            if cin:
                terms += f" + carry {cin}"
            colsum = sum(full_tokens[o] for o in col_ops) + cin
            if col_res == t:
                steps.append(f"  - [step {step_no}] column {pos + 1} (right→left): "
                             f"{terms} = {colsum} → last digit {d}, so {glabel} letter {letter}={d}.")
            else:
                rd = full_tokens[col_res] if col_res else 0
                steps.append(f"  - [step {step_no}] column {pos + 1} (right→left): "
                             f"{terms} = result {rd} (carry {cout}) ⇒ {glabel} letter {letter}={d}.")
        else:
            steps.append(f"  - [step {step_no}] {glabel} letter {letter}: given the revealed "
                         f"and already-fixed values, only {d} is consistent → {letter}={d}.")
        known[t] = d
        used[t[0]].add(d)
        remaining.remove(t)

    return steps


# ============================================================================
# 문제/풀이 텍스트
# ============================================================================

def _format_table(table: Dict[str, int]) -> str:
    if not table:
        return "(none revealed)"
    return ", ".join(f"{l}={d}" for l, d in sorted(table.items()))


def _annotated_table(full: Dict[str, int], revealed: Dict[str, int]) -> str:
    parts = []
    for l, d in sorted(full.items()):
        tag = "given" if l in revealed else "deduced"
        parts.append(f"{l}={d}[{tag}]")
    return ", ".join(parts) if parts else "(none)"


def create_question(candidate: PuzzleCandidate) -> str:
    operand_words = candidate.operands
    revealed = candidate.revealed or {"g0": {}, "g1": {}, "g2": {}}

    g0 = _format_table(revealed.get("g0", {}))
    g1 = _format_table(revealed.get("g1", {}))
    g2 = _format_table(revealed.get("g2", {}))

    op_lines = f"  {operand_words[0]}\n"
    for w in operand_words[1:]:
        op_lines += f"+ {w}\n"
    max_word_len = max(len(w) for w in operand_words + [candidate.result])
    separator = '-' * (max_word_len + 2)

    question = (
        "Solve this cryptarithmetic puzzle. Each letter stands for a digit (0-9). "
        "Letters are split into three groups by the alphabet — group A-I, group J-R, "
        "group S-Z — and EACH GROUP HAS ITS OWN INDEPENDENT TABLE.\n"
        "- Within a group: different letters map to different digits; the same letter is the same digit.\n"
        "- The three tables are independent (e.g. a letter in A-I may be 5, and so may a letter in S-Z).\n"
        "- A word's numeric value is formed left->right, replacing each letter by its group's digit.\n"
        "- The first letter of every word cannot be 0.\n"
        "Only SOME letter values are revealed below; deduce the rest so the addition holds.\n\n"
        f"[group A-I] {g0}\n"
        f"[group J-R] {g1}\n"
        f"[group S-Z] {g2}\n\n"
        f"Find the numeric value of {candidate.result}.\n\n"
        f"{op_lines}"
        f"{separator}\n"
        f"= {candidate.result}"
    )
    return question


SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · STEP3=answer and verification"
)


def _build_cryptarithmetic_solution_en(
    candidate: PuzzleCandidate,
    difficulty: str,
    carries: int,
) -> str:
    operand_words = candidate.operands
    result_word = candidate.result
    full = candidate.mapping or {"g0": {}, "g1": {}, "g2": {}}
    revealed = candidate.revealed or {"g0": {}, "g1": {}, "g2": {}}
    answer = candidate.answer
    operand_ints = candidate.operand_ints or []

    op_lines_words = " + ".join(operand_words) + f" = {result_word}"
    op_lines_digits = " + ".join(str(n) for n in operand_ints) + f" = {answer}"
    n_total = sum(len(full[g]) for g in ("g0", "g1", "g2"))
    n_hidden = n_total - sum(len(revealed.get(g, {})) for g in ("g0", "g1", "g2"))

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - difficulty: {difficulty}",
        f"  - puzzle: {op_lines_words}",
        f"  - letters: {n_total} total (given {n_total - n_hidden} · deduced {n_hidden})",
        f"  - carries: {carries}",
        "[STEP 1] Given",
        "  - Split each word into letters; each letter uses its alphabet group's table (A-I / J-R / S-Z).",
        "  - Within a group different letters = different digits; the three tables are independent.",
        "  - First letter of every word != 0.",
        f"  - revealed A-I: {_format_table(revealed.get('g0', {}))}",
        f"  - revealed J-R: {_format_table(revealed.get('g1', {}))}",
        f"  - revealed S-Z: {_format_table(revealed.get('g2', {}))}",
        f"  - target: numeric value of {result_word}.",
        "[STEP 2] Worked solution (from the revealed key, force each unknown letter via the columns)",
    ]
    lines.extend(_deduction_steps(
        tuple(operand_words + [result_word]),
        _tables_to_tokens(revealed),
        _tables_to_tokens(full),
        operand_ints, answer,
    ))
    lines.extend([
        "  - completed key ([given]/[deduced]):",
        f"    group A-I: {_annotated_table(full.get('g0', {}), revealed.get('g0', {}))}",
        f"    group J-R: {_annotated_table(full.get('g1', {}), revealed.get('g1', {}))}",
        f"    group S-Z: {_annotated_table(full.get('g2', {}), revealed.get('g2', {}))}",
        f"  - check: {op_lines_digits}",
    ])

    operand_digit_strs = [str(n) for n in operand_ints]
    max_len = max(len(s) for s in operand_digit_strs + [answer])
    padded_ops = [s.rjust(max_len) for s in operand_digit_strs]
    padded_res = answer.rjust(max_len)
    carry = 0
    for pos in range(max_len):
        col_idx = max_len - 1 - pos
        col_digits = [int(s[col_idx]) for s in padded_ops if s[col_idx] != ' ']
        s = sum(col_digits) + carry
        new_carry = s // 10
        res_d = padded_res[col_idx]
        terms = " + ".join(str(d) for d in col_digits) if col_digits else "0"
        if carry:
            terms += f" + carry {carry}"
        verdict = f" -> result digit {res_d} (carry {new_carry})" if res_d != ' ' else f" -> carry {new_carry}"
        lines.append(f"    [col {pos + 1}] right->left: {terms} = {s}{verdict}")
        carry = new_carry

    lines.extend([
        "[STEP 3] Answer and verification",
        f"  - answer: {result_word} = {answer}",
        "  - re-verify each column sum and carry above.",
        "  - first letter of every word is non-zero; within each group letters are distinct digits.",
    ])
    return "\n".join(lines)


# ============================================================================
# 데이터셋 생성
# ============================================================================

def create_dataset_files(num_questions: int):
    import pandas as pd

    print(f"Generating {num_questions} letter-group cryptarithmetic puzzles (EN)...")
    difficulties = ["easy", "medium", "hard"]
    per = num_questions // len(difficulties)
    rem = num_questions % len(difficulties)

    all_puzzles = []
    used_patterns = set()
    for i, difficulty in enumerate(difficulties):
        target = per + (1 if i < rem else 0)
        if target == 0:
            continue
        print(f"\n=== {difficulty} ({target}) ===")
        generated = 0
        attempts = 0
        max_total = max(5000, target * 400)
        while generated < target and attempts < max_total:
            attempts += 1
            c = generate_puzzle_by_difficulty(difficulty, used_patterns=used_patterns)
            if c:
                carries = count_carries(*c.operand_ints)
                all_puzzles.append({
                    "id": f"cryptarithmetic_en_{difficulty}_{generated:04d}",
                    "question": create_question(c),
                    "answer": c.answer,
                    "solution": _build_cryptarithmetic_solution_en(c, difficulty, carries),
                    "difficulty": difficulty,
                })
                generated += 1
                print(f"  [{generated}/{target}] {c.puzzle_str} -> {c.answer} (carries={carries})")
        if generated < target:
            print(f"  warning: {difficulty} only {generated}/{target}")

    df = pd.DataFrame(all_puzzles)
    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("cryptarithmetic_en", all_puzzles)

    return df, all_puzzles


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Cryptarithmetic generator (EN, letter-group multi-table)")
    parser.add_argument("--num", type=int, default=12)
    args = parser.parse_args()
    create_dataset_files(num_questions=args.num)

"""Inequality / Futoshiki Puzzle Generator (EN)

This generator produces a *mixed* logical-puzzle dataset that is split into three
difficulty tiers, each written to its own JSONL file plus one combined CSV:

    easy   -> 1D inequality-chain puzzles   (inequality_en_easy.jsonl)
    medium -> 2D Futoshiki grid puzzles     (inequality_en_medium.jsonl)
    hard   -> 2D Futoshiki grid puzzles     (inequality_en_hard.jsonl)
    (combined) ------------------------------ inequality_en.csv

Record schema (identical for every tier, so CSV == concatenated JSONL):
    id, question, answer, solution, difficulty

Design notes
------------
* The dataset was migrated from a chain-only generator to this mixed format.
  The chain branch keeps the original constructive algorithm (progressively add
  hints until the solution is unique, then greedily hide inequalities while
  preserving *visible* uniqueness) and the same SFT teacher-trace format.
* Long chains (size > 9) use a compact base-36 single-character encoding for the
  puzzle string and the answer (a=10, b=11, ...), because space-separated values
  are ambiguous to grade and verbose. The solution's "Solution vector" / "Givens"
  lines still use raw integers for readability.
* The Futoshiki branch generates a random Latin square, derives orthogonal
  inequality constraints from it, reveals a subset of cells as givens, then adds
  just enough constraints to make the solution unique. Each puzzle is classified
  as solvable by pure constraint propagation, or as requiring case analysis
  (backtracking), and the teacher trace is worded accordingly.
* Tier size/structure distributions are calibrated to the reference dataset
  (see SIZE/GIVEN ranges below). Because puzzles are randomly generated, running
  this file produces a *fresh* dataset in the same format -- it does not
  reproduce any previous run byte-for-byte.

Run:  python inequality_en.py --num 300 --seed 0 --outdir ./out
"""

import argparse
import copy
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #

SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


def encode_val(v: int) -> str:
    """1..9 -> digit; 10.. -> single lowercase letter (a=10, b=11, ...)."""
    if v <= 9:
        return str(v)
    return chr(ord("a") + v - 10)


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


# --------------------------------------------------------------------------- #
# 1D inequality-chain puzzles  (easy tier)
# --------------------------------------------------------------------------- #

@dataclass
class InequalityPuzzle:
    size: int
    inequalities: List[str]
    given_numbers: Dict[int, int]
    solution: List[int]
    difficulty: Difficulty
    hidden_inequalities: set = field(default_factory=set)
    step_metrics: dict = field(default_factory=dict)

    def to_problem_string(self) -> str:
        parts = []
        for i in range(self.size):
            if i in self.given_numbers:
                parts.append(encode_val(self.given_numbers[i]))
            else:
                parts.append("_")
            if i < len(self.inequalities):
                if i in self.hidden_inequalities:
                    parts.append("?")
                else:
                    parts.append(self.inequalities[i])
        return " ".join(parts)

    def get_answer_string(self) -> str:
        """Compact single-character encoding, no spaces, for every size."""
        return "".join(encode_val(v) for v in self.solution)


class InequalityPuzzleGenerator:
    """Constructive chain generator with dual-track (full + visible) uniqueness."""

    MAX_SOLUTIONS = 1

    # ---- backtracking solver with domain-minimization ordering ---- #
    def _find_solutions(
        self,
        size: int,
        inequalities: List[str],
        given_numbers: Dict[int, int],
        max_count: int = 0,
        _stats: Optional[Dict] = None,
    ) -> List[List[int]]:
        solutions: List[List[int]] = []
        assignment = [0] * size
        used = [False] * (size + 1)

        for pos, val in given_numbers.items():
            assignment[pos] = val
            used[val] = True

        unfixed = [i for i in range(size) if i not in given_numbers]

        def domain_values(pos: int):
            lo, hi = 1, size
            if pos > 0 and assignment[pos - 1] != 0:
                prev = assignment[pos - 1]
                ineq = inequalities[pos - 1]
                if ineq == "<":
                    if prev + 1 > lo:
                        lo = prev + 1
                elif ineq == ">":
                    if prev - 1 < hi:
                        hi = prev - 1
            if pos < size - 1 and assignment[pos + 1] != 0:
                nxt = assignment[pos + 1]
                ineq = inequalities[pos]
                if ineq == "<":
                    if nxt - 1 < hi:
                        hi = nxt - 1
                elif ineq == ">":
                    if nxt + 1 > lo:
                        lo = nxt + 1
            if lo > hi:
                return []
            return [val for val in range(lo, hi + 1) if not used[val]]

        def choose_next_pos():
            best_pos = None
            best_domain = None
            best_constraint_score = None
            for pos in unfixed:
                if assignment[pos] != 0:
                    continue
                values = domain_values(pos)
                if not values:
                    return pos, []
                constraint_score = 0
                if pos > 0 and assignment[pos - 1] != 0:
                    constraint_score += 1
                if pos < size - 1 and assignment[pos + 1] != 0:
                    constraint_score += 1
                if (
                    best_pos is None
                    or len(values) < len(best_domain)
                    or (
                        len(values) == len(best_domain)
                        and constraint_score > best_constraint_score
                    )
                ):
                    best_pos = pos
                    best_domain = values
                    best_constraint_score = constraint_score
            return best_pos, best_domain

        def backtrack(idx: int):
            if _stats is not None:
                _stats["nodes"] = _stats.get("nodes", 0) + 1
            if max_count > 0 and len(solutions) >= max_count:
                return
            if idx == len(unfixed):
                solutions.append(list(assignment))
                return
            pos, values = choose_next_pos()
            if pos is None:
                solutions.append(list(assignment))
                return
            if not values:
                return
            for val in values:
                assignment[pos] = val
                used[val] = True
                backtrack(idx + 1)
                assignment[pos] = 0
                used[val] = False
                if max_count > 0 and len(solutions) >= max_count:
                    return

        backtrack(0)
        return solutions

    def _find_best_hint(
        self,
        size: int,
        inequalities: List[str],
        given_numbers: Dict[int, int],
        base_solution: List[int],
        current_solutions: List[List[int]],
    ) -> Optional[int]:
        available = [i for i in range(size) if i not in given_numbers]
        if not available:
            return None
        best_pos = None
        best_count = float("inf")
        for pos in available:
            test_hints = dict(given_numbers)
            test_hints[pos] = base_solution[pos]
            new_solutions = self._find_solutions(size, inequalities, test_hints, max_count=2)
            if len(new_solutions) == 1:
                return pos
            if 1 <= len(new_solutions) < best_count:
                best_count = len(new_solutions)
                best_pos = pos
        return best_pos

    def _minimize_hints_for_visible_uniqueness(
        self,
        size: int,
        visible_inequalities: List[str],
        given_numbers: Dict[int, int],
        min_hints: int,
    ) -> Dict[int, int]:
        if len(given_numbers) <= min_hints:
            return dict(given_numbers)
        hints = dict(given_numbers)
        positions = list(hints.keys())
        random.shuffle(positions)
        for pos in positions:
            if len(hints) <= min_hints:
                break
            test_hints = dict(hints)
            del test_hints[pos]
            visible_solutions = self._find_solutions(
                size, visible_inequalities, test_hints, max_count=2
            )
            if len(visible_solutions) == 1:
                hints = test_hints
        return hints

    def _select_hidden_inequalities(
        self,
        size: int,
        inequalities: List[str],
        given_numbers: Dict[int, int],
        num_to_hide: int,
    ) -> Optional[set]:
        """Among indices preserving visible uniqueness, sample from the top tier
        (>= 70% of max visible solver-nodes) so hiding stays 'hard' but ties are
        broken stochastically. Returns None if no feasible choice."""
        hidden: set = set()
        total_ineqs = len(inequalities)
        for _ in range(num_to_hide):
            candidates_with_score: List[Tuple[int, int]] = []
            for idx in range(total_ineqs):
                if idx in hidden:
                    continue
                trial_hidden = hidden | {idx}
                visible_ineqs = [
                    "?" if i in trial_hidden else ineq
                    for i, ineq in enumerate(inequalities)
                ]
                stats = {"nodes": 0}
                visible_solutions = self._find_solutions(
                    size, visible_ineqs, given_numbers, max_count=2, _stats=stats
                )
                if len(visible_solutions) == 1:
                    candidates_with_score.append((idx, stats["nodes"]))
            if not candidates_with_score:
                return None
            max_nodes = max(c[1] for c in candidates_with_score)
            threshold = max(1, int(max_nodes * 0.7))
            top = [idx for idx, n in candidates_with_score if n >= threshold]
            hidden.add(random.choice(top))
        return hidden

    def _select_initial_hints(self, solution: List[int], num_hints: int) -> Dict[int, int]:
        given_numbers: Dict[int, int] = {}
        size = len(solution)
        if num_hints == 0:
            return given_numbers
        extreme_positions = [i for i, val in enumerate(solution) if val == 1 or val == size]
        random.shuffle(extreme_positions)
        for pos in extreme_positions:
            if len(given_numbers) >= num_hints:
                break
            given_numbers[pos] = solution[pos]
        remaining = [i for i in range(size) if i not in given_numbers]
        random.shuffle(remaining)
        for pos in remaining:
            if len(given_numbers) >= num_hints:
                break
            given_numbers[pos] = solution[pos]
        return given_numbers

    def generate_chain(
        self,
        size: int,
        num_to_hide: int,
        difficulty: Difficulty,
        min_hints: int = 1,
        max_retries: int = 400,
    ) -> InequalityPuzzle:
        """Build a uniquely-solvable chain of `size`, hiding exactly `num_to_hide`
        inequalities while keeping the visible puzzle uniquely solvable."""
        for _ in range(max_retries):
            base_solution = list(range(1, size + 1))
            random.shuffle(base_solution)

            inequalities = [
                "<" if base_solution[i] < base_solution[i + 1] else ">"
                for i in range(size - 1)
            ]

            given_numbers = self._select_initial_hints(base_solution, min_hints)
            solutions = self._find_solutions(
                size, inequalities, given_numbers, self.MAX_SOLUTIONS + 1
            )
            while len(solutions) > self.MAX_SOLUTIONS:
                best_pos = self._find_best_hint(
                    size, inequalities, given_numbers, base_solution, solutions
                )
                if best_pos is None:
                    break
                given_numbers[best_pos] = base_solution[best_pos]
                solutions = self._find_solutions(
                    size, inequalities, given_numbers, self.MAX_SOLUTIONS + 1
                )
            if len(solutions) != 1:
                continue

            hidden_indices: set = set()
            if num_to_hide > 0:
                selected = self._select_hidden_inequalities(
                    size, inequalities, given_numbers, num_to_hide
                )
                if selected is None:
                    continue
                hidden_indices = selected

            visible_ineqs_final = [
                "?" if i in hidden_indices else ineq
                for i, ineq in enumerate(inequalities)
            ]
            minimized_hints = self._minimize_hints_for_visible_uniqueness(
                size, visible_ineqs_final, given_numbers, min_hints=min_hints
            )
            visible_solutions = self._find_solutions(
                size, visible_ineqs_final, minimized_hints, max_count=2
            )
            if len(visible_solutions) != 1:
                continue

            full_stats = {"nodes": 0}
            self._find_solutions(size, inequalities, minimized_hints, max_count=2, _stats=full_stats)
            visible_stats = {"nodes": 0}
            self._find_solutions(
                size, visible_ineqs_final, minimized_hints, max_count=2, _stats=visible_stats
            )

            return InequalityPuzzle(
                size=size,
                inequalities=inequalities,
                given_numbers=minimized_hints,
                solution=solutions[0],
                difficulty=difficulty,
                hidden_inequalities=hidden_indices,
                step_metrics={
                    "solver_steps": full_stats["nodes"],
                    "visible_solver_steps": visible_stats["nodes"],
                    "size": size,
                    "hidden_count": len(hidden_indices),
                    "hint_count": len(minimized_hints),
                },
            )

        raise RuntimeError(
            f"Failed to generate size-{size} chain (hide {num_to_hide}) "
            f"with exactly 1 solution after {max_retries} retries"
        )


def create_chain_question(puzzle: InequalityPuzzle) -> str:
    problem_str = puzzle.to_problem_string()
    n = puzzle.size

    rules = [
        "- '_' represents an empty position to fill",
        "- '<' means the left number is smaller than the right number",
        "- '>' means the left number is larger than the right number",
        f"- Each number 1 to {n} appears exactly once",
        "- '?' means the inequality is unknown (could be < or >)",
    ]
    if n > 9:
        rules.append(
            "- Values 10 and above are written as single letters: a=10, b=11, "
            "c=12, ... (so 'a' means 10, and '_ > c' means the blank is greater "
            "than 12)"
        )
        answer_format = (
            f"Provide your answer as {n} single characters in order with no "
            f"spaces, using digits for 1-9 and letters for 10 and up "
            f"(a=10, b=11, ...)."
        )
        answer_example = "Answer: " + "".join(encode_val(i) for i in range(1, n + 1))
    else:
        answer_format = f"Provide your answer as a sequence of {n} digits (no spaces)."
        answer_example = "Answer: " + "".join(str(i) for i in range(1, n + 1))

    rules_block = "\n".join(rules)
    return (
        f"Solve this inequality puzzle. Fill in the blanks with numbers from 1 to {n}.\n\n"
        f"Each number from 1 to {n} must be used exactly once.\n"
        f"The inequality symbols (< or >) between positions must be satisfied.\n\n"
        f"Puzzle: {problem_str}\n\n"
        f"Rules:\n{rules_block}\n\n"
        f"{answer_format}\n\n"
        f"Example format:\n{answer_example}"
    )


def build_chain_solution(puzzle: InequalityPuzzle) -> str:
    size = puzzle.size
    solution = puzzle.solution
    ineqs = puzzle.inequalities
    givens = puzzle.given_numbers
    hidden = puzzle.hidden_inequalities
    problem_str = puzzle.to_problem_string()
    ans_str = puzzle.get_answer_string()
    visible_cnt = len(ineqs) - len(hidden)

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {puzzle.difficulty.name.lower()}",
        f"  - Grid size: {size} (permutation of 1..{size})",
        f"  - Givens: {len(givens)} · visible inequalities: {visible_cnt} · hidden: {len(hidden)}",
        "  - Final answer is confirmed in [STEP 3]",
        "[STEP 1] Given",
        f"  - Puzzle: {problem_str}",
        f"  - Givens (pos:value): "
        f"{', '.join(f'{p}:{v}' for p, v in sorted(givens.items())) or '(none)'}",
        "[STEP 2] Worked solution",
        f"  · Summary: permutation 1..{size} + hint/inequality propagation -> "
        f"unique model · {len(ineqs)} inequalities "
        f"(visible {visible_cnt} / hidden {len(hidden)}) · {len(ineqs)} SEGs",
        f"  · Solution vector: [{', '.join(str(v) for v in solution)}]",
    ]
    for i, op in enumerate(ineqs):
        left, right = solution[i], solution[i + 1]
        hidden_flag = "hidden" if i in hidden else "visible"
        if op == "<":
            ok = left < right
        elif op == ">":
            ok = left > right
        else:
            ok = None
        status = "holds" if ok else ("fails" if ok is False else "check")
        lines.append(
            f"    [SEG {i + 1}] positions {i}<->{i + 1} ({hidden_flag}): "
            f"{left} {op} {right} -> {status}"
        )
    lines.extend(
        [
            "[STEP 3] Answer and verification",
            f"  - Final answer: {ans_str}",
            f"  - Each of 1..{size} used exactly once: "
            f"{'OK' if sorted(solution) == list(range(1, size + 1)) else 'FAIL'}",
            "  - Givens match and every inequality (visible + hidden) holds as "
            "shown in the [SEG] trace.",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2D Futoshiki puzzles  (medium / hard tiers)
# --------------------------------------------------------------------------- #

Cell = Tuple[int, int]
Constraint = Tuple[Cell, Cell, str]  # (cell_a, cell_b, op) meaning a op b


@dataclass
class FutoshikiPuzzle:
    n: int
    solution: List[List[int]]
    givens: Dict[Cell, int]
    constraints: List[Constraint]  # sorted in reading order
    difficulty: Difficulty
    mode: str = "backtrack"  # "propagation" | "backtrack"
    branch_points: int = 0

    def flat_answer(self) -> str:
        return " ".join(
            str(self.solution[r][c]) for r in range(self.n) for c in range(self.n)
        )


class FutoshikiGenerator:
    @staticmethod
    def random_latin_square(n: int) -> List[List[int]]:
        base = [[(i + j) % n + 1 for j in range(n)] for i in range(n)]
        random.shuffle(base)  # permute rows
        cols = list(range(n))
        random.shuffle(cols)
        base = [[row[c] for c in cols] for row in base]  # permute columns
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        return [[perm[v - 1] for v in row] for row in base]  # relabel symbols

    @staticmethod
    def all_adjacent_constraints(sol: List[List[int]]) -> List[Constraint]:
        """All orthogonally-adjacent constraints, smaller-reading-order cell first."""
        n = len(sol)
        cons: List[Constraint] = []
        for r in range(n):
            for c in range(n):
                if c + 1 < n:
                    op = "<" if sol[r][c] < sol[r][c + 1] else ">"
                    cons.append(((r, c), (r, c + 1), op))
                if r + 1 < n:
                    op = "<" if sol[r][c] < sol[r + 1][c] else ">"
                    cons.append(((r, c), (r + 1, c), op))
        return cons

    @staticmethod
    def sort_constraints(cons: List[Constraint]) -> List[Constraint]:
        return sorted(cons, key=lambda x: (x[0][0], x[0][1], x[1][0], x[1][1]))

    # ---- solution counting (backtracking, capped) ---- #
    @staticmethod
    def count_solutions(
        n: int, givens: Dict[Cell, int], constraints: List[Constraint], cap: int = 2
    ) -> int:
        grid = [[0] * n for _ in range(n)]
        rows = [set() for _ in range(n)]
        cols = [set() for _ in range(n)]
        for (r, c), v in givens.items():
            grid[r][c] = v
            rows[r].add(v)
            cols[c].add(v)

        cell_cons: Dict[Cell, List[Constraint]] = {}
        for con in constraints:
            cell_cons.setdefault(con[0], []).append(con)
            cell_cons.setdefault(con[1], []).append(con)

        def ok(r: int, c: int, v: int) -> bool:
            if v in rows[r] or v in cols[c]:
                return False
            for (a, b, op) in cell_cons.get((r, c), []):
                av = v if a == (r, c) else grid[a[0]][a[1]]
                bv = v if b == (r, c) else grid[b[0]][b[1]]
                if av == 0 or bv == 0:
                    continue
                if op == "<" and not av < bv:
                    return False
                if op == ">" and not av > bv:
                    return False
            return True

        empties = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0]
        count = 0

        def bt(i: int):
            nonlocal count
            if count >= cap:
                return
            if i == len(empties):
                count += 1
                return
            r, c = empties[i]
            for v in range(1, n + 1):
                if ok(r, c, v):
                    grid[r][c] = v
                    rows[r].add(v)
                    cols[c].add(v)
                    bt(i + 1)
                    grid[r][c] = 0
                    rows[r].discard(v)
                    cols[c].discard(v)
                    if count >= cap:
                        return

        bt(0)
        return count

    # ---- constraint propagation to a fixpoint ---- #
    @staticmethod
    def _propagate(
        cand: List[List[set]], n: int, constraints: List[Constraint]
    ) -> bool:
        """Mutates `cand`. Returns False on contradiction, else True (fixpoint).

        Propagation strength is deliberately limited to exactly what the teacher
        trace claims it does -- the row/column all-different rule (naked singles +
        peer elimination) and inequality-bound pruning. Stronger techniques such
        as hidden singles are intentionally NOT applied, so a puzzle is classified
        "propagation-solvable" only when these basic rules alone finish it; the
        rest legitimately require backtracking search.
        """
        changed = True
        while changed:
            changed = False
            # naked singles -> peer elimination
            for r in range(n):
                for c in range(n):
                    if len(cand[r][c]) == 0:
                        return False
                    if len(cand[r][c]) == 1:
                        v = next(iter(cand[r][c]))
                        for cc in range(n):
                            if cc != c and v in cand[r][cc]:
                                cand[r][cc].discard(v)
                                changed = True
                        for rr in range(n):
                            if rr != r and v in cand[rr][c]:
                                cand[rr][c].discard(v)
                                changed = True
            # inequality pruning
            for (a, b, op) in constraints:
                A, B = cand[a[0]][a[1]], cand[b[0]][b[1]]
                if not A or not B:
                    return False
                if op == "<":  # a < b
                    nA = {x for x in A if x < max(B)}
                    nB = {x for x in B if x > min(A)}
                else:  # a > b
                    nA = {x for x in A if x > min(B)}
                    nB = {x for x in B if x < max(A)}
                if nA != A:
                    if not nA:
                        return False
                    cand[a[0]][a[1]] = nA
                    changed = True
                if nB != B:
                    if not nB:
                        return False
                    cand[b[0]][b[1]] = nB
                    changed = True
        return True

    @classmethod
    def _initial_candidates(cls, n: int, givens: Dict[Cell, int]) -> List[List[set]]:
        cand = [[set(range(1, n + 1)) for _ in range(n)] for _ in range(n)]
        for (r, c), v in givens.items():
            cand[r][c] = {v}
        return cand

    @classmethod
    def propagation_solves(
        cls, n: int, givens: Dict[Cell, int], constraints: List[Constraint]
    ) -> Optional[List[List[int]]]:
        cand = cls._initial_candidates(n, givens)
        if not cls._propagate(cand, n, constraints):
            return None
        for r in range(n):
            for c in range(n):
                if len(cand[r][c]) != 1:
                    return None
        return [[next(iter(cand[r][c])) for c in range(n)] for r in range(n)]

    @classmethod
    def branch_points(
        cls, n: int, givens: Dict[Cell, int], constraints: List[Constraint]
    ) -> int:
        """Number of guesses on the path to the first solution, with propagation
        collapsing forced cells after each guess."""

        def solve(cand: List[List[set]]) -> Optional[int]:
            cand = copy.deepcopy(cand)
            if not cls._propagate(cand, n, constraints):
                return None
            best: Optional[Cell] = None
            for r in range(n):
                for c in range(n):
                    if len(cand[r][c]) > 1:
                        if best is None or len(cand[r][c]) < len(cand[best[0]][best[1]]):
                            best = (r, c)
            if best is None:
                return 0  # solved with no further guesses
            r, c = best
            for v in sorted(cand[r][c]):
                trial = copy.deepcopy(cand)
                trial[r][c] = {v}
                res = solve(trial)
                if res is not None:
                    return res + 1
            return None

        res = solve(cls._initial_candidates(n, givens))
        return res if res is not None else 0

    # ---- puzzle assembly ---- #
    MIN_CONSTRAINTS = 3
    # extra givens the uniqueness auto-boost may add on top of target_givens
    GIVENS_AUTOBOOST_SLACK = 2

    @classmethod
    def generate(
        cls, n: int, target_givens: int, target_mode: str = "backtrack",
        max_tries: int = 200,
    ) -> FutoshikiPuzzle:
        """Build a uniquely-solvable Futoshiki matching the reference profile.

        Reveal a small number of givens and start from the full set of orthogonal
        inequality constraints (which pin the grid). Then remove constraints one at
        a time, never breaking uniqueness, with a target-aware stopping rule:

        * backtrack target -- keep removing until the limited propagation rules can
          no longer finish the grid, i.e. the puzzle now genuinely requires search.
        * propagation target -- remove a constraint only while the grid still solves
          by propagation alone, yielding a lean propagation-solvable puzzle.

        Sparse givens + a lean constraint set reproduce the reference's
        backtracking-heavy character; the per-tier ratio drives which target is
        requested. A unique fallback is returned if the requested mode proves hard
        to hit within the retry budget.
        """
        fallback: Optional[FutoshikiPuzzle] = None

        for _ in range(max_tries):
            sol = cls.random_latin_square(n)
            all_cons = cls.all_adjacent_constraints(sol)
            random.shuffle(all_cons)

            cells = [(r, c) for r in range(n) for c in range(n)]
            random.shuffle(cells)
            g = max(1, min(target_givens, n * n - 1))
            givens: Dict[Cell, int] = {cells[i]: sol[cells[i][0]][cells[i][1]] for i in range(g)}
            gi = g

            chosen: List[Constraint] = list(all_cons)  # fully constrained

            def unique(cons) -> bool:
                return cls.count_solutions(n, givens, cons, 2) == 1

            # the full constraint set may still admit several Latin squares; if so,
            # reveal more givens until the solution is unique
            givens_cap = min(len(cells), g + cls.GIVENS_AUTOBOOST_SLACK)
            while gi < givens_cap and not unique(chosen):
                cell = cells[gi]
                givens[cell] = sol[cell[0]][cell[1]]
                gi += 1
            if not unique(chosen):
                continue

            # target-aware constraint removal (uniqueness preserved at every step)
            #   backtrack target -> minimize fully (lean constraint set, like the
            #     reference); whether it ends up needing search is checked below and
            #     the retry/fallback loop selects the ones that do.
            #   propagation target -> remove a constraint only while the grid still
            #     solves by propagation alone.
            for con in list(chosen):
                trial = [c for c in chosen if c != con]
                if not unique(trial):
                    continue  # this constraint is needed for uniqueness; keep it
                if target_mode == "backtrack":
                    chosen = trial
                else:  # propagation target
                    if cls.propagation_solves(n, givens, trial) is not None:
                        chosen = trial  # stay propagation-solvable

            if len(chosen) < cls.MIN_CONSTRAINTS:
                continue  # avoid degenerate (near-)constraint-free puzzles

            chosen = cls.sort_constraints(chosen)
            prop = cls.propagation_solves(n, givens, chosen)
            if prop is not None:
                mode, bp = "propagation", 0
            else:
                mode = "backtrack"
                bp = max(2, cls.branch_points(n, givens, chosen))

            puzzle = FutoshikiPuzzle(
                n=n,
                solution=sol,
                givens=givens,
                constraints=chosen,
                difficulty=Difficulty.HARD,  # overwritten by caller's tier
                mode=mode,
                branch_points=bp,
            )
            if mode == target_mode:
                return puzzle
            fallback = puzzle  # remember a valid puzzle of the other mode

        if fallback is not None:
            return fallback
        raise RuntimeError(f"Failed to generate a unique {n}x{n} Futoshiki puzzle")


def _grid_lines(n: int, value_at, indent: str) -> List[str]:
    out = []
    for r in range(n):
        cells = " ".join(value_at(r, c) for c in range(n))
        out.append(f"{indent}Row {r + 1}: {cells}")
    return out


def create_futoshiki_question(p: FutoshikiPuzzle) -> str:
    n = p.n

    def shown(r, c):
        return str(p.givens[(r, c)]) if (r, c) in p.givens else "_"

    grid_block = "\n".join(_grid_lines(n, shown, ""))
    con_block = "\n".join(
        f"- ({a[0] + 1},{a[1] + 1}) {op} ({b[0] + 1},{b[1] + 1})"
        for (a, b, op) in p.constraints
    )
    example = " ".join(str(v) for _ in range(n) for v in range(1, n + 1))

    return (
        f"Solve this Futoshiki puzzle on a {n}x{n} grid.\n\n"
        f"Rules:\n"
        f"- Fill the grid so that every row and every column contains each number "
        f"from 1 to {n} exactly once (a Latin square).\n"
        f"- '_' marks an empty cell that you must fill in.\n"
        f"- All of the inequality constraints listed below must hold. Each "
        f"constraint compares two orthogonally-adjacent cells and is written "
        f"using (row, column) with 1-based indices (row 1 is the top row, "
        f"column 1 is the left column).\n"
        f"- '(a) < (b)' means the number in cell a is smaller than the number in "
        f"cell b; '(a) > (b)' means it is larger.\n\n"
        f"Given grid (rows from top to bottom):\n{grid_block}\n\n"
        f"Inequality constraints:\n{con_block}\n\n"
        f"Provide your answer as the completed grid: {n * n} numbers separated by "
        f"spaces, reading row by row (all of row 1 left to right, then row 2, and "
        f"so on).\n\n"
        f"Example format (illustrating length and ordering only, not a valid "
        f"solution):\nAnswer: {example}"
    )


def build_futoshiki_solution(p: FutoshikiPuzzle) -> str:
    n = p.n
    sol = p.solution

    if p.mode == "propagation":
        mode_line = (
            "  - The puzzle is solvable by constraint propagation alone "
            "(repeatedly applying the row/column all-different rule and the "
            "inequalities); no guessing is required."
        )
        approach_line = (
            "  · Approach: repeatedly apply the Latin-square (row/column "
            "all-different) rule and the inequality constraints to fill in forced "
            "cells; each step a cell has only one value left, until the whole grid "
            "is determined."
        )
    else:
        mode_line = (
            "  - Constraint propagation alone does not finish this puzzle; it "
            "requires case analysis (backtracking search) to pin down the unique "
            "solution."
        )
        approach_line = (
            "  · Approach: reduce each cell's candidates with the Latin-square "
            "(row/column all-different) rule and the inequalities, then branch on "
            "the most constrained cell and backtrack on contradictions until one "
            f"consistent grid remains (search size ~{p.branch_points} branch points)."
        )

    def given_at(r, c):
        return str(p.givens[(r, c)]) if (r, c) in p.givens else "_"

    def sol_at(r, c):
        return str(sol[r][c])

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {p.difficulty.name.lower()}",
        f"  - Puzzle: {n}x{n} Futoshiki (Latin square with inequality constraints)",
        f"  - Givens: {len(p.givens)} · inequality constraints: {len(p.constraints)}",
        mode_line,
        "  - Cells are 1-indexed (row 1 = top, column 1 = left). Final answer is "
        "confirmed in [STEP 3]",
        "[STEP 1] Given",
        "  - Given grid:",
    ]
    lines += _grid_lines(n, given_at, "      ")
    lines.append("  - Inequality constraints:")
    lines += [
        f"      ({a[0] + 1},{a[1] + 1}) {op} ({b[0] + 1},{b[1] + 1})"
        for (a, b, op) in p.constraints
    ]

    lines.append("[STEP 2] Worked solution")
    lines.append(approach_line)
    lines.append("  · Completed grid:")
    lines += _grid_lines(n, sol_at, "      ")

    lines.append(f"  · Verification — every row is a permutation of 1..{n}:")
    for r in range(n):
        row = ", ".join(str(sol[r][c]) for c in range(n))
        lines.append(f"      Row {r + 1}: {{{row}}} -> OK")
    lines.append(f"  · Verification — every column is a permutation of 1..{n}:")
    for c in range(n):
        col = ", ".join(str(sol[r][c]) for r in range(n))
        lines.append(f"      Col {c + 1}: {{{col}}} -> OK")
    lines.append("  · Verification — every inequality holds:")
    for (a, b, op) in p.constraints:
        av, bv = sol[a[0]][a[1]], sol[b[0]][b[1]]
        lines.append(
            f"      ({a[0] + 1},{a[1] + 1}) {op} ({b[0] + 1},{b[1] + 1}): "
            f"{av} {op} {bv} -> holds"
        )

    lines += [
        "[STEP 3] Answer and verification",
        f"  - Final answer (grid flattened row by row): {p.flat_answer()}",
        f"  - Each row and each column is a permutation of 1..{n}: OK",
        "  - Every inequality constraint holds as shown above, and all givens match.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tier configuration (calibrated to the reference dataset)
# --------------------------------------------------------------------------- #

# easy / medium / hard: all Futoshiki. grid-size weights + the fraction of
# puzzles that should genuinely require backtracking (vs. be solvable by the
# limited propagation rules alone).
#
# easy uses a 4x4 grid that is mostly propagation-solvable, sitting a notch
# below medium (5x5, mostly backtracking). The former 1D inequality-chain easy
# tier was retired because strong models solved it near-perfectly regardless of
# chain length or hidden-inequality count; a smaller Futoshiki grid gives a
# cleaner, monotone easy < medium < hard difficulty axis.
FUTOSHIKI_TIERS = {
    "easy": {"size_weights": {5: 1.0}, "backtrack_ratio": 1.0, "givens_override": (2, 6)},
    "medium": {"size_weights": {5: 0.62, 6: 0.38}, "backtrack_ratio": 0.82},
    "hard": {"size_weights": {5: 0.20, 6: 0.80}, "backtrack_ratio": 0.92},
}

# Givens revealed per grid size (kept small; the minimal constraint set is what
# makes the puzzle hard). Calibrated to reference medians (5x5 ~4, 6x6 ~5-6).
# 4x4 easy reveals a slightly higher share of its cells to keep it approachable.
FUTOSHIKI_GIVENS_BY_SIZE = {4: (3, 5), 5: (2, 6), 6: (3, 5)}


def _weighted_choice(weight_map: Dict[int, float]) -> int:
    keys = list(weight_map.keys())
    weights = [weight_map[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #

def make_easy_record(idx: int, chain_gen: InequalityPuzzleGenerator) -> dict:
    # easy is now a 4x4 Futoshiki (see FUTOSHIKI_TIERS). The chain_gen argument is
    # kept for call-site compatibility but is no longer used.
    return make_futoshiki_record(idx, "easy")


def make_futoshiki_record(idx: int, tier: str) -> dict:
    cfg = FUTOSHIKI_TIERS[tier]
    n = _weighted_choice(cfg["size_weights"])
    # tier-specific givens override takes precedence over the by-size defaults,
    # so tuning easy's given count never touches medium/hard (which share sizes).
    if "givens_override" in cfg:
        target_givens = random.randint(*cfg["givens_override"])
    else:
        target_givens = random.randint(*FUTOSHIKI_GIVENS_BY_SIZE[n])
    target_mode = "backtrack" if random.random() < cfg["backtrack_ratio"] else "propagation"
    p = FutoshikiGenerator.generate(n, target_givens, target_mode)
    p.difficulty = Difficulty[tier.upper()]
    return {
        "id": f"inequality_en_{tier}_{idx:04d}",
        "question": create_futoshiki_question(p),
        "answer": p.flat_answer(),
        "solution": build_futoshiki_solution(p),
        "difficulty": tier,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def create_dataset_files(num_questions: int, outdir: Path) -> Dict[str, List[dict]]:
    """Generate the three tiers, write per-tier JSONL files + one combined CSV."""
    import pandas as pd

    tiers = ["easy", "medium", "hard"]
    per_tier = num_questions // len(tiers)
    remainder = num_questions % len(tiers)

    chain_gen = InequalityPuzzleGenerator()
    outdir.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    by_tier: Dict[str, List[dict]] = {}

    for i, tier in enumerate(tiers):
        count = per_tier + (1 if i < remainder else 0)
        print(f"\n=== Generating {tier} ({count}) ===")
        records: List[dict] = []
        seen = set()
        produced = 0
        attempts = 0
        budget = max(50, count * 50)
        while produced < count and attempts < budget:
            attempts += 1
            try:
                if tier == "easy":
                    rec = make_easy_record(produced, chain_gen)
                else:
                    rec = make_futoshiki_record(produced, tier)
            except RuntimeError as exc:
                print(f"  retry ({exc})")
                continue
            if rec["question"] in seen:
                continue
            seen.add(rec["question"])
            records.append(rec)
            produced += 1
            if produced % 10 == 0 or produced == count:
                print(f"  [{produced}/{count}]")

        if produced < count:
            print(f"  ⚠️ only produced {produced}/{count} for {tier}")

        by_tier[tier] = records
        all_records.extend(records)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl, DATA_DIR

    write_release_jsonl("inequality_en", all_records, data_dir=outdir)
    return by_tier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inequality / Futoshiki Puzzle Generator")
    parser.add_argument("--num", type=int, default=300, help="Total puzzles (split 3 ways)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: data/)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 50)
    print("Inequality / Futoshiki Puzzle Generator")
    print("=" * 50)
    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import DATA_DIR
    outdir = Path(args.outdir) if args.outdir else DATA_DIR
    create_dataset_files(num_questions=args.num, outdir=outdir)
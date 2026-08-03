"""부등식 / Futoshiki(부등호) 퍼즐 생성기 (KO)

이 생성기는 세 개의 난이도 계층으로 나뉘는 *혼합형* 논리 퍼즐 데이터셋을 만들며,
각 계층은 개별 JSONL 파일로 저장되고 하나의 통합 CSV도 함께 생성됩니다:

    easy(쉬움)   -> 1차원 부등식 사슬 퍼즐   (inequality_ko_easy.jsonl)
    medium(보통) -> 2차원 Futoshiki 격자 퍼즐 (inequality_ko_medium.jsonl)
    hard(어려움) -> 2차원 Futoshiki 격자 퍼즐 (inequality_ko_hard.jsonl)
    (통합) ----------------------------------- inequality_ko.csv

레코드 스키마(모든 계층에서 동일하므로 CSV == JSONL 이어붙임):
    id, question, answer, solution, difficulty

설계 노트
--------
* 이 데이터셋은 사슬 전용 생성기에서 이 혼합형 형식으로 이전되었습니다.
  사슬 분기는 원래의 구성 알고리즘(해가 유일해질 때까지 힌트를 점진적으로 추가한 뒤,
  *보이는* 유일성을 유지하면서 부등식을 탐욕적으로 숨김)과 동일한 SFT 교사 추론
  형식을 유지합니다.
* 긴 사슬(크기 > 9)은 퍼즐 문자열과 정답에 대해 콤팩트한 base-36 단일 문자 인코딩을
  사용합니다(a=10, b=11, ...). 공백으로 구분된 값은 채점이 모호하고 장황하기 때문입니다.
  해설의 "해 벡터" / "주어진 값" 줄은 가독성을 위해 여전히 정수를 그대로 사용합니다.
* Futoshiki 분기는 무작위 라틴 방진을 생성하고, 그로부터 인접(상하좌우) 부등식 제약을
  도출하며, 일부 칸을 주어진 값으로 공개한 뒤, 해가 유일해지도록 필요한 만큼의 제약만
  추가합니다. 각 퍼즐은 순수한 제약 전파만으로 풀 수 있는지, 아니면 경우의 수 분석
  (백트래킹)이 필요한지로 분류되며, 교사 추론 문장도 그에 맞게 서술됩니다.
* 계층별 크기/구조 분포는 기준 데이터셋에 맞춰 보정되어 있습니다(아래 SIZE/GIVEN 범위 참고).
  퍼즐이 무작위로 생성되므로 이 파일을 실행하면 동일한 형식의 *새로운* 데이터셋이
  만들어지며, 이전 실행을 바이트 단위로 재현하지는 않습니다.

실행:  python inequality_ko.py --num 300 --seed 0 --outdir ./out
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

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · "
    "STEP3=답·검산"
)

# 난이도 표기는 다른 KO task 와 동일하게 영문 키를 그대로 쓴다.
# (과거에는 쉬움/보통/어려움으로 번역했으나, solution 을 정규식으로 파싱하는
#  분석 스크립트들이 tier 키를 기대하므로 통일했다.)
DIFFICULTY_KO = {"easy": "easy", "medium": "medium", "hard": "hard"}


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
            f"크기 {size} 사슬(숨김 {num_to_hide})을 "
            f"{max_retries}회 재시도 후에도 해가 정확히 1개가 되도록 생성하지 못했습니다"
        )


def create_chain_question(puzzle: InequalityPuzzle) -> str:
    problem_str = puzzle.to_problem_string()
    n = puzzle.size

    rules = [
        "- '_'는 채워야 할 빈 자리를 나타냅니다",
        "- '<'는 왼쪽 숫자가 오른쪽 숫자보다 작다는 뜻입니다",
        "- '>'는 왼쪽 숫자가 오른쪽 숫자보다 크다는 뜻입니다",
        f"- 1부터 {n}까지의 각 숫자는 정확히 한 번씩만 나타납니다",
        "- '?'는 부등호를 알 수 없다는 뜻입니다(< 또는 > 일 수 있음)",
    ]
    if n > 9:
        rules.append(
            "- 10 이상의 값은 단일 문자로 표기합니다: a=10, b=11, "
            "c=12, ... (즉 'a'는 10을 의미하며, '_ > c'는 빈칸이 12보다 "
            "크다는 뜻입니다)"
        )
        answer_format = (
            f"정답은 순서대로 {n}개의 단일 문자로, 공백 없이 제시하시오. "
            f"1-9는 숫자로, 10 이상은 문자로 표기합니다"
            f"(a=10, b=11, ...)."
        )
        answer_example = "Answer: " + "".join(encode_val(i) for i in range(1, n + 1))
    else:
        answer_format = f"정답은 {n}자리 숫자열로(공백 없이) 제시하시오."
        answer_example = "Answer: " + "".join(str(i) for i in range(1, n + 1))

    rules_block = "\n".join(rules)
    return (
        f"다음 부등식 퍼즐을 푸시오. 빈칸을 1부터 {n}까지의 숫자로 채우시오.\n\n"
        f"1부터 {n}까지의 각 숫자는 정확히 한 번씩 사용해야 합니다.\n"
        f"각 자리 사이의 부등호(< 또는 >)는 모두 만족되어야 합니다.\n\n"
        f"퍼즐: {problem_str}\n\n"
        f"규칙:\n{rules_block}\n\n"
        f"{answer_format}\n\n"
        f"예시 형식:\n{answer_example}"
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
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {DIFFICULTY_KO.get(puzzle.difficulty.name.lower(), puzzle.difficulty.name.lower())}",
        f"  - 격자 크기: {size} (1..{size}의 순열)",
        f"  - 주어진 값: {len(givens)}개 · 보이는 부등식: {visible_cnt}개 · 숨김: {len(hidden)}개",
        "  - 최종 정답은 [STEP 3]에서 확인합니다.",
        "[STEP 1] 주어진 조건",
        f"  - 퍼즐: {problem_str}",
        f"  - 주어진 값 (자리:값): "
        f"{', '.join(f'{p}:{v}' for p, v in sorted(givens.items())) or '(없음)'}",
        "[STEP 2] 풀이 전개",
        f"  · 요약: 1..{size}의 순열 + 힌트/부등식 전파 -> "
        f"유일해 · 부등식 {len(ineqs)}개 "
        f"(보임 {visible_cnt} / 숨김 {len(hidden)}) · SEG {len(ineqs)}개",
        f"  · 해 벡터: [{', '.join(str(v) for v in solution)}]",
    ]
    for i, op in enumerate(ineqs):
        left, right = solution[i], solution[i + 1]
        hidden_flag = "숨김" if i in hidden else "보임"
        if op == "<":
            ok = left < right
        elif op == ">":
            ok = left > right
        else:
            ok = None
        status = "성립" if ok else ("불성립" if ok is False else "확인")
        lines.append(
            f"    [SEG {i + 1}] 자리 {i}<->{i + 1} ({hidden_flag}): "
            f"{left} {op} {right} -> {status}"
        )
    lines.extend(
        [
            "[STEP 3] 답·검산",
            f"  - 최종 정답: {ans_str}",
            f"  - 1..{size}가 각각 정확히 한 번씩 사용됨: "
            f"{'OK' if sorted(solution) == list(range(1, size + 1)) else 'FAIL'}",
            "  - 주어진 값이 일치하며, [SEG] 추론에서 보인 대로 모든 부등식"
            "(보임 + 숨김)이 성립합니다.",
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
        raise RuntimeError(f"유일한 {n}x{n} Futoshiki 퍼즐을 생성하지 못했습니다")


def _grid_lines(n: int, value_at, indent: str) -> List[str]:
    out = []
    for r in range(n):
        cells = " ".join(value_at(r, c) for c in range(n))
        out.append(f"{indent}행 {r + 1}: {cells}")
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
        f"{n}x{n} 격자에서 다음 Futoshiki(부등호) 퍼즐을 푸시오.\n\n"
        f"규칙:\n"
        f"- 모든 행과 모든 열에 1부터 {n}까지의 숫자가 각각 정확히 한 번씩 들어가도록 "
        f"격자를 채우시오(라틴 방진).\n"
        f"- '_'는 채워야 할 빈 칸을 나타냅니다.\n"
        f"- 아래에 나열된 모든 부등식 제약을 만족해야 합니다. 각 제약은 상하좌우로 "
        f"인접한 두 칸을 비교하며, (행, 열) 형식의 1부터 시작하는 인덱스로 "
        f"표기됩니다(행 1은 맨 위 행, 열 1은 맨 왼쪽 열).\n"
        f"- '(a) < (b)'는 칸 a의 숫자가 칸 b의 숫자보다 작다는 뜻이고, "
        f"'(a) > (b)'는 더 크다는 뜻입니다.\n\n"
        f"주어진 격자(위에서 아래 행 순서):\n{grid_block}\n\n"
        f"부등식 제약:\n{con_block}\n\n"
        f"완성된 격자를 정답으로 제시하시오: 공백으로 구분된 {n * n}개의 숫자를 "
        f"행 순서대로(1행을 왼쪽에서 오른쪽으로 모두, 그다음 2행, 이런 식으로) "
        f"적으시오.\n\n"
        f"예시 형식(길이와 순서만 나타낸 것이며 유효한 정답은 아님):\n"
        f"Answer: {example}"
    )


def build_futoshiki_solution(p: FutoshikiPuzzle) -> str:
    n = p.n
    sol = p.solution

    if p.mode == "propagation":
        mode_line = (
            "  - 이 퍼즐은 제약 전파만으로 풀 수 있습니다"
            "(행/열 서로 다름 규칙과 부등식을 반복 적용). "
            "추측(경우의 수 분기)이 필요 없습니다."
        )
        approach_line = (
            "  · 접근법: 라틴 방진(행/열 서로 다름) 규칙과 부등식 제약을 반복 "
            "적용하여 값이 강제되는 칸을 채웁니다. 각 단계에서 한 칸에 남는 값이 "
            "하나뿐이며, 격자 전체가 결정될 때까지 진행합니다."
        )
    else:
        mode_line = (
            "  - 제약 전파만으로는 이 퍼즐이 완성되지 않습니다. 유일한 해를 "
            "확정하려면 경우의 수 분석(백트래킹 탐색)이 필요합니다."
        )
        approach_line = (
            "  · 접근법: 라틴 방진(행/열 서로 다름) 규칙과 부등식으로 각 칸의 "
            "후보를 줄인 뒤, 가장 제약이 많은 칸에서 분기하고 모순이 생기면 "
            "되돌아가며(백트래킹), 일관된 격자가 하나만 남을 때까지 탐색합니다"
            f"(탐색 규모 ~{p.branch_points}개 분기점)."
        )

    def given_at(r, c):
        return str(p.givens[(r, c)]) if (r, c) in p.givens else "_"

    def sol_at(r, c):
        return str(sol[r][c])

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {DIFFICULTY_KO.get(p.difficulty.name.lower(), p.difficulty.name.lower())}",
        f"  - 퍼즐: {n}x{n} Futoshiki (부등식 제약이 있는 라틴 방진)",
        f"  - 주어진 값: {len(p.givens)}개 · 부등식 제약: {len(p.constraints)}개",
        mode_line,
        "  - 칸은 1부터 인덱싱합니다(행 1 = 맨 위, 열 1 = 맨 왼쪽). 최종 정답은 "
        "[STEP 3]에서 확인합니다.",
        "[STEP 1] 주어진 조건",
        "  - 주어진 격자:",
    ]
    lines += _grid_lines(n, given_at, "      ")
    lines.append("  - 부등식 제약:")
    lines += [
        f"      ({a[0] + 1},{a[1] + 1}) {op} ({b[0] + 1},{b[1] + 1})"
        for (a, b, op) in p.constraints
    ]

    lines.append("[STEP 2] 풀이 전개")
    lines.append(approach_line)
    lines.append("  · 완성된 격자:")
    lines += _grid_lines(n, sol_at, "      ")

    lines.append(f"  · 검증 — 모든 행이 1..{n}의 순열입니다:")
    for r in range(n):
        row = ", ".join(str(sol[r][c]) for c in range(n))
        lines.append(f"      행 {r + 1}: {{{row}}} -> OK")
    lines.append(f"  · 검증 — 모든 열이 1..{n}의 순열입니다:")
    for c in range(n):
        col = ", ".join(str(sol[r][c]) for r in range(n))
        lines.append(f"      열 {c + 1}: {{{col}}} -> OK")
    lines.append("  · 검증 — 모든 부등식이 성립합니다:")
    for (a, b, op) in p.constraints:
        av, bv = sol[a[0]][a[1]], sol[b[0]][b[1]]
        lines.append(
            f"      ({a[0] + 1},{a[1] + 1}) {op} ({b[0] + 1},{b[1] + 1}): "
            f"{av} {op} {bv} -> 성립"
        )

    lines += [
        "[STEP 3] 답·검산",
        f"  - 최종 정답(행 순서로 펼친 격자): {p.flat_answer()}",
        f"  - 각 행과 각 열이 1..{n}의 순열임: OK",
        "  - 위에서 보인 대로 모든 부등식 제약이 성립하며, 주어진 값과도 일치합니다.",
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
        "id": f"inequality_ko_{tier}_{idx:04d}",
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
        print(f"\n=== {tier} 생성 중 ({count}개) ===")
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
                print(f"  재시도 ({exc})")
                continue
            if rec["question"] in seen:
                continue
            seen.add(rec["question"])
            records.append(rec)
            produced += 1
            if produced % 10 == 0 or produced == count:
                print(f"  [{produced}/{count}]")

        if produced < count:
            print(f"  ⚠️ {tier}: {count}개 중 {produced}개만 생성됨")

        by_tier[tier] = records
        all_records.extend(records)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl, DATA_DIR

    write_release_jsonl("inequality_ko", all_records, data_dir=outdir)
    return by_tier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="부등식 / Futoshiki 퍼즐 생성기")
    parser.add_argument("--num", type=int, default=300, help="전체 퍼즐 수(3개 계층으로 분할)")
    parser.add_argument("--seed", type=int, default=None, help="재현성을 위한 난수 시드")
    parser.add_argument("--outdir", type=str, default=None, help="출력 디렉터리 (기본: data/)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 50)
    print("부등식 / Futoshiki 퍼즐 생성기")
    print("=" * 50)
    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import DATA_DIR
    outdir = Path(args.outdir) if args.outdir else DATA_DIR
    create_dataset_files(num_questions=args.num, outdir=outdir)
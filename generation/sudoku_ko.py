"""스도쿠 퍼즐 생성기 (KO).

logical-puzzles-me/sudoku/*의 단일 파일 포트:
- DIFFICULTY_CONFIGS (탐색 노드 게이팅 + spotcheck_k: easy=3, medium=5, hard=6)
- generate_complete (표준형 기본 격자 + 무작위 변환)
- _create_removal_groups (rot180 대칭 셀 쌍 제거)
- count_solutions / has_valid_solutions / find_all_solutions (MRV DFS, MAX_SOLUTIONS=1)
- solve_backtrack (SearchStats 반환: nodes, max_depth, avg_candidates)
- LogicSolver (L1: naked_single, hidden_single + L2: locked_candidates, naked_pair)
- rate() (DifficultyMeta 반환: label, tech_profile, max_tech_level, search_nodes)
- Spotcheck: select_spotcheck_positions + make_spotcheck_code + make_code (HMAC-SHA256)
  정답 형식: 스팟체크 위치의 숫자 K개를 공백으로 구분.
"""

import argparse
import hashlib
import hmac
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


Grid = List[List[int]]

MAX_SOLUTIONS = 1


# ============================================================================
# 격자 유틸리티
# ============================================================================

def from_string(s: str) -> Grid:
    s = s.strip()
    if len(s) != 81:
        raise ValueError(f"문자열 길이는 81이어야 합니다 (입력: {len(s)})")
    grid: Grid = []
    for i in range(9):
        row = []
        for j in range(9):
            ch = s[i * 9 + j]
            if ch in '.0':
                row.append(0)
            elif '1' <= ch <= '9':
                row.append(int(ch))
            else:
                raise ValueError(f"잘못된 문자 '{ch}' (위치 {i*9+j})")
        grid.append(row)
    return grid


def to_string(g: Grid, blanks: str = '.') -> str:
    chars = []
    for row in g:
        for val in row:
            chars.append(blanks if val == 0 else str(val))
    return ''.join(chars)


def copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def get_cell_candidates(g: Grid, r: int, c: int) -> Set[int]:
    if g[r][c] != 0:
        return set()
    used = set()
    for val in g[r]:
        if val != 0:
            used.add(val)
    for row in g:
        if row[c] != 0:
            used.add(row[c])
    box_r, box_c = (r // 3) * 3, (c // 3) * 3
    for br in range(box_r, box_r + 3):
        for bc in range(box_c, box_c + 3):
            if g[br][bc] != 0:
                used.add(g[br][bc])
    return set(range(1, 10)) - used


def is_solved(g: Grid) -> bool:
    for row in g:
        if 0 in row:
            return False
    for row in g:
        if set(row) != set(range(1, 10)):
            return False
    for c in range(9):
        if set(g[r][c] for r in range(9)) != set(range(1, 10)):
            return False
    for box_idx in range(9):
        box_r = (box_idx // 3) * 3
        box_c = (box_idx % 3) * 3
        vals = []
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                vals.append(g[r][c])
        if set(vals) != set(range(1, 10)):
            return False
    return True


def count_givens(puzzle: Grid) -> int:
    return sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)


# ============================================================================
# 격자 변환
# ============================================================================

def rotate_90_cw(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[c][8 - r] = g[r][c]
    return result


def rotate_180(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[8 - r][8 - c] = g[r][c]
    return result


def rotate_270_cw(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[8 - c][r] = g[r][c]
    return result


def mirror_horizontal(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[r][8 - c] = g[r][c]
    return result


def mirror_vertical(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[8 - r][c] = g[r][c]
    return result


def transpose(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[c][r] = g[r][c]
    return result


def anti_transpose(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[8 - c][8 - r] = g[r][c]
    return result


SYMMETRY_OPS = {
    'none': lambda g: copy_grid(g),
    'rot90': rotate_90_cw,
    'rot180': rotate_180,
    'rot270': rotate_270_cw,
    'mirror_h': mirror_horizontal,
    'mirror_v': mirror_vertical,
    'transpose': transpose,
    'anti_transpose': anti_transpose,
}


def apply_symmetry(g: Grid, symmetry: str) -> Grid:
    if symmetry not in SYMMETRY_OPS:
        raise ValueError(f"알 수 없는 대칭 변환: {symmetry}")
    return SYMMETRY_OPS[symmetry](g)


def relabel_digits(g: Grid, perm: dict) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            val = g[r][c]
            result[r][c] = 0 if val == 0 else perm.get(val, val)
    return result


def random_digit_permutation(rng: random.Random) -> dict:
    digits = list(range(1, 10))
    rng.shuffle(digits)
    return {i + 1: digits[i] for i in range(9)}


def shuffle_rows_in_band(g: Grid, band: int, rng: random.Random) -> Grid:
    result = copy_grid(g)
    rows_idx = [band * 3, band * 3 + 1, band * 3 + 2]
    rng.shuffle(rows_idx)
    for i, src_r in enumerate(rows_idx):
        result[band * 3 + i] = g[src_r]
    return result


def shuffle_cols_in_stack(g: Grid, stack: int, rng: random.Random) -> Grid:
    result = copy_grid(g)
    cols_idx = [stack * 3, stack * 3 + 1, stack * 3 + 2]
    rng.shuffle(cols_idx)
    for r in range(9):
        for i, src_c in enumerate(cols_idx):
            result[r][stack * 3 + i] = g[r][src_c]
    return result


def shuffle_bands(g: Grid, rng: random.Random) -> Grid:
    band_order = [0, 1, 2]
    rng.shuffle(band_order)
    result = [[0] * 9 for _ in range(9)]
    for i, band in enumerate(band_order):
        for offset in range(3):
            result[i * 3 + offset] = g[band * 3 + offset][:]
    return result


def shuffle_stacks(g: Grid, rng: random.Random) -> Grid:
    stack_order = [0, 1, 2]
    rng.shuffle(stack_order)
    result = copy_grid(g)
    for r in range(9):
        new_row = [0] * 9
        for i, stack in enumerate(stack_order):
            for offset in range(3):
                new_row[i * 3 + offset] = g[r][stack * 3 + offset]
        result[r] = new_row
    return result


def apply_random_transforms(g: Grid, rng: random.Random) -> Grid:
    result = copy_grid(g)
    perm = random_digit_permutation(rng)
    result = relabel_digits(result, perm)
    result = shuffle_bands(result, rng)
    result = shuffle_stacks(result, rng)
    for band in range(3):
        result = shuffle_rows_in_band(result, band, rng)
    for stack in range(3):
        result = shuffle_cols_in_stack(result, stack, rng)
    if rng.random() < 0.5:
        sym = rng.choice(['rot90', 'rot180', 'rot270', 'mirror_h',
                          'mirror_v', 'transpose', 'anti_transpose'])
        result = apply_symmetry(result, sym)
    return result


# ============================================================================
# 완성 격자 생성
# ============================================================================

def _base_solution() -> Grid:
    grid: Grid = []
    for r in range(9):
        row = []
        for c in range(9):
            val = (r * 3 + r // 3 + c) % 9 + 1
            row.append(val)
        grid.append(row)
    return grid


def generate_complete(seed: Optional[int] = None) -> Grid:
    rng = random.Random(seed)
    base = _base_solution()
    result = apply_random_transforms(base, rng)
    assert is_solved(result), "생성된 격자가 유효하지 않음"
    return result


# ============================================================================
# 풀이 개수 세기 (MRV DFS)
# ============================================================================

def count_solutions(puzzle: Grid, limit: int = 2) -> int:
    """퍼즐의 해 개수를 센다 (limit 도달 시 조기 종료).

    제약은 row/col/box 별 비트마스크로 점진적으로 유지: O(81) 셀 후보 계산을
    O(1) 로 압축. 이전 구현은 매 재귀 단계마다 27 셀(행+열+박스)을 다시
    스캔했음.
    """
    grid = copy_grid(puzzle)
    # bit i (1..9) = 해당 unit 에 숫자 i 가 이미 있음
    row_mask = [0] * 9
    col_mask = [0] * 9
    box_mask = [0] * 9
    full_mask = 0b1111111110  # bits 1..9 set
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v != 0:
                bit = 1 << v
                b = (r // 3) * 3 + (c // 3)
                row_mask[r] |= bit
                col_mask[c] |= bit
                box_mask[b] |= bit

    count = [0]

    def solve() -> bool:
        if count[0] >= limit:
            return True

        # MRV: 후보가 가장 적은 빈칸 선택
        min_candidates = 10
        best_cell = None  # (r, c, candidate_bits)
        for r in range(9):
            row_used = row_mask[r]
            for c in range(9):
                if grid[r][c] != 0:
                    continue
                b = (r // 3) * 3 + (c // 3)
                avail = full_mask & ~(row_used | col_mask[c] | box_mask[b])
                pop = bin(avail).count('1')
                if pop == 0:
                    return False  # 모순
                if pop < min_candidates:
                    min_candidates = pop
                    best_cell = (r, c, avail)
                    if pop == 1:
                        break  # naked single — 더 줄어들 수 없음
            if best_cell is not None and min_candidates == 1:
                break

        if best_cell is None:
            # 모두 채워짐 = 해 발견
            count[0] += 1
            return count[0] >= limit

        r, c, avail = best_cell
        b = (r // 3) * 3 + (c // 3)
        bits = avail
        while bits:
            low = bits & -bits  # 최하위 비트
            num = low.bit_length() - 1
            grid[r][c] = num
            row_mask[r] |= low
            col_mask[c] |= low
            box_mask[b] |= low
            if solve():
                return True
            grid[r][c] = 0
            row_mask[r] ^= low
            col_mask[c] ^= low
            box_mask[b] ^= low
            bits &= bits - 1

        return False

    solve()
    return count[0]


def has_valid_solutions(puzzle: Grid) -> bool:
    return count_solutions(puzzle, limit=MAX_SOLUTIONS + 1) == 1


def has_unique_solution(puzzle: Grid) -> bool:
    return count_solutions(puzzle, limit=2) == 1


def find_all_solutions(puzzle: Grid, limit: int = 2) -> List[Grid]:
    solutions: List[Grid] = []

    def solve(g: Grid) -> bool:
        if len(solutions) >= limit:
            return True
        min_candidates = 10
        best_cell = None
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    cands = get_cell_candidates(g, r, c)
                    if len(cands) == 0:
                        return False
                    if len(cands) < min_candidates:
                        min_candidates = len(cands)
                        best_cell = (r, c, cands)
        if best_cell is None:
            solutions.append(copy_grid(g))
            return len(solutions) >= limit
        r, c, cands = best_cell
        for num in cands:
            g[r][c] = num
            if solve(g):
                return True
            g[r][c] = 0
        return False

    grid = copy_grid(puzzle)
    solve(grid)
    return solutions


# ============================================================================
# 백트래킹 솔버 (통계 포함)
# ============================================================================

@dataclass
class SearchStats:
    nodes: int
    max_depth: int
    avg_candidates: float


def solve_backtrack(puzzle: Grid) -> Tuple[Optional[Grid], SearchStats]:
    grid = copy_grid(puzzle)
    stats = {'nodes': 0, 'max_depth': 0, 'total_candidates': 0, 'candidate_count': 0}

    def backtrack(depth: int) -> bool:
        stats['nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        min_cands = 10
        best_cell = None
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    cands = get_cell_candidates(grid, r, c)
                    if len(cands) == 0:
                        return False
                    if len(cands) < min_cands:
                        min_cands = len(cands)
                        best_cell = (r, c, cands)
        if best_cell is None:
            return is_solved(grid)
        r, c, cands = best_cell
        stats['total_candidates'] += len(cands)
        stats['candidate_count'] += 1
        for num in cands:
            grid[r][c] = num
            if backtrack(depth + 1):
                return True
            grid[r][c] = 0
        return False

    success = backtrack(0)
    avg_cands = (stats['total_candidates'] / stats['candidate_count']
                 if stats['candidate_count'] > 0 else 0.0)
    return (
        grid if success else None,
        SearchStats(
            nodes=stats['nodes'],
            max_depth=stats['max_depth'],
            avg_candidates=avg_cands,
        ),
    )


def search_stats(puzzle: Grid) -> SearchStats:
    _, stats = solve_backtrack(puzzle)
    return stats


# ============================================================================
# 논리 솔버 (L1 + L2)
# ============================================================================

@dataclass
class SolveSummary:
    solved: bool
    steps: int
    max_tech_level: str
    counts: dict
    guess_used: bool


class LogicSolver:
    def __init__(self, grid: Grid):
        self.grid = copy_grid(grid)
        self.candidates: List[List[Set[int]]] = []
        self._init_candidates()
        self.counts: dict = {}
        self.steps = 0
        self.max_tech_level = 'L0'

    def _init_candidates(self):
        self.candidates = []
        for r in range(9):
            row = []
            for c in range(9):
                if self.grid[r][c] == 0:
                    row.append(get_cell_candidates(self.grid, r, c))
                else:
                    row.append(set())
            self.candidates.append(row)

    def _update_level(self, level: str):
        levels = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']
        if levels.index(level) > levels.index(self.max_tech_level):
            self.max_tech_level = level

    def _record_technique(self, name: str, level: str):
        self.counts[name] = self.counts.get(name, 0) + 1
        self._update_level(level)
        self.steps += 1

    def _set_value(self, r: int, c: int, val: int):
        self.grid[r][c] = val
        self.candidates[r][c] = set()
        for i in range(9):
            self.candidates[r][i].discard(val)
            self.candidates[i][c].discard(val)
        box_r, box_c = (r // 3) * 3, (c // 3) * 3
        for br in range(box_r, box_r + 3):
            for bc in range(box_c, box_c + 3):
                self.candidates[br][bc].discard(val)

    def naked_single(self) -> bool:
        found = False
        for r in range(9):
            for c in range(9):
                if self.grid[r][c] == 0 and len(self.candidates[r][c]) == 1:
                    val = next(iter(self.candidates[r][c]))
                    self._set_value(r, c, val)
                    self._record_technique('naked_single', 'L1')
                    found = True
        return found

    def hidden_single(self) -> bool:
        to_set: List[Tuple[int, int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for row in range(9):
            for num in range(1, 10):
                positions = [(row, c) for c in range(9)
                             if num in self.candidates[row][c]]
                if len(positions) == 1:
                    pos = positions[0]
                    if pos not in seen:
                        to_set.append((pos[0], pos[1], num))
                        seen.add(pos)
        for col in range(9):
            for num in range(1, 10):
                positions = [(r, col) for r in range(9)
                             if num in self.candidates[r][col]]
                if len(positions) == 1:
                    pos = positions[0]
                    if pos not in seen:
                        to_set.append((pos[0], pos[1], num))
                        seen.add(pos)
        for box_idx in range(9):
            box_r = (box_idx // 3) * 3
            box_c = (box_idx % 3) * 3
            for num in range(1, 10):
                positions = []
                for br in range(box_r, box_r + 3):
                    for bc in range(box_c, box_c + 3):
                        if num in self.candidates[br][bc]:
                            positions.append((br, bc))
                if len(positions) == 1:
                    pos = positions[0]
                    if pos not in seen:
                        to_set.append((pos[0], pos[1], num))
                        seen.add(pos)
        for r, c, val in to_set:
            if self.grid[r][c] == 0 and val in self.candidates[r][c]:
                self._set_value(r, c, val)
                self._record_technique('hidden_single', 'L1')
        return len(to_set) > 0

    def locked_candidates(self) -> bool:
        found = False
        for box_idx in range(9):
            box_r = (box_idx // 3) * 3
            box_c = (box_idx % 3) * 3
            for num in range(1, 10):
                positions = []
                for r in range(box_r, box_r + 3):
                    for c in range(box_c, box_c + 3):
                        if num in self.candidates[r][c]:
                            positions.append((r, c))
                if not positions:
                    continue
                if len(set(r for r, c in positions)) == 1:
                    row = positions[0][0]
                    for c in range(9):
                        if c < box_c or c >= box_c + 3:
                            if num in self.candidates[row][c]:
                                self.candidates[row][c].discard(num)
                                found = True
                if len(set(c for r, c in positions)) == 1:
                    col = positions[0][1]
                    for r in range(9):
                        if r < box_r or r >= box_r + 3:
                            if num in self.candidates[r][col]:
                                self.candidates[r][col].discard(num)
                                found = True
        if found:
            self._record_technique('locked', 'L2')
        return found

    def naked_pair(self) -> bool:
        found = False
        for r in range(9):
            for c1 in range(9):
                cands1 = self.candidates[r][c1]
                if len(cands1) != 2:
                    continue
                for c2 in range(c1 + 1, 9):
                    if self.candidates[r][c2] == cands1:
                        for c in range(9):
                            if c != c1 and c != c2:
                                before = len(self.candidates[r][c])
                                self.candidates[r][c] -= cands1
                                if len(self.candidates[r][c]) < before:
                                    found = True
        for c in range(9):
            for r1 in range(9):
                cands1 = self.candidates[r1][c]
                if len(cands1) != 2:
                    continue
                for r2 in range(r1 + 1, 9):
                    if self.candidates[r2][c] == cands1:
                        for r in range(9):
                            if r != r1 and r != r2:
                                before = len(self.candidates[r][c])
                                self.candidates[r][c] -= cands1
                                if len(self.candidates[r][c]) < before:
                                    found = True
        for box_idx in range(9):
            box_r = (box_idx // 3) * 3
            box_c = (box_idx % 3) * 3
            box_cells = [(r, c) for r in range(box_r, box_r + 3)
                         for c in range(box_c, box_c + 3)]
            for i, (r1, c1) in enumerate(box_cells):
                cands1 = self.candidates[r1][c1]
                if len(cands1) != 2:
                    continue
                for r2, c2 in box_cells[i + 1:]:
                    if self.candidates[r2][c2] == cands1:
                        for r, c in box_cells:
                            if (r, c) != (r1, c1) and (r, c) != (r2, c2):
                                before = len(self.candidates[r][c])
                                self.candidates[r][c] -= cands1
                                if len(self.candidates[r][c]) < before:
                                    found = True
        if found:
            self._record_technique('naked_pair', 'L2')
        return found

    def solve_l1(self) -> bool:
        progress = True
        while progress:
            progress = False
            progress |= self.naked_single()
            progress |= self.hidden_single()
        return is_solved(self.grid)

    def solve_l2(self) -> bool:
        progress = True
        while progress:
            progress = False
            progress |= self.naked_single()
            progress |= self.hidden_single()
            progress |= self.locked_candidates()
            progress |= self.naked_pair()
        return is_solved(self.grid)

    def solve_with_limit(self, max_level: str) -> SolveSummary:
        if max_level == 'L1':
            solved = self.solve_l1()
        else:
            solved = self.solve_l2()
        return SolveSummary(
            solved=solved,
            steps=self.steps,
            max_tech_level=self.max_tech_level,
            counts=self.counts.copy(),
            guess_used=False,
        )


def solve_with_limit(puzzle: Grid, max_level: str) -> SolveSummary:
    solver = LogicSolver(puzzle)
    return solver.solve_with_limit(max_level)


# ============================================================================
# 난이도 평가
# ============================================================================

@dataclass
class DifficultyMeta:
    label: str
    tech_profile: dict
    max_tech_level: str
    search_nodes: int
    steps: int


def rate(puzzle: Grid) -> DifficultyMeta:
    summary_l1 = solve_with_limit(puzzle, 'L1')
    if summary_l1.solved:
        return DifficultyMeta(
            label='trivial',
            tech_profile=summary_l1.counts,
            max_tech_level=summary_l1.max_tech_level,
            search_nodes=0,
            steps=summary_l1.steps,
        )
    summary_l2 = solve_with_limit(puzzle, 'L2')
    if summary_l2.solved:
        stats = search_stats(puzzle)
        if stats.nodes <= 500:
            label = 'easy'
        elif stats.nodes <= 3000:
            label = 'medium'
        else:
            label = 'hard'
        return DifficultyMeta(
            label=label,
            tech_profile=summary_l2.counts,
            max_tech_level=summary_l2.max_tech_level,
            search_nodes=stats.nodes,
            steps=summary_l2.steps,
        )
    stats = search_stats(puzzle)
    tech_level = 'L3' if stats.nodes <= 3000 else 'L4'
    return DifficultyMeta(
        label='hard',
        tech_profile={},
        max_tech_level=tech_level,
        search_nodes=stats.nodes,
        steps=0,
    )


# ============================================================================
# 난이도 기반 생성
# ============================================================================

@dataclass
class DifficultyConfig:
    min_givens: int
    max_givens: int
    target_givens: int
    forbid_advanced: bool = False
    forbid_trivial: bool = False
    min_search_nodes: int = 0
    max_search_nodes: Optional[int] = None
    spotcheck_k: int = 6
    symmetry: str = 'rot180'
    minimal: bool = False
    forbidden_rate_labels: Set[str] = field(default_factory=set)


# 난이도 설정.
#
# Gemini 3 Flash는 힌트가 많이 보이면 스도쿠에 매우 강하고,
# 고급 추론이 필요한 "minimal" 퍼즐에는 매우 약합니다.
# 이전 설정은 medium(97%)과 hard(7%) 사이에 90% 절벽을 만들었습니다.
# 새 설정은 `givens_count`만 난이도 축으로 쓰고 `minimal` 플래그를
# 전 구간에서 끕니다 — 더 완만한 난이도 곡선을 만듭니다.
DIFFICULTY_CONFIGS = {
    # 재보정: 이전 35-37 힌트도 여전히 LLM에게 어려움(모든 모델 ~0%).
    # 60 힌트는 빈칸 21개 — L1 naked-single 스캔만으로 풀 수 있어
    # frontier 모델이 목표 75%에 도달 가능. medium/hard는 힌트를 점진적으로 줄임.
    'easy': DifficultyConfig(
        # gemini-3-flash: 정답률은 40-42 힌트 근처에서 급격히 변함
        # (42 힌트 ≈ 90-95%, 40 힌트 ≈ 50%). 70%는 41 힌트.
        # k=4 유지(k3는 80%, k4는 70%). 최종: 41 힌트/k4 = 14/20(70%).
        # 목표 ~70%.
        min_givens=40,
        max_givens=43,
        target_givens=41,
        symmetry='rot180',
        minimal=False,
        forbid_trivial=False,
        max_search_nodes=400,
        spotcheck_k=4,
    ),
    'medium': DifficultyConfig(
        # 39 힌트/k5는 61%(100샘플) — 39-40 힌트는 ~61%로 평탄.
        # 37 힌트=47.5%, 38 힌트=52.5%(둘 다 n=40). ~50%를 위해 38 선택.
        # 목표 ~50%.
        min_givens=37,
        max_givens=39,
        target_givens=38,
        symmetry='rot180',
        minimal=False,
        forbid_trivial=False,
        max_search_nodes=800,
        spotcheck_k=5,
    ),
    'hard': DifficultyConfig(
        # 26 힌트/k6는 7%(30샘플), 목표 25%보다 낮음 → 힌트를 더 공개(쉽게).
        # 목표 ~25%.
        min_givens=31,
        max_givens=35,
        target_givens=33,
        symmetry='rot180',
        minimal=False,
        forbid_trivial=False,
        spotcheck_k=6,
    )
}


def _create_removal_groups(rng: random.Random, symmetry: str) -> List[List[Tuple[int, int]]]:
    positions = [(i, j) for i in range(9) for j in range(9)]
    if symmetry == 'rot180':
        pairs = []
        used = set()
        for i, j in positions:
            if (i, j) not in used:
                sym_i, sym_j = 8 - i, 8 - j
                if (i, j) != (sym_i, sym_j):
                    pairs.append([(i, j), (sym_i, sym_j)])
                    used.add((i, j))
                    used.add((sym_i, sym_j))
                else:
                    pairs.append([(i, j)])
                    used.add((i, j))
        removal_groups = pairs
    else:
        removal_groups = [[(i, j)] for i, j in positions]
    rng.shuffle(removal_groups)
    return removal_groups


def generate_difficulty_puzzle(
    difficulty: str,
    seed: Optional[int] = None,
) -> Tuple[str, List[str], dict]:
    """정확히 1개의 해를 갖는 스도쿠 퍼즐을 구성적으로 생성."""
    if difficulty not in DIFFICULTY_CONFIGS:
        raise ValueError(f"잘못된 난이도: {difficulty}")

    config = DIFFICULTY_CONFIGS[difficulty]
    rng = random.Random(seed)

    solution_seed = rng.randint(0, 1_000_000_000)
    solution = generate_complete(solution_seed)
    puzzle = [row[:] for row in solution]

    removal_groups = _create_removal_groups(rng, config.symmetry)

    for group in removal_groups:
        temp_puzzle = [row[:] for row in puzzle]
        for i, j in group:
            temp_puzzle[i][j] = 0
        if count_givens(temp_puzzle) < config.min_givens:
            continue
        solution_count = count_solutions(temp_puzzle, limit=MAX_SOLUTIONS + 1)
        if solution_count == 1:
            puzzle = temp_puzzle
        current_givens = count_givens(puzzle)
        if current_givens <= config.target_givens:
            if not config.minimal:
                break

    final_givens = count_givens(puzzle)
    if final_givens > config.max_givens:
        raise RuntimeError(
            f"단서 수 {final_givens}개가 max_givens={config.max_givens}를 초과 "
            f"(seed={seed}, difficulty={difficulty})"
        )

    final_count = count_solutions(puzzle, limit=2)
    if final_count != 1:
        raise RuntimeError(
            f"최종 퍼즐의 해가 {final_count}개 (1개 예상). "
            f"seed={seed}, difficulty={difficulty}"
        )

    solutions = find_all_solutions(puzzle, limit=2)
    if not solutions:
        raise RuntimeError(
            f"count=1인데 find_all_solutions가 비어있음. seed={seed}"
        )

    rating = rate(puzzle)
    is_advanced = rating.max_tech_level in {'L3', 'L4', 'L5'}
    is_trivial = (rating.max_tech_level == 'L1' and rating.search_nodes == 0)

    if config.forbid_advanced and is_advanced:
        raise RuntimeError(
            f"forbid_advanced인데 max_tech={rating.max_tech_level}. seed={seed}"
        )
    if config.forbid_trivial and is_trivial:
        raise RuntimeError(
            f"forbid_trivial인데 L1으로 자명 (nodes=0). seed={seed}"
        )
    if config.forbidden_rate_labels and rating.label in config.forbidden_rate_labels:
        raise RuntimeError(
            f"rate().label={rating.label}은 금지됨. seed={seed}"
        )
    if rating.search_nodes < config.min_search_nodes:
        raise RuntimeError(
            f"search_nodes={rating.search_nodes} < min={config.min_search_nodes}. seed={seed}"
        )
    if config.max_search_nodes is not None and rating.search_nodes > config.max_search_nodes:
        raise RuntimeError(
            f"search_nodes={rating.search_nodes} > max={config.max_search_nodes}. seed={seed}"
        )

    puzzle_str = to_string(puzzle)
    solution_strs = [to_string(sol) for sol in solutions]

    metadata = {
        'difficulty': difficulty,
        'givens_count': count_givens(puzzle),
        'solution_count': final_count,
        'symmetry': config.symmetry,
        'seed': seed,
        'step_metrics': {
            'solver_steps': rating.steps,
            'search_nodes': rating.search_nodes,
            'max_tech_level': rating.max_tech_level,
            'tech_profile': rating.tech_profile,
            'rate_label': rating.label,
        },
    }
    return puzzle_str, solution_strs, metadata


# ============================================================================
# 스팟체크 (HMAC-SHA256 결정적 K-셀 선택)
# ============================================================================

def select_spotcheck_positions(canonical_hash: str, secret_hex: str, k: int) -> List[str]:
    """HMAC 기반 결정적 K개 위치 선택. 'r{R}c{C}' (1-기반) 형식 반환."""
    secret = bytes.fromhex(secret_hex)
    message = canonical_hash.encode('utf-8')
    mac = hmac.new(secret, message, hashlib.sha256).digest()
    seed = int.from_bytes(mac[:8], 'big')
    rng = random.Random(seed)
    all_positions = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(all_positions)
    selected = all_positions[:k]
    return [f"r{r+1}c{c+1}" for r, c in selected]


def _parse_position(pos: str) -> Tuple[int, int]:
    parts = pos[1:].split('c')
    return int(parts[0]) - 1, int(parts[1]) - 1


def make_spotcheck_code(solution: Grid, positions: List[str]) -> int:
    """레거시 스팟체크 코드: K개 위치 값의 합계 (디버깅용)."""
    total = 0
    for pos in positions:
        r, c = _parse_position(pos)
        total += solution[r][c]
    return total


def make_code(solution: Grid, positions: List[str]) -> int:
    """make_spotcheck_code의 별칭 (Repo A 네이밍)."""
    return make_spotcheck_code(solution, positions)


def make_spotcheck_answer(solution: Grid, positions: List[str]) -> str:
    """Repo A 정답 형식: 스팟체크 위치의 숫자 K개를 공백으로 구분한 문자열."""
    digits = []
    for pos in positions:
        r, c = _parse_position(pos)
        digits.append(str(solution[r][c]))
    return ' '.join(digits)


# ============================================================================
# 문제 텍스트 구성
# ============================================================================

def _format_grid(puzzle_str: str) -> str:
    cleaned = puzzle_str.replace('0', '.')
    rows = [cleaned[i:i + 9] for i in range(0, 81, 9)]
    return "\n".join(" ".join(row) for row in rows)


def _positions_as_rc_pairs(positions: List[str]) -> str:
    pairs = []
    for pos in positions:
        r, c = _parse_position(pos)
        pairs.append(f"({r+1}, {c+1})")
    return ", ".join(pairs)


SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_sudoku_solution_ko(
    puzzle_str: str,
    solution_str: str,
    positions: List[str],
    answer_str: str,
    difficulty: str,
    givens_count: int,
) -> str:
    """SFT teacher trace: 스도쿠 · 완전한 해와 스팟체크 SEG."""
    puzzle_grid = _format_grid(puzzle_str)
    solution_grid_text = _format_grid(solution_str)
    solution_grid = from_string(solution_str)

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {difficulty}",
        f"  - 주어진 힌트 셀 수: {givens_count}",
        f"  - 스팟체크 좌표 수: {len(positions)}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 규칙: 각 행·열·3×3 박스에 1–9가 정확히 한 번.",
        "  - 퍼즐 격자:",
    ]
    for row in puzzle_grid.splitlines():
        lines.append(f"    {row}")

    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: 행/열/박스 제약으로 단일 후보 확정 반복 → 유일 해 · "
        f"스팟체크 SEG {len(positions)}개"
    )
    lines.append("  · 완성 격자:")
    for row in solution_grid_text.splitlines():
        lines.append(f"    {row}")
    for i, pos in enumerate(positions, 1):
        r, c = _parse_position(pos)
        val = solution_grid[r][c]
        lines.append(
            f"    [SEG {i}] 위치 (r{r + 1}, c{c + 1}) = {val} "
            f"(행/열/박스 제약 모두 만족)"
        )

    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답(스팟체크 {len(positions)}자리): {answer_str}",
        "  - 완성 격자의 모든 행·열·3×3 박스가 1–9의 순열인지 확인.",
        "  - 각 힌트 셀 값이 완성 격자와 일치하는지 확인.",
    ])
    return "\n".join(lines)


def create_question(puzzle_str: str, positions: List[str]) -> str:
    grid = _format_grid(puzzle_str)
    rc_str = _positions_as_rc_pairs(positions)
    return (
        "당신은 스도쿠 풀이 전문가입니다. 다음 스도쿠 퍼즐을 완전히 풀어주세요.\n\n"
        "스도쿠 규칙:\n"
        "- 9×9 격자를 채우세요. 각 행, 열, 3×3 박스에 1-9 숫자가 들어갑니다\n"
        "- 각 숫자는 각 행, 열, 박스에 정확히 한 번만 나타나야 합니다\n\n"
        "퍼즐은 아래 9줄 격자로 제공됩니다.\n"
        "'.'은 빈 셀을 나타냅니다.\n\n"
        f"퍼즐 격자:\n{grid}\n\n"
        "먼저 퍼즐을 단계별로 완전히 풀어주세요.\n"
        "그런 다음, 다음 (행, 열) 좌표 (1-기반)의 값을 같은 순서로 제시하세요:\n"
        f"{rc_str}\n\n"
        "중요: 풀이 후 반드시 마지막 줄에 다음 형식으로 작성하세요:\n"
        "Answer: [해당 좌표의 숫자들을 공백으로 구분]\n\n"
        "예시 (값이 5, 3, 4, 6, 7, 8인 경우):\n"
        "Answer: 5 3 4 6 7 8"
    )


# ============================================================================
# 데이터셋 생성
# ============================================================================

def _apply_full_transform(
    puzzle_grid: Grid,
    solution_grid: Grid,
    rng: random.Random,
) -> Tuple[Grid, Grid]:
    """퍼즐·해답 격자 모두에 무작위 스도쿠 대칭 변환 적용.

    두 격자에 *동일한* 치환을 적용해 일관성을 유지합니다.
    변환: 숫자 재라벨링 + 밴드 내 행 셔플 + 스택 내 열 셔플
    + 밴드 셔플 + 스택 셔플 + 기하 대칭.
    정확한 스도쿠 대칭이므로 유일성과 힌트 수가 보존됩니다.
    """
    perm = random_digit_permutation(rng)
    p = relabel_digits(puzzle_grid, perm)
    s = relabel_digits(solution_grid, perm)

    for band in range(3):
        seed_b = rng.randint(0, 2**31)
        p = shuffle_rows_in_band(p, band, random.Random(seed_b))
        s = shuffle_rows_in_band(s, band, random.Random(seed_b))

    for stack in range(3):
        seed_st = rng.randint(0, 2**31)
        p = shuffle_cols_in_stack(p, stack, random.Random(seed_st))
        s = shuffle_cols_in_stack(s, stack, random.Random(seed_st))

    seed_bd = rng.randint(0, 2**31)
    p = shuffle_bands(p, random.Random(seed_bd))
    s = shuffle_bands(s, random.Random(seed_bd))

    seed_sk = rng.randint(0, 2**31)
    p = shuffle_stacks(p, random.Random(seed_sk))
    s = shuffle_stacks(s, random.Random(seed_sk))

    sym = rng.choice(list(SYMMETRY_OPS.keys()))
    op = SYMMETRY_OPS[sym]
    p = op(p)
    s = op(s)
    return p, s


def _make_puzzle_record(
    puzzle_grid: Grid,
    solution_grid: Grid,
    difficulty: str,
    idx: int,
    secret_hex: str,
) -> Dict:
    puzzle_str = to_string(puzzle_grid)
    solution_str = to_string(solution_grid)
    k = DIFFICULTY_CONFIGS[difficulty].spotcheck_k
    canonical_hash = f"sha256:{hashlib.sha256(puzzle_str.encode()).hexdigest()}"
    positions = select_spotcheck_positions(canonical_hash, secret_hex, k)
    answer_str = make_spotcheck_answer(solution_grid, positions)
    question = create_question(puzzle_str, positions)
    return {
        'id': f'sudoku_ko_{difficulty}_{idx:04d}',
        'question': question,
        'answer': answer_str,
        'solution': _build_sudoku_solution_ko(
            puzzle_str=puzzle_str,
            solution_str=solution_str,
            positions=positions,
            answer_str=answer_str,
            difficulty=difficulty,
            givens_count=count_givens(puzzle_grid),
        ),
        'difficulty': difficulty,
    }


def create_dataset_files(num_questions: int, base_per_diff: int = 5):
    """스도쿠 데이터셋 파일(CSV + JSONL) 생성.

    빠른 전략: 난이도별 BASE_PER_DIFF개 기본 퍼즐을 전체 솔버로 생성한 뒤,
    나머지는 스도쿠 대칭 변환(숫자 재라벨링 + 행/열/밴드 셔플 + 반사/회전)으로
    파생합니다. 이 변환은 정확한 대칭이므로 유일성과 힌트 수가 보장되어
    추가 solve 호출이 필요 없습니다.
    """
    import pandas as pd

    BASE_PER_DIFF = base_per_diff  # 난이도별 느린 생성 기본 퍼즐 수

    print(f"스도쿠 퍼즐 {num_questions}개 생성 중 (transform-fast mode)...")

    difficulties = ['easy', 'medium', 'hard']
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles: List[Dict] = []
    secret_hex = '0' * 64
    max_retries = 140

    for di, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if di < remainder else 0)
        if count == 0:
            continue

        print(f"\n=== 난이도 '{difficulty}' 퍼즐 생성 중 ({count}개 필요) ===")

        # --- 1단계: 느린 솔버로 기본 퍼즐 생성 ---
        bases_needed = min(BASE_PER_DIFF, count)
        base_pairs: List[Tuple[Grid, Grid, dict]] = []
        for b in range(bases_needed):
            last_err = None
            for retry in range(max_retries):
                seed = 7919 * di + 1031 * b + retry
                try:
                    p_str, sol_strs, meta = generate_difficulty_puzzle(difficulty, seed)
                    base_pairs.append((from_string(p_str), from_string(sol_strs[0]), meta))
                    print(f"  [base {b+1}/{bases_needed}] givens={meta['givens_count']}")
                    break
                except RuntimeError as e:
                    last_err = e
            else:
                print(f"  [base {b+1}/{bases_needed}] {max_retries}회 재시도 후 실패: {last_err}")

        if not base_pairs:
            print(f"  {difficulty} base 퍼즐 없음; 건너뜀")
            continue

        # --- 2단계: 대칭 변환으로 나머지 퍼즐 파생 ---
        generated = 0
        for j in range(count):
            base_pg, base_sg, base_meta = base_pairs[j % len(base_pairs)]
            if j < len(base_pairs):
                pg, sg = base_pg, base_sg
            else:
                rng = random.Random(999983 * di + 1000003 * j)
                pg, sg = _apply_full_transform(base_pg, base_sg, rng)
            try:
                record = _make_puzzle_record(pg, sg, difficulty, generated, secret_hex)
                all_puzzles.append(record)
                generated += 1
                print(
                    f"  [{j+1}/{count}] givens={count_givens(pg)}, "
                    f"answer='{record['answer']}'"
                )
            except Exception as e:
                print(f"  [{j+1}/{count}] 후처리 실패: {e}")

        if generated < count:
            print(f"  경고: {difficulty} 퍼즐 {generated}/{count}개만 생성")

    print(f"\n총 {len(all_puzzles)}개 퍼즐 생성 완료")

    df = pd.DataFrame(all_puzzles)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("sudoku_ko", all_puzzles)

    return df, all_puzzles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="스도쿠 퍼즐 생성기 (한국어)")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")
    parser.add_argument("--workers", type=int, default=0, help="호환성용; fast mode는 템플릿 뱅크 사용")
    parser.add_argument("--base-per-diff", type=int, default=5,
                        help="대칭 변환 전 난이도별 slow-generated base 퍼즐 수")
    args = parser.parse_args()

    print("=" * 60)
    print("스도쿠 퍼즐 생성기 (한국어)")
    print("=" * 60)

    create_dataset_files(num_questions=args.num, base_per_diff=args.base_per_diff)

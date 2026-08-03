"""Sudoku Puzzle Generator (EN).

Monolithic port of logical-puzzles-me/sudoku/*:
- DIFFICULTY_CONFIGS with search-node gating + spotcheck_k (easy=3, medium=5, hard=6)
- generate_complete (canonical-form base + random transforms)
- _create_removal_groups (rot180-symmetric cell-pair removal)
- count_solutions / has_valid_solutions / find_all_solutions (MRV DFS, MAX_SOLUTIONS=1)
- solve_backtrack returning SearchStats (nodes, max_depth, avg_candidates)
- LogicSolver with L1 (naked_single, hidden_single) + L2 (locked_candidates, naked_pair)
- rate() returning DifficultyMeta (label, tech_profile, max_tech_level, search_nodes)
- Spotcheck: select_spotcheck_positions + make_spotcheck_code + make_code (HMAC-SHA256)
  Answer format: K space-separated digits at spotcheck positions.
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
# Grid utilities
# ============================================================================

def from_string(s: str) -> Grid:
    s = s.strip()
    if len(s) != 81:
        raise ValueError(f"String length must be 81 (got: {len(s)})")
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
                raise ValueError(f"Invalid character '{ch}' at position {i*9+j}")
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
# Grid transformations
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
        raise ValueError(f"Unknown symmetry: {symmetry}")
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
# Complete grid generation
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
    assert is_solved(result), "Generated grid is not valid"
    return result


# ============================================================================
# Solution counter (MRV DFS)
# ============================================================================

def count_solutions(puzzle: Grid, limit: int = 2) -> int:
    """Count puzzle solutions (early-stop at `limit`).

    Uses row/col/box bitmasks maintained incrementally: O(81) per-cell candidate
    computation is compressed to O(1). The previous implementation re-scanned 27
    cells (row+col+box) per recursive step.
    """
    grid = copy_grid(puzzle)
    # bit i (1..9) = digit i is already present in the corresponding unit
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

        # MRV: pick the empty cell with fewest candidates
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
                    return False  # contradiction
                if pop < min_candidates:
                    min_candidates = pop
                    best_cell = (r, c, avail)
                    if pop == 1:
                        break  # naked single — can't shrink further
            if best_cell is not None and min_candidates == 1:
                break

        if best_cell is None:
            # All cells filled = solution found
            count[0] += 1
            return count[0] >= limit

        r, c, avail = best_cell
        b = (r // 3) * 3 + (c // 3)
        bits = avail
        while bits:
            low = bits & -bits  # lowest set bit
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
# Backtrack solver with statistics
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
# Logic solver (L1 + L2)
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
# Difficulty rating
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
# Difficulty-based generation
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


# Difficulty configurations.
#
# Gemini 3 Flash is very strong at Sudoku when many givens are visible, and
# very weak on "minimal" puzzles that require advanced deduction techniques.
# The old configs created a 90% cliff between medium (97%) and hard (7%).
# New configs rely purely on `givens_count` as the difficulty knob and
# disable the `minimal` flag everywhere — this yields a smoother progression.
DIFFICULTY_CONFIGS = {
    # recalibration: previous 35-37 givens was still LLM-hard (all models ~0%).
    # 60 givens means 21 blanks — L1 naked-single scan alone should solve, enabling
    # frontier models to hit target 75%. medium / hard progressively reduce givens.
    'easy': DifficultyConfig(
        # gemini-3-flash: solve-accuracy has a steep cliff around 40-42
        # givens (42 givens ≈ 90-95%, 40 givens ≈ 50%). 70% sits at 41 givens.
        # k stays at 4 (k3 at 41 givens measured 80%, k4 measured 70%). Final:
        # 41 givens / k4 = 14/20 (70%) on the calibration batch. Target ~70%.
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
        # 39 givens/k5 was 61% (100-sample) — givens 39-40 are flat ~61%.
        # Pushed lower: 37 givens = 47.5%, 38 givens = 52.5% (both n=40). 38 chosen
        # for ~50%. Target ~50%.
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
        # 26 givens/k6 measured 7% (30-sample), below the 25% target → MORE
        # givens (easier). Target ~25%.
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
    """Constructively generate a Sudoku puzzle with exactly 1 solution."""
    if difficulty not in DIFFICULTY_CONFIGS:
        raise ValueError(f"Invalid difficulty: {difficulty}")

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
            f"Puzzle has {final_givens} givens > max_givens={config.max_givens} "
            f"(seed={seed}, difficulty={difficulty})"
        )

    final_count = count_solutions(puzzle, limit=2)
    if final_count != 1:
        raise RuntimeError(
            f"Final puzzle has {final_count} solutions (expected 1). "
            f"seed={seed}, difficulty={difficulty}"
        )

    solutions = find_all_solutions(puzzle, limit=2)
    if not solutions:
        raise RuntimeError(
            f"find_all_solutions empty despite count=1. seed={seed}"
        )

    rating = rate(puzzle)
    is_advanced = rating.max_tech_level in {'L3', 'L4', 'L5'}
    is_trivial = (rating.max_tech_level == 'L1' and rating.search_nodes == 0)

    if config.forbid_advanced and is_advanced:
        raise RuntimeError(
            f"forbid_advanced but max_tech={rating.max_tech_level}. seed={seed}"
        )
    if config.forbid_trivial and is_trivial:
        raise RuntimeError(
            f"forbid_trivial but puzzle L1-trivial (nodes=0). seed={seed}"
        )
    if config.forbidden_rate_labels and rating.label in config.forbidden_rate_labels:
        raise RuntimeError(
            f"rate().label={rating.label} forbidden. seed={seed}"
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
# Spotcheck (HMAC-SHA256 deterministic K-cell selection)
# ============================================================================

def select_spotcheck_positions(canonical_hash: str, secret_hex: str, k: int) -> List[str]:
    """HMAC-based deterministic K-position selection. Returns 'r{R}c{C}' (1-based)."""
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
    """Legacy spotcheck code: sum of values at K positions (kept for debugging)."""
    total = 0
    for pos in positions:
        r, c = _parse_position(pos)
        total += solution[r][c]
    return total


def make_code(solution: Grid, positions: List[str]) -> int:
    """Alias of make_spotcheck_code (Repo A naming)."""
    return make_spotcheck_code(solution, positions)


def make_spotcheck_answer(solution: Grid, positions: List[str]) -> str:
    """Repo A answer format: K space-separated digits at spotcheck positions."""
    digits = []
    for pos in positions:
        r, c = _parse_position(pos)
        digits.append(str(solution[r][c]))
    return ' '.join(digits)


# ============================================================================
# Question construction
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


SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


def _build_sudoku_solution_en(
    puzzle_str: str,
    solution_str: str,
    positions: List[str],
    answer_str: str,
    difficulty: str,
    givens_count: int,
) -> str:
    """SFT teacher trace: sudoku with full grid + spot-check SEGs."""
    puzzle_grid = _format_grid(puzzle_str)
    solution_grid_text = _format_grid(solution_str)
    solution_grid = from_string(solution_str)

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {difficulty}",
        f"  - Given hints: {givens_count}",
        f"  - Spot-check positions: {len(positions)}",
        "  - Final answer is confirmed in [STEP 3]",
        "[STEP 1] Given",
        "  - Rule: each row, column, and 3x3 box contains 1-9 exactly once.",
        "  - Puzzle grid:",
    ]
    for row in puzzle_grid.splitlines():
        lines.append(f"    {row}")

    lines.append("[STEP 2] Worked solution")
    lines.append(
        f"  · Summary: iterative row/col/box single-candidate propagation -> unique grid · "
        f"{len(positions)} spot-check SEGs"
    )
    lines.append("  · Completed grid:")
    for row in solution_grid_text.splitlines():
        lines.append(f"    {row}")
    for i, pos in enumerate(positions, 1):
        r, c = _parse_position(pos)
        val = solution_grid[r][c]
        lines.append(
            f"    [SEG {i}] position (r{r + 1}, c{c + 1}) = {val} "
            f"(row/col/box constraints all hold)"
        )

    lines.extend([
        "[STEP 3] Answer and verification",
        f"  - Final answer ({len(positions)} spot-check digits): {answer_str}",
        "  - Every row, column, and 3x3 box of the completed grid is a permutation of 1-9.",
        "  - Every given hint cell matches the completed grid.",
    ])
    return "\n".join(lines)


def create_question(puzzle_str: str, positions: List[str]) -> str:
    grid = _format_grid(puzzle_str)
    rc_str = _positions_as_rc_pairs(positions)
    return (
        "You are an expert Sudoku solver. Solve the following Sudoku puzzle completely.\n\n"
        "Sudoku rules:\n"
        "- Fill the 9x9 grid so that each row, column, and 3x3 box contains digits 1-9\n"
        "- Each digit must appear exactly once in each row, column, and box\n\n"
        "The puzzle is given below as a 9-line grid.\n"
        "'.' represents an empty cell.\n\n"
        f"Puzzle grid:\n{grid}\n\n"
        "First, solve the entire puzzle step by step.\n"
        "Then, report the values at these (row, col) coordinates (1-based) in the same order:\n"
        f"{rc_str}\n\n"
        "CRITICAL: After solving, your very last line MUST be exactly:\n"
        "Answer: [digits at those coordinates, separated by spaces]\n\n"
        "Example (if the values are 5, 3, 4, 6, 7, 8):\n"
        "Answer: 5 3 4 6 7 8"
    )


# ============================================================================
# Dataset generation
# ============================================================================

def _apply_full_transform(
    puzzle_grid: Grid,
    solution_grid: Grid,
    rng: random.Random,
) -> Tuple[Grid, Grid]:
    """Apply random Sudoku symmetry transforms to both puzzle and solution grids.

    The *same* permutation is applied to both grids so they stay consistent.
    Transforms: digit relabelling + row-in-band shuffle + col-in-stack shuffle
    + band shuffle + stack shuffle + geometric symmetry.
    These are exact Sudoku symmetries: uniqueness and givens count are preserved.
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
        'id': f'sudoku_en_{difficulty}_{idx:04d}',
        'question': question,
        'answer': answer_str,
        'solution': _build_sudoku_solution_en(
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
    """Create sudoku dataset files (CSV + JSONL).

    Fast strategy: generate BASE_PER_DIFF base puzzles per difficulty via the
    full solver, then derive the remaining puzzles via Sudoku symmetry transforms
    (digit relabelling + row/col/band shuffles + reflections/rotations).
    These transforms are exact symmetries: uniqueness and givens-count are
    guaranteed to be preserved, so no additional solve calls are needed.
    """
    import pandas as pd

    BASE_PER_DIFF = base_per_diff  # number of slow-generated base puzzles per difficulty

    print(f"Generating {num_questions} sudoku puzzles (transform-fast mode)...")

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

        print(f"\n=== Generating {difficulty} puzzles ({count} needed) ===")

        # --- Phase 1: generate base puzzles via slow solver ---
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
                print(f"  [base {b+1}/{bases_needed}] failed after {max_retries} retries: {last_err}")

        if not base_pairs:
            print(f"  No base puzzles generated for {difficulty}; skipping")
            continue

        # --- Phase 2: derive remaining puzzles via symmetry transforms ---
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
                print(f"  [{j+1}/{count}] post-process failed: {e}")

        if generated < count:
            print(f"  Warning: only generated {generated}/{count} {difficulty} puzzles")

    print(f"\nGenerated {len(all_puzzles)} puzzles total")

    df = pd.DataFrame(all_puzzles)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("sudoku_en", all_puzzles)

    return df, all_puzzles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sudoku Puzzle Generator (EN)")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")
    parser.add_argument("--workers", type=int, default=0, help="Accepted for compatibility; fast mode uses template bank")
    parser.add_argument("--base-per-diff", type=int, default=5,
                        help="Distinct slow-generated base puzzles per difficulty before "
                             "symmetry-transform fill. Set equal to per-difficulty count "
                             "for full structural diversity (calibration).")
    args = parser.parse_args()

    print("=" * 60)
    print("Sudoku Puzzle Generator (EN)")
    print("=" * 60)

    create_dataset_files(num_questions=args.num, base_per_diff=args.base_per_diff)

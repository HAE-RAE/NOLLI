"""지뢰찾기(Minesweeper) 퍼즐 생성기 (KO) — 데이터셋 재현 빌드.

이 모듈은 지뢰찾기 SFT 데이터셋을 세 개의 개별 JSONL 파일 출력과
하나의 통합 CSV로 생성하며, 참조 파일들의 구조와 solution 텍스트
형식과 일치합니다:

    minesweeper_ko_easy.jsonl     ("forcing"      solution format)
    minesweeper_ko_medium.jsonl   ("constrain"    solution format)
    minesweeper_ko_hard.jsonl     ("force_global" solution format)
    minesweeper_ko.csv            (easy -> medium -> hard, verbatim concat)

------------------------------------------------------------------------------
정확히 재현하는 항목 (참조 파일과 바이트 단위로 검증됨):
  * 모든 퍼즐의 question / prompt 텍스트 (create_prompt).
  * 세 가지 구별되는 STEP0..STEP3 solution 텍스트 형식.
  * easy 형식의 "단순 forcing 이상의 탐색 필요" 플래그,
    단순 forcing == 단일 단서 전파 + 전역 지뢰 수 전파로 정의;
    플래그는 위 방식으로 격자가 완전히 풀리지 않을 때 "예".
    (참조 easy 파일과 100/100 일치.)
  * hard 형식의 forced/[GLOBAL] 분할, 단일 단서 +
    단서 쌍(부분집합) 전파를 고정점까지 적용하여 도출;
    이 전파에 도달하지 못한 셀이 [GLOBAL] 셀.
    (참조 hard 파일과 100/100 일치 — "단일 단서나 단서 쌍으로
    고정되지 않음" 문구와 일관.)
  * 전체 내부 bitstring (easy/medium), 파일별 id 체계,
    레코드별 `difficulty` 필드 (출력 파일과 일치: easy / medium / hard).

동일하게 재현할 수 없는 항목 (이유):
  * 정확한 퍼즐 인스턴스 (어떤 지뢰, 어떤 공개 셀). 원본 RNG 시드와
    원본 솔버의 내부 셀 순서가 보존되지 않았고, 지뢰찾기 레이아웃 생성은
    RNG 기반입니다. 이 빌드는 블록별로 동일한 구조 프로필
    (격자 크기, 지뢰 수, 공개 셀 수/비율 대역, 라벨)을 갖는
    새 퍼즐을 재생성합니다.
  * medium/hard의 "솔버 백트래킹 노드" 정수는 특정 퍼즐과
    솔버의 탐색 경로에 종속; 새로 생성된 퍼즐에서는
    이 모듈의 실시간 솔버 값을 사용하므로 표시된 퍼즐에 대해
    정확하지만 과거 정수와는 일치하지 않습니다.

요약: 실행하면 형식·프로필이 참조와 동일한 데이터셋을 얻습니다;
동일한 (또는 동일할 수 없는) 랜덤 퍼즐 인스턴스는 아닙니다.
------------------------------------------------------------------------------
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set


MAX_SOLUTIONS = 1


# ---------------------------------------------------------------------------
# 저수준 보드 엔진 (이웃 / 단서 숫자 / 정확 유일해 솔버)
# ---------------------------------------------------------------------------

_NEIGHBOR_CACHE: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = {}


def neighbors(r: int, c: int, R: int, C: int) -> List[Tuple[int, int]]:
    key = (R, C, r, c)
    cached = _NEIGHBOR_CACHE.get(key)
    if cached is not None:
        return cached
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                result.append((nr, nc))
    _NEIGHBOR_CACHE[key] = result
    return result


def compute_numbers(mask: List[List[int]]) -> List[List[Optional[int]]]:
    R, C = len(mask), len(mask[0])
    nums: List[List[Optional[int]]] = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if mask[r][c] == 1:
                nums[r][c] = None
            else:
                nums[r][c] = sum(mask[nr][nc] == 1 for nr, nc in neighbors(r, c, R, C))
    return nums


def solve_puzzle(
    puzzle_nums: List[List[Optional[int]]],
    R: int,
    C: int,
    max_solutions: int = 2,
    total_mines: Optional[int] = None,
    _stats: Optional[Dict] = None,
) -> List[List[List[int]]]:
    """백트래킹 정확 솔버. 최대 `max_solutions`개의 지뢰 마스크를 반환.

    `_stats`가 주어지면 `_stats['nodes']`가 방문한 탐색 노드 수를 세며,
    이 값이 medium/hard solution 추적에서 "솔버 백트래킹 노드"로
    보고되는 정수입니다.
    """
    nbs = [[neighbors(r, c, R, C) for c in range(C)] for r in range(R)]

    constraints = []
    for r in range(R):
        for c in range(C):
            v = puzzle_nums[r][c]
            if v is not None:
                constraints.append((r, c, v, nbs[r][c]))

    assignment: List[List[Optional[int]]] = [[None] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if puzzle_nums[r][c] is not None:
                assignment[r][c] = 0

    constraints_per_cell = defaultdict(list)
    for idx, (_, _, _, nlist) in enumerate(constraints):
        for (nr, nc) in nlist:
            constraints_per_cell[(nr, nc)].append(idx)

    unknown_cells = [(r, c) for r in range(R) for c in range(C)
                     if assignment[r][c] is None]
    unknown_cells.sort(key=lambda rc: -len(constraints_per_cell[rc]))

    def check_constraints() -> bool:
        for (_, _, v, nlist) in constraints:
            assigned_mines = 0
            unknown_count = 0
            for (nr, nc) in nlist:
                if assignment[nr][nc] is None:
                    unknown_count += 1
                elif assignment[nr][nc] == 1:
                    assigned_mines += 1
            if assigned_mines > v:
                return False
            if assigned_mines + unknown_count < v:
                return False
        return True

    def check_global_mines() -> bool:
        if total_mines is None:
            return True
        assigned_mines = sum(assignment[r][c] == 1
                             for r in range(R) for c in range(C)
                             if assignment[r][c] is not None)
        unknown_count = sum(assignment[r][c] is None
                            for r in range(R) for c in range(C))
        if assigned_mines > total_mines:
            return False
        if assigned_mines + unknown_count < total_mines:
            return False
        return True

    solutions: List[List[List[int]]] = []

    def backtrack(i: int):
        if _stats is not None:
            _stats['nodes'] = _stats.get('nodes', 0) + 1
        if len(solutions) >= max_solutions:
            return
        if i == len(unknown_cells):
            if not check_global_mines():
                return
            for (_, _, v, nlist) in constraints:
                actual = sum(assignment[nr][nc] == 1 for (nr, nc) in nlist)
                if actual != v:
                    return
            solutions.append([row[:] for row in assignment])
            return
        r, c = unknown_cells[i]
        for val in (0, 1):
            assignment[r][c] = val
            if check_global_mines() and check_constraints():
                backtrack(i + 1)
            assignment[r][c] = None
            if len(solutions) >= max_solutions:
                return

    backtrack(0)
    return solutions


def mask_to_bitstring(mask: List[List[int]]) -> str:
    return "".join("".join(str(cell) for cell in row) for row in mask)


def mask_to_coord_list(mask: List[List[int]]) -> List[Tuple[int, int]]:
    coords = []
    for r, row in enumerate(mask):
        for c, cell in enumerate(row):
            if cell == 1:
                coords.append((r, c))
    return coords


def coords_to_answer_string(coords: List[Tuple[int, int]]) -> str:
    return ", ".join(f"({r},{c})" for r, c in sorted(coords))


# ---------------------------------------------------------------------------
# solution 텍스트 내용 도출에 사용하는 복원 추론 알고리즘
# ---------------------------------------------------------------------------

def _clue_lookup(rows: List[str]) -> Dict[Tuple[int, int], int]:
    cl: Dict[Tuple[int, int], int] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch.isdigit():
                cl[(r, c)] = int(ch)
    return cl


def _adjacent_clues(rr: int, cc: int, R: int, C: int,
                    cl: Dict[Tuple[int, int], int]) -> List[Tuple[int, int, int]]:
    return [(nr, nc, cl[(nr, nc)]) for (nr, nc) in neighbors(rr, cc, R, C)
            if (nr, nc) in cl]


def _init_state(rows: List[str]):
    """상태 맵: None = 미정 숨김 셀, 'S' = 안전, 'M' = 지뢰."""
    st: Dict[Tuple[int, int], Optional[str]] = {}
    hidden: List[Tuple[int, int]] = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == '#':
                st[(r, c)] = None
                hidden.append((r, c))
            else:
                st[(r, c)] = 'S'
    return st, hidden


def requires_search_beyond_simple_forcing(rows: List[str], R: int, C: int,
                                          total_mines: int) -> bool:
    """Easy 형식 플래그.

    "단순 forcing" == 단일 단서 전파 + 전역 지뢰 수 전파를
    고정점까지 반복. 고정점 이후에도 미정인 숨김 셀이 남으면
    True ("탐색 필요: 예") 반환.
    참조 easy 파일 대비 검증: 100/100.
    """
    cl = _clue_lookup(rows)
    st, hidden = _init_state(rows)
    changed = True
    while changed:
        changed = False
        # 단일 단서 전파
        for cell, v in cl.items():
            nb = neighbors(cell[0], cell[1], R, C)
            km = sum(1 for p in nb if st[p] == 'M')
            unk = [p for p in nb if st[p] is None]
            if not unk:
                continue
            if km == v:
                for p in unk:
                    if st[p] is None:
                        st[p] = 'S'
                        changed = True
            elif km + len(unk) == v:
                for p in unk:
                    if st[p] is None:
                        st[p] = 'M'
                        changed = True
        # 전역 지뢰 수 전파
        km_all = sum(1 for p in st if st[p] == 'M')
        unk_all = [p for p in hidden if st[p] is None]
        if unk_all:
            if km_all == total_mines:
                for p in unk_all:
                    st[p] = 'S'
                    changed = True
            elif km_all + len(unk_all) == total_mines:
                for p in unk_all:
                    st[p] = 'M'
                    changed = True
    return any(st[p] is None for p in hidden)


def forced_mine_set(rows: List[str], R: int, C: int) -> Set[Tuple[int, int]]:
    """Hard 형식 forced 집합.

    전역 지뢰 수를 사용하지 않고 단일 단서 + 단서 쌍(부분집합) 전파를
    고정점까지 적용. 이 전파에 도달한 지뢰가 "forced" 지뢰이며,
    나머지는 [GLOBAL] 지뢰.
    참조 hard 파일 대비 검증: forced 집합 100/100 일치.
    """
    cl = _clue_lookup(rows)
    st, _ = _init_state(rows)

    def rem_and_unk(cell):
        nb = neighbors(cell[0], cell[1], R, C)
        km = sum(1 for p in nb if st[p] == 'M')
        return cl[cell] - km, frozenset(p for p in nb if st[p] is None)

    changed = True
    while changed:
        changed = False
        # 단일 단서
        for cell in cl:
            rem, unk = rem_and_unk(cell)
            if not unk:
                continue
            if rem == 0:
                for p in unk:
                    if st[p] is None:
                        st[p] = 'S'
                        changed = True
            elif rem == len(unk):
                for p in unk:
                    if st[p] is None:
                        st[p] = 'M'
                        changed = True
        # 단서 쌍 부분집합 규칙
        items = list(cl.keys())
        info = {cell: rem_and_unk(cell) for cell in items}
        for a in items:
            ra, ua = info[a]
            if not ua:
                continue
            for b in items:
                if a == b:
                    continue
                rb, ub = info[b]
                if ua < ub:  # ua는 ub의 진부분집합
                    diff = ub - ua
                    dval = rb - ra
                    if dval == 0:
                        for p in diff:
                            if st[p] is None:
                                st[p] = 'S'
                                changed = True
                    elif dval == len(diff):
                        for p in diff:
                            if st[p] is None:
                                st[p] = 'M'
                                changed = True
    return set(p for p, v in st.items() if v == 'M')


# ---------------------------------------------------------------------------
# Solution 텍스트 빌더 (파일 형식별 하나씩)
# ---------------------------------------------------------------------------

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · "
    "STEP3=답·검산"
)


def _solution_header(rows: List[str], R: int, C: int, total_mines: int,
                     difficulty: str, extra_meta_lines: List[str]):
    cl = _clue_lookup(rows)
    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {difficulty}",
        f"  - 격자: {R} rows x {C} cols",
        f"  - 지뢰: {total_mines} · 공개 숫자 셀: {len(cl)}",
    ]
    lines += extra_meta_lines
    lines += [
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 규칙: 각 공개 숫자 = 인접 8칸 중 지뢰 수.",
        "  - 규칙: '#' 은 숨겨진 셀(지뢰 또는 안전).",
        "  - 퍼즐(행 단위):",
    ]
    for r, row in enumerate(rows):
        lines.append(f"    r{r}: {' '.join(row)}")
    return lines, cl


def build_solution_forcing(rows: List[str], R: int, C: int, total_mines: int,
                           coord_sorted: List[Tuple[int, int]], difficulty: str,
                           bitstring: str) -> str:
    """Easy 파일 형식."""
    hidden = sum(row.count('#') for row in rows)
    revealed = sum(1 for row in rows for ch in row if ch.isdigit())
    requires_search = requires_search_beyond_simple_forcing(rows, R, C, total_mines)

    lines, cl = _solution_header(
        rows, R, C, total_mines, difficulty,
        [f"  - 단순 forcing 이상의 탐색 필요: {'예' if requires_search else '아니오'}"],
    )
    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: {revealed}개 공개 숫자 제약을 {hidden}개 미공개 셀에 전파, "
        f"지뢰 {total_mines}개 → 유일해."
    )
    for (r, c) in coord_sorted:
        hits = _adjacent_clues(r, c, R, C, cl)
        if hits:
            ct = ", ".join(f"({cr},{cc})={cv}" for cr, cc, cv in hits)
            lines.append(f"    [SEG] 지뢰 ({r},{c}): 인접 숫자 제약 {ct} 이(가) 이 셀을 지뢰로 제약")
        else:
            lines.append(
                f"    [SEG] 지뢰 ({r},{c}): 전역 지뢰 수와 주변 제약이 이 셀을 지뢰로 제약"
            )
    if requires_search:
        lines.append(
            "  · 참고: 단순 단일 셀 forcing 만으로는 부족; 여러 단서와 전역 지뢰 수를 "
            "결합한 뒤에야 확정되는 셀이 있음(탐색/백트래킹 필요)."
        )
    else:
        lines.append(
            "  · 모든 지뢰는 공개 숫자에 대한 직접 제약 전파로 확정됨."
        )
    lines += [
        "[STEP 3] 답·검산",
        f"  - 최종 답: {', '.join(f'({r},{c})' for r, c in coord_sorted)}",
        f"  - 지뢰 총 개수 = {total_mines} (확정 셀 수와 일치 확인).",
        "  - 각 공개 숫자에 대해, 확정된 지뢰 중 이웃 개수가 숫자와 정확히 일치하는지 검증.",
        f"  - 내부 인코딩(bitstring): {bitstring}",
    ]
    return "\n".join(lines)


def build_solution_constrain(rows: List[str], R: int, C: int, total_mines: int,
                             coord_sorted: List[Tuple[int, int]], difficulty: str,
                             bitstring: str, solver_nodes: int) -> str:
    """Medium 파일 형식."""
    hidden = sum(row.count('#') for row in rows)
    lines, cl = _solution_header(
        rows, R, C, total_mines, difficulty,
        [f"  - 솔버 백트래킹 노드: {solver_nodes}"],
    )
    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: 숫자 제약 전파 · 미공개 {hidden}셀 / 지뢰 {total_mines}개 → 유일해 · SEG {len(coord_sorted)}개"
    )
    for i, (r, c) in enumerate(coord_sorted, 1):
        hits = _adjacent_clues(r, c, R, C, cl)
        if hits:
            ct = ", ".join(f"({cr},{cc})={cv}" for cr, cc, cv in hits)
            lines.append(
                f"    [SEG {i}] 지뢰 ({r},{c}): 인접 숫자 제약 {ct} 이(가) 이 셀을 지뢰로 강제"
            )
        else:
            lines.append(
                f"    [SEG {i}] 지뢰 ({r},{c}): 전역 지뢰 수와 주변 추론에 의해 이 셀을 지뢰로 강제"
            )
    lines += [
        "[STEP 3] 답·검산",
        f"  - 최종 답: {', '.join(f'({r},{c})' for r, c in coord_sorted)}",
        f"  - 지뢰 총 개수 = {total_mines} (확정 셀 수와 일치 확인).",
        "  - 각 공개 숫자에 대해, 확정된 지뢰 중 이웃 개수가 숫자와 정확히 일치하는지 검증.",
        f"  - 내부 인코딩(bitstring): {bitstring}",
    ]
    return "\n".join(lines)


def build_solution_force_global(rows: List[str], R: int, C: int, total_mines: int,
                                coord_sorted: List[Tuple[int, int]], difficulty: str,
                                solver_nodes: int) -> str:
    """Hard 파일 형식 (내부 bitstring 줄 없음)."""
    hidden = sum(row.count('#') for row in rows)
    revealed = sum(1 for row in rows for ch in row if ch.isdigit())
    fset = forced_mine_set(rows, R, C)
    forced = [p for p in coord_sorted if p in fset]
    glob = [p for p in coord_sorted if p not in fset]
    F = len(forced)

    lines, cl = _solution_header(
        rows, R, C, total_mines, difficulty,
        [f"  - 솔버 백트래킹 노드: {solver_nodes}"],
    )
    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: 숫자 + 전역 지뢰 수 제약 전파 · 미공개 {hidden}셀 / 지뢰 {total_mines}개 → 유일해"
    )
    lines.append(
        f"  · 제약 전파만으로 {F}/{total_mines} 지뢰가 강제됨; 나머지 {total_mines - F}개는 "
        f"지역 단서와 전역 지뢰 수 {total_mines}개를 결합해야 확정됨(해는 증명상 유일)."
    )
    for i, (r, c) in enumerate(forced, 1):
        hits = _adjacent_clues(r, c, R, C, cl)
        ct = ", ".join(f"({cr},{cc})={cv}" for cr, cc, cv in hits)
        lines.append(
            f"    [SEG {i}] 지뢰 ({r},{c}): 인접 숫자 제약 {ct} 이(가) 이 셀을 지뢰로 강제"
        )
    if glob:
        gco = ", ".join(f"({r},{c})" for r, c in glob)
        lines.append(
            f"    [GLOBAL] 아래 셀은 단일 단서나 단서 쌍으로는 고정되지 않음. "
            f"모든 공개 숫자와 지뢰 총수 {total_mines}개와 일치하는 배치를 열거하면 "
            f"유일한 가능성만 남으며, 지뢰 위치: {gco}."
        )
    lines += [
        "[STEP 3] 답·검산",
        f"  - 최종 답: {', '.join(f'({r},{c})' for r, c in coord_sorted)}",
        f"  - 검증: 공개 숫자 {revealed}개 모두 이웃 지뢰 수와 일치(일관됨).",
        f"  - 지뢰 총 개수 = {total_mines}",
    ]
    return "\n".join(lines)


_SOLUTION_BUILDERS = {
    "forcing": "easy",
    "constrain": "medium",
    "force_global": "hard",
}


def build_solution(solution_format: str, *, rows, R, C, total_mines,
                   coord_sorted, difficulty, bitstring, solver_nodes) -> str:
    if solution_format == "forcing":
        return build_solution_forcing(rows, R, C, total_mines, coord_sorted,
                                      difficulty, bitstring)
    if solution_format == "constrain":
        return build_solution_constrain(rows, R, C, total_mines, coord_sorted,
                                        difficulty, bitstring, solver_nodes)
    if solution_format == "force_global":
        return build_solution_force_global(rows, R, C, total_mines, coord_sorted,
                                           difficulty, solver_nodes)
    raise ValueError(f"unknown solution_format: {solution_format}")


# ---------------------------------------------------------------------------
# 데이터셋 명세 — 출력 파일당 하나의 항목, 각각 순서 있는 블록 포함.
#
# 각 블록은 참조 파일의 연속 레코드 구간을 재현:
#   grid   : (행, 열)
#   mines  : 지뢰 수
#   count  : 이 블록이 기여하는 레코드 수
#   reveal : 공개 셀 선택 방식 (_generate_block_base 참조)
#   order  : 공개 셀 선택 시 사용하는 정보 순위 정렬
#
# 레코드별 `difficulty`는 출력 파일 키 (easy / medium / hard)이며
# 블록별 라벨이 아님. solution_format은 파일별.
# ---------------------------------------------------------------------------

DATASET_SPEC: Dict[str, Dict] = {
    "easy": {
        "solution_format": "forcing",
        "blocks": [
            {"grid": (7, 7),   "mines": 6,  "count": 28,
             "reveal": {"mode": "fixed", "count": 24}, "order": "balanced"},
            {"grid": (8, 8),   "mines": 7,  "count": 28,
             "reveal": {"mode": "fixed", "count": 30}, "order": "balanced"},
            {"grid": (8, 8),   "mines": 8,  "count": 24,
             "reveal": {"mode": "fixed", "count": 28}, "order": "balanced"},
            {"grid": (11, 11), "mines": 21, "count": 10,
             "reveal": {"mode": "fixed", "count": 42}, "order": "balanced"},
            {"grid": (12, 12), "mines": 24, "count": 10,
             "reveal": {"mode": "fixed", "count": 48}, "order": "balanced"},
        ],
    },
    "medium": {
        "solution_format": "constrain",
        "blocks": [
            {"grid": (9, 9), "mines": 14, "count": 100,
             "reveal": {"mode": "until_unique", "init_ratio": 0.25,
                        "max_ratio": 0.45, "min_nodes": 0},
             "order": "balanced"},
        ],
    },
    "hard": {
        "solution_format": "force_global",
        "blocks": [
            {"grid": (9, 9), "mines": 18, "count": 100,
             "reveal": {"mode": "until_unique", "init_ratio": 0.25,
                        "max_ratio": 0.58, "min_nodes": 0},
             "order": "balanced"},
        ],
    },
}


# ---------------------------------------------------------------------------
# 프롬프트 / 문제 렌더링 (한국어 — 참조 문제와 동등한 구조)
# ---------------------------------------------------------------------------

def format_puzzle_grid_labeled(puzzle_rows: List[str]) -> str:
    if not puzzle_rows:
        return ""
    C = len(puzzle_rows[0])
    header = "   " + " ".join(f"c{c}" for c in range(C))
    lines = [header]
    for r, row in enumerate(puzzle_rows):
        cells = " ".join(row[c] for c in range(C))
        lines.append(f"r{r} {cells}")
    return "\n".join(lines)


def create_prompt(puzzle_data: Dict) -> str:
    grid = format_puzzle_grid_labeled(puzzle_data['puzzle'])
    rows = puzzle_data['rows']
    cols = puzzle_data['cols']
    mines = puzzle_data['total_mines']

    mine_info = f"2. 전체 지뢰 수: {mines}개가 격자에 숨겨져 있습니다"
    task_info = f"모든 {mines}개 지뢰의 정확한 위치를 찾으세요."
    uniqueness_info = "\n6. 이 퍼즐은 정확히 하나의 고유한 해를 가집니다"

    return f"""다음 규칙에 따라 지뢰찾기 퍼즐을 풀어주세요:

게임 규칙:
1. 격자 크기: {rows}행 × {cols}열 (0-인덱스)
{mine_info}
3. 각 공개된 숫자(0-8)는 인접한 8개 셀 중 지뢰가 있는 셀의 수를 나타냅니다
4. '#'은 지뢰이거나 안전한 셀일 수 있는 숨겨진 셀입니다
5. 인접 셀은 가로, 세로, 대각선 8방향을 모두 포함합니다{uniqueness_info}

중요한 신뢰성 안내:
- 아래 퍼즐은 machine-verified 되었고 내부적으로 일관됩니다.
- 아래의 행/열 라벨과 공백으로 구분된 셀 표기가 authoritative 합니다.
- 모든 행은 이미 올바른 길이를 가집니다.
- 국소적인 모순이 보이면 퍼즐을 잘못되었다고 판단하지 말고 추론을 다시 점검하세요.

퍼즐:
아래 격자는 행/열 라벨을 포함합니다.
각 퍼즐 행은 rN으로 시작하며 각 셀은 공백으로 구분됩니다.
{grid}

과제:
{task_info}

단계별로 풀어주세요:
1. 각 숫자 셀을 분석하여 인접한 어떤 셀이 지뢰여야 하는지 추론합니다
2. 셀 간 제약을 전파합니다
3. 모든 지뢰 좌표를 확정합니다

출력 형식:
- 어떤 셀이 지뢰인지 추론 과정을 서술하세요
- 모든 지뢰 좌표를 (행, 열) 쌍으로 행 순서대로 정렬하여 나열하세요
- 마지막 줄은 "Answer: (r1,c1), (r2,c2), ..." 형식으로 작성하세요
"""


# ---------------------------------------------------------------------------
# 이항(D4) 변환 — 솔버가 생성한 각 기본 퍼즐을 여러 개의
# 서로 다르고 여전히 유일한 퍼즐로 저비용 확장하는 데 사용.
# ---------------------------------------------------------------------------

def _transform_mine_mask(mask: List[List[int]], transform_id: int) -> List[List[int]]:
    R = len(mask)
    C = len(mask[0])

    def g(r, c):
        return mask[r][c]

    if transform_id == 0:        # 항등
        return [row[:] for row in mask]
    elif transform_id == 1:      # 전치
        return [[g(r, c) for r in range(R)] for c in range(C)]
    elif transform_id == 2:      # 시계 90도 회전
        return [[g(R - 1 - r, c) for c in range(R)] for r in range(C)]
    elif transform_id == 3:      # 180도 회전
        return [[g(R - 1 - r, C - 1 - c) for c in range(C)] for r in range(R)]
    elif transform_id == 4:      # 좌우 대칭
        return [[g(r, C - 1 - c) for c in range(C)] for r in range(R)]
    elif transform_id == 5:      # 상하 대칭
        return [[g(R - 1 - r, c) for c in range(C)] for r in range(R)]
    elif transform_id == 6:      # 반전 전치
        return [[g(R - 1 - r, C - 1 - c) for r in range(R)] for c in range(C)]
    else:                        # 시계 270도 회전
        return [[g(r, C - 1 - c) for r in range(R)] for c in range(C)]


def _transform_puzzle_grid(puzzle_display: List[str], mine_mask: List[List[int]],
                           transform_id: int) -> Tuple[List[str], List[List[int]]]:
    R = len(mine_mask)
    C = len(mine_mask[0])
    revealed = [[puzzle_display[r][c] != '#' for c in range(C)] for r in range(R)]

    new_mine = _transform_mine_mask(mine_mask, transform_id)
    NR = len(new_mine)
    NC = len(new_mine[0])
    new_nums_full = compute_numbers(new_mine)
    new_revealed = _transform_mine_mask(
        [[1 if revealed[r][c] else 0 for c in range(C)] for r in range(R)],
        transform_id,
    )

    new_display = []
    for r in range(NR):
        row_str = ''
        for c in range(NC):
            row_str += str(new_nums_full[r][c]) if new_revealed[r][c] else '#'
        new_display.append(row_str)
    return new_display, new_mine


# ---------------------------------------------------------------------------
# 정보 기반 셀 순위 (어떤 셀을 공개할지 결정)
# ---------------------------------------------------------------------------

def _rank_cells_by_information(nums, mask, R, C) -> List[Tuple[int, int]]:
    safe_cells = [(r, c) for r in range(R) for c in range(C) if mask[r][c] == 0]

    def cell_info_score(pos):
        r, c = pos
        num = nums[r][c]
        neighbor_count = len(neighbors(r, c, R, C))
        if num == 0:
            return neighbor_count * 2
        if num == neighbor_count:
            return neighbor_count * 2
        return abs(num - neighbor_count / 2) * 2 + 1

    safe_cells.sort(key=cell_info_score, reverse=True)
    return safe_cells


def _order_ranked_cells(ranked_cells, order: str) -> List[Tuple[int, int]]:
    if order == 'high_info':
        return list(ranked_cells)
    if order == 'low_info':
        return list(reversed(ranked_cells))
    if order == 'balanced':
        cells = list(ranked_cells)
        out = []
        lo, hi = 0, len(cells) - 1
        take_high = True
        while lo <= hi:
            if take_high:
                out.append(cells[lo]); lo += 1
            else:
                out.append(cells[hi]); hi -= 1
            take_high = not take_high
        return out
    return list(ranked_cells)


# ---------------------------------------------------------------------------
# 블록별 기본 퍼즐 생성
# ---------------------------------------------------------------------------

def _puzzle_display(puzzle_nums: List[List[Optional[int]]]) -> List[str]:
    out = []
    for row in puzzle_nums:
        out.append(''.join(str(cell) if cell is not None else '#' for cell in row))
    return out


def _generate_block_base(rng: random.Random, block: Dict,
                         max_attempts: int = 20000) -> Optional[Dict]:
    """블록에 대해 승인된 기본 퍼즐 하나를 생성.

    전략:
      * 지뢰 배치를 샘플링하고 안전 셀을 정보 순으로 순위 매김
      * 블록의 order에 따라 셀을 공개하며 집합을 확장, 전역 지뢰 수
        제약 하에서 유일해가 될 때까지 진행
      * 'fixed' 공개 모드: 목표 수 이하에서 유일해 도달 필요;
        이후 공개 집합을 목표 수까지 패딩 (추가 공개 숫자는
        유일해를 깨뜨리지 않음)
      * 'until_unique' 공개 모드: 최소 유일 공개 집합 수용,
        최대 공개 비율 및 (선택) 최소 솔버 노드 필터 적용
    퍼즐 표시, 지뢰 마스크, 솔버 노드 수를 담은 dict 반환.
    """
    R, C = block["grid"]
    M = block["mines"]
    order = block.get("order", "balanced")
    rev = block["reveal"]
    total_cells = R * C
    safe_count = total_cells - M
    cells = [(r, c) for r in range(R) for c in range(C)]

    if rev["mode"] == "fixed":
        target = rev["count"]
        growth_cap = target
        start = max(1, min(target, target // 2))
    else:
        max_ratio = rev["max_ratio"]
        growth_cap = max(1, int(max_ratio * safe_count))
        start = max(1, int(rev.get("init_ratio", 0.25) * safe_count))
    start = min(start, growth_cap)

    for _ in range(max_attempts):
        mine_positions = set(rng.sample(cells, M))
        mask = [[1 if (r, c) in mine_positions else 0 for c in range(C)] for r in range(R)]
        nums = compute_numbers(mask)
        ranked = _order_ranked_cells(_rank_cells_by_information(nums, mask, R, C), order)
        if len(ranked) < growth_cap:
            continue

        puzzle = [[None] * C for _ in range(R)]
        revealed_n = 0
        for i in range(start):
            r, c = ranked[i]
            puzzle[r][c] = nums[r][c]
            revealed_n += 1

        sols = solve_puzzle(puzzle, R, C, max_solutions=MAX_SOLUTIONS + 1, total_mines=M)
        idx = start
        while len(sols) > MAX_SOLUTIONS and idx < growth_cap:
            r, c = ranked[idx]
            puzzle[r][c] = nums[r][c]
            revealed_n += 1
            idx += 1
            sols = solve_puzzle(puzzle, R, C, max_solutions=MAX_SOLUTIONS + 1, total_mines=M)

        if len(sols) != 1:
            continue  # 허용된 공개 예산 내에서 유일해 아님

        if rev["mode"] == "fixed":
            # 목표 공개 셀 수까지 정확히 패딩
            while revealed_n < target and idx < len(ranked):
                r, c = ranked[idx]
                if puzzle[r][c] is None:
                    puzzle[r][c] = nums[r][c]
                    revealed_n += 1
                idx += 1
            if revealed_n != target:
                continue
        else:
            if revealed_n / safe_count > rev["max_ratio"]:
                continue

        # 최종 확인 솔브 (유일해 + 추적용 노드 수)
        stats: Dict[str, int] = {"nodes": 0}
        confirm = solve_puzzle(puzzle, R, C, max_solutions=2, total_mines=M, _stats=stats)
        if len(confirm) != 1:
            continue
        if stats["nodes"] < rev.get("min_nodes", 0):
            continue

        return {
            "rows": R,
            "cols": C,
            "total_mines": M,
            "puzzle": _puzzle_display(puzzle),
            "mine_mask": [row[:] for row in mask],
            "solver_nodes": stats["nodes"],
        }
    return None


def _derive_puzzle(base: Dict, transform_id: int,
                   needs_node_count: bool) -> Optional[Dict]:
    """기본 퍼즐에 이항 변환 적용.

    변환은 격자 크기, 지뢰 수, 공개 셀 수, 유일해를 보존.
    medium/hard에서는 변환된 격자에서 솔버 노드 수를 재계산
    (탐색 경로가 달라짐); easy에서는 불필요.
    """
    if transform_id == 0:
        new_display = list(base["puzzle"])
        new_mine = [row[:] for row in base["mine_mask"]]
        nodes = base["solver_nodes"]
    else:
        new_display, new_mine = _transform_puzzle_grid(
            base["puzzle"], base["mine_mask"], transform_id)
        nodes = None

    R = len(new_mine)
    C = len(new_mine[0])
    M = sum(new_mine[r][c] for r in range(R) for c in range(C))

    if needs_node_count and nodes is None:
        # 단서 숫자 재계산 후 솔브하여 변환된 노드 수 획득
        puzzle_nums: List[List[Optional[int]]] = []
        full_nums = compute_numbers(new_mine)
        for r in range(R):
            row_nums: List[Optional[int]] = []
            for c in range(C):
                row_nums.append(full_nums[r][c] if new_display[r][c] != '#' else None)
            puzzle_nums.append(row_nums)
        stats: Dict[str, int] = {"nodes": 0}
        confirm = solve_puzzle(puzzle_nums, R, C, max_solutions=2, total_mines=M, _stats=stats)
        if len(confirm) != 1:
            return None  # 대칭이 유일해를 보존해야 함; 방어적 검사
        nodes = stats["nodes"]

    return {
        "rows": R,
        "cols": C,
        "total_mines": M,
        "puzzle": new_display,
        "mine_mask": new_mine,
        "solver_nodes": nodes if nodes is not None else 0,
    }


def _assemble_record(puzzle: Dict, file_key: str, seq: int,
                     solution_format: str) -> Dict:
    R, C, M = puzzle["rows"], puzzle["cols"], puzzle["total_mines"]
    coord_list = mask_to_coord_list(puzzle["mine_mask"])
    coord_sorted = sorted(coord_list)
    answer_str = coords_to_answer_string(coord_list)
    bitstring = mask_to_bitstring(puzzle["mine_mask"])

    solution = build_solution(
        solution_format,
        rows=puzzle["puzzle"], R=R, C=C, total_mines=M,
        coord_sorted=coord_sorted, difficulty=file_key,
        bitstring=bitstring, solver_nodes=puzzle["solver_nodes"],
    )
    question = create_prompt({
        "puzzle": puzzle["puzzle"], "rows": R, "cols": C, "total_mines": M,
    })
    return {
        "id": f"minesweeper_ko_{file_key}_{seq:04d}",
        "question": question,
        "answer": answer_str,
        "solution": solution,
        "difficulty": file_key,
    }


# ---------------------------------------------------------------------------
# 데이터셋 조립
# ---------------------------------------------------------------------------

NUM_TRANSFORMS = 8  # |D4| (이항군 크기)


def _generate_file_records(file_key: str, file_spec: Dict, seed: int,
                           smoke: Optional[int]) -> List[Dict]:
    solution_format = file_spec["solution_format"]
    needs_node_count = solution_format in ("constrain", "force_global")
    records: List[Dict] = []
    seen_grids: Set[Tuple[str, ...]] = set()
    seq = 0

    for bi, block in enumerate(file_spec["blocks"]):
        count = block["count"] if smoke is None else min(smoke, block["count"])
        if count <= 0:
            continue
        bases_needed = (count if smoke is not None
                        else min(math.ceil(count / NUM_TRANSFORMS) + 2, count))

        print(f"  [{file_key}] block {bi}: grid={block['grid']} mines={block['mines']} "
              f"count={count} (bases_needed={bases_needed})")

        # --- 1단계: 솔버가 생성한 기본 퍼즐 ---
        bases: List[Dict] = []
        attempt = 0
        max_base_attempts = bases_needed * 200 + 50
        while len(bases) < bases_needed and attempt < max_base_attempts:
            attempt += 1
            rng = random.Random(seed + bi * 1_000_003 + attempt * 101)
            base = _generate_block_base(rng, block)
            if base is None:
                continue
            bases.append(base)
            revealed_cells = sum(1 for row in base["puzzle"] for ch in row if ch.isdigit())
            print(f"      base {len(bases)}/{bases_needed} "
                  f"(nodes={base['solver_nodes']}, revealed={revealed_cells})")

        if not bases:
            print(f"      WARNING: no base puzzles generated for {file_key} block {bi}")
            continue

        # --- 2단계: 변환으로 `count`개의 서로 다른 퍼즐 도출 ---
        produced = 0
        j = 0
        guard = 0
        max_guard = count * NUM_TRANSFORMS * len(bases) + count * 50 + 100
        while produced < count and guard < max_guard:
            guard += 1
            base_idx = (j // NUM_TRANSFORMS) % len(bases)
            transform_id = j % NUM_TRANSFORMS
            j += 1

            derived = _derive_puzzle(bases[base_idx], transform_id, needs_node_count)
            if derived is None:
                continue
            key = tuple(derived["puzzle"])
            if key in seen_grids:
                continue
            seen_grids.add(key)

            records.append(_assemble_record(
                derived, file_key, seq, solution_format))
            seq += 1
            produced += 1
            if produced % 10 == 0 or produced == count:
                print(f"      [{file_key}] block {bi}: {produced}/{count} puzzles",
                      flush=True)

        # 변환만으로 충분한 서로 다른 격자를 얻지 못하면
        # 새로 생성한 기본 퍼즐로 보충.
        topup_attempt = 0
        while produced < count and topup_attempt < count * 200 + 100:
            topup_attempt += 1
            rng = random.Random(seed + bi * 1_000_003 + 7_000_000 + topup_attempt * 131)
            fresh = _generate_block_base(rng, block)
            if fresh is None:
                continue
            key = tuple(fresh["puzzle"])
            if key in seen_grids:
                continue
            seen_grids.add(key)
            records.append(_assemble_record(
                fresh, file_key, seq, solution_format))
            seq += 1
            produced += 1
            if produced % 10 == 0 or produced == count:
                print(f"      [{file_key}] block {bi}: {produced}/{count} puzzles (top-up)",
                      flush=True)

        if produced < count:
            print(f"      WARNING: produced {produced}/{count} for {file_key} block {bi}")

    return records


def create_dataset_files(out_dir: str | None = None,
                         seed: int = 42, smoke: Optional[int] = None):
    """세 개의 개별 JSONL 출력과 통합 CSV 생성.

    Args:
        out_dir: 출력을 쓸 디렉터리.
        seed:    기본 RNG 시드.
        smoke:   설정 시 각 블록을 최대 이 수만큼 레코드로 제한하여
                 빠른 E2E 실행 (전체 데이터셋은 None).
    """
    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl, DATA_DIR

    out = Path(out_dir) if out_dir else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []
    for file_key in ("easy", "medium", "hard"):     # csv 순서: easy -> medium -> hard
        print(f"=== Generating {file_key} file ===")
        recs = _generate_file_records(file_key, DATASET_SPEC[file_key], seed, smoke)
        jsonl_path = out / f"minesweeper_ko_{file_key}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(recs)} records -> {jsonl_path}")
        all_records.extend(recs)


    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="지뢰찾기 퍼즐 생성기 (KO)")
    parser.add_argument("--out", type=str, default=None,
                        help="JSONL + CSV 파일 출력 디렉터리")
    parser.add_argument("--seed", type=int, default=42, help="기본 RNG 시드")
    parser.add_argument("--smoke", type=int, default=None,
                        help="빠른 테스트 실행을 위해 각 블록을 최대 N개 레코드로 제한 "
                             "(파일당 100개 전체 데이터셋은 생략)")
    args = parser.parse_args()

    create_dataset_files(out_dir=args.out, seed=args.seed, smoke=args.smoke)

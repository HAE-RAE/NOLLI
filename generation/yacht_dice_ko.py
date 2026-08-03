"""요트 다이스 문제 생성기 (한국어 버전)

난이도 기반 주사위 패턴으로 요트 다이스 최적화 문제를 생성합니다.
보너스 인지 완전 탐색 풀이기(C(12,6) x 720 x 2)를 사용합니다.

logical-puzzles-me/yacht_dice/generator.py 에서 이식:
- greedy_gap 지표용 그리디 참조 풀이기
- 라운드별 top1/top2 margin 및 decision_complexity 지표
- 코프라임 시드 재시도를 통한 델타 밴드 난이도 필터링
- step_metrics는 생성 시 내부 지표로만 사용 (CSV/JSONL에는 미포함)

스팟체크 헬퍼(이전 evaluation/yacht_dice_spotcheck.py)는
이 파일이 생성의 단일 진실 출처가 되도록 아래에 인라인 정의됩니다.
"""

import hashlib
import random
import json
import itertools
import sys
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import List, Dict, Tuple, Optional, Literal, Callable
from collections import Counter
from dataclasses import dataclass

import numpy as np


# ============================================================
# 스팟체크 헬퍼 (단일 진실 출처)
# ============================================================

# 스팟체크 부분합에서 사용할 라운드 수 (0 => 정답은 전체 12라운드 최적 총점).
# 보정 (목표 75/50/25; 수정 전 측정 easy 99%, medium 54%, hard 43%):
# easy 구성적 생성 제거→랜덤 in-band, K 3→2 / medium 밴드 소폭 완화 / hard band_violation 거부.
SPOTCHECK_K: Dict[str, int] = {
    "easy": 2,
    "medium": 2,
    "hard": 8,
}


def deterministic_round_pick_1based(public_id: str, k: int) -> List[int]:
    """``public_id``에 대해 안정적인 1..12 범위의 ``k``개 라운드 인덱스 반환."""
    if k <= 0:
        return []
    digest = hashlib.sha256(f"yacht_dice_spotcheck_v1|{public_id}".encode()).digest()
    h = int.from_bytes(digest[:16], "big")
    combs = list(combinations(range(1, 13), k))
    chosen = combs[h % len(combs)]
    return sorted(chosen)


def _category_for_round(optimal_assignment: Dict, round_idx0: int) -> str:
    if round_idx0 in optimal_assignment:
        return optimal_assignment[round_idx0]
    return optimal_assignment[str(round_idx0)]


def spotcheck_grading_total(
    difficulty: str,
    public_id: str,
    dice_results: List[List[int]],
    optimal_assignment: Dict,
    score_fn: Callable[[List[int], str], int],
) -> Optional[int]:
    """스팟체크 라운드의 부분합, ``k == 0``이면 ``None`` 반환."""
    k = SPOTCHECK_K.get(difficulty, 0)
    if k <= 0:
        return None
    total = 0
    for r1 in deterministic_round_pick_1based(public_id, k):
        i0 = r1 - 1
        cat = _category_for_round(optimal_assignment, i0)
        total += score_fn(dice_results[i0], cat)
    return total


def append_spotcheck_user_suffix(
    base_question: str,
    difficulty: str,
    public_id: str,
) -> str:
    """``SPOTCHECK_K[difficulty] > 0``일 때 스팟체크 안내를 질문에 추가."""
    k = SPOTCHECK_K.get(difficulty, 0)
    if k <= 0:
        return base_question

    rounds = deterministic_round_pick_1based(public_id, k)
    rlist = ", ".join(str(r) for r in rounds)

    suffix = (
        f"\n\n중요 — 스팟체크 채점 (12라운드 전체 총점이 아님):\n"
        f"먼저 기존 규칙과 보너스와 동일하게 12라운드를 한 번씩 배정해 전체 총점을 "
        f"최대화하는 최적 배정을 구하세요. 그 최적 배정을 기준으로, 라운드 {rlist}에 "
        f"배정된 칸의 점수 합만 출력하세요. 마지막 줄은 반드시 "
        f"`Answer: <정수>` 형식이며, 위 부분합 하나의 정수여야 합니다.\n"
    )

    return base_question + suffix


# (0..5)의 720개 순열을 한 번만 미리 계산 (모든 solve 호출에서 재사용).
_PERMS_6 = np.array(list(itertools.permutations(range(6))), dtype=np.int8)  # (720, 6)
_ROW_IDX_6 = np.arange(6, dtype=np.int8)  # fancy-indexing 용


# ============================================================
# 설정
# ============================================================

@dataclass
class YachtDiceConfig:
    """요트 다이스 점수 규칙용 설정"""

    bonus_threshold: int = 63
    bonus_points: int = 35

    full_house_points: int = 25
    small_straight_points: int = 30
    large_straight_points: int = 40
    yacht_points: int = 50

    optimization_goal: Literal["maximize", "minimize"] = "maximize"

    def get_system_prompt(self) -> str:
        """Answer: 형식 지시가 포함된 한국어 시스템 프롬프트 반환"""
        goal_word = "최대" if self.optimization_goal == "maximize" else "최소"
        return f"""당신은 요트 다이스 최적화 문제를 푸는 전문가입니다.

요트 다이스는 5개의 주사위를 12라운드 동안 굴려 각 라운드를 점수 카테고리에 배정하는 주사위 게임입니다.

점수 카테고리:
- 에이스~식스: 해당 숫자가 나온 주사위의 합
- 쓰리 오브 어 카인드: 3개 이상 같은 숫자면 모든 주사위의 합
- 포 오브 어 카인드: 4개 이상 같은 숫자면 모든 주사위의 합
- 풀 하우스: 한 숫자 3개와 다른 숫자 2개면 {self.full_house_points}점
- 스몰 스트레이트: 연속 4개 숫자면 {self.small_straight_points}점
- 라지 스트레이트: 연속 5개 숫자면 {self.large_straight_points}점
- 요트: 5개 모두 같은 숫자면 {self.yacht_points}점

상단 섹션 보너스: 에이스~식스의 합이 {self.bonus_threshold} 이상이면 {self.bonus_points} 보너스 점수가 추가됩니다.

각 라운드를 카테고리에 최적 배정하여 {goal_word} 총점을 구하세요.
각 카테고리는 한 번만 사용할 수 있습니다.

필수: 반드시 마지막 줄을 아래 형식으로 작성하세요:
Answer: [숫자]

이 줄을 생략하지 마세요. 문제에서 요구하는 숫자(총점 또는 스팟체크 부분합 등)를 제시하세요.
"""

    def get_user_prompt(self, dice_results: List[List[int]]) -> str:
        """점수 규칙을 명시한 한국어 사용자 프롬프트 생성."""
        prompt = (
            "다음 12라운드의 주사위 결과가 주어졌을 때, 최적으로 카테고리를 배정하세요.\n\n"
            "점수 규칙 (아래 점수를 정확히 사용하세요):\n"
            f"- 풀 하우스(Full House): {self.full_house_points}점\n"
            f"- 스몰 스트레이트(Small Straight): {self.small_straight_points}점\n"
            f"- 라지 스트레이트(Large Straight): {self.large_straight_points}점\n"
            f"- 요트(Yacht): {self.yacht_points}점\n"
            "- 에이스~식스: 해당 숫자가 나온 주사위의 합\n"
            "- 쓰리/포 오브 어 카인드: 조건을 만족하면 모든 주사위의 합\n"
            f"- 상단 보너스: 에이스~식스 합이 {self.bonus_threshold} 이상이면 "
            f"+{self.bonus_points}점\n\n"
        )
        for i, dice in enumerate(dice_results):
            prompt += f"라운드 {i+1}: {dice}\n"
        prompt += (
            "\n최적 배정을 계산하세요. 아래 추가 안내에서 특정 라운드를 지정하면, "
            "12라운드 전체 총점이 아니라 요청된 부분합만 제시하세요."
        )
        return prompt


# ============================================================
# 점수 기본 함수
# ============================================================

def get_all_categories() -> List[str]:
    return [
        "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
        "Three-Of-A-Kind", "Four-Of-A-Kind", "Full House",
        "Small Straight", "Large Straight", "Yacht"
    ]


UPPER_CATEGORIES = ["Aces", "Twos", "Threes", "Fours", "Fives", "Sixes"]


def calculate_score_with_config(dice: List[int], category: str, config: YachtDiceConfig) -> int:
    """사용자 지정 설정값으로 점수 계산."""
    counts = Counter(dice)
    sorted_dice = sorted(dice)

    if category == "Aces":
        return dice.count(1) * 1
    elif category == "Twos":
        return dice.count(2) * 2
    elif category == "Threes":
        return dice.count(3) * 3
    elif category == "Fours":
        return dice.count(4) * 4
    elif category == "Fives":
        return dice.count(5) * 5
    elif category == "Sixes":
        return dice.count(6) * 6
    elif category == "Three-Of-A-Kind":
        for _, count in counts.items():
            if count >= 3:
                return sum(dice)
        return 0
    elif category == "Four-Of-A-Kind":
        for _, count in counts.items():
            if count >= 4:
                return sum(dice)
        return 0
    elif category == "Full House":
        counts_values = sorted(counts.values())
        if counts_values == [2, 3]:
            return config.full_house_points
        return 0
    elif category == "Small Straight":
        unique = set(sorted_dice)
        for straight in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]:
            if straight.issubset(unique):
                return config.small_straight_points
        return 0
    elif category == "Large Straight":
        unique = set(sorted_dice)
        if unique == {1, 2, 3, 4, 5} or unique == {2, 3, 4, 5, 6}:
            return config.large_straight_points
        return 0
    elif category == "Yacht":
        if len(counts) == 1:
            return config.yacht_points
        return 0
    return 0


def calculate_score(dice: List[int], category: str) -> int:
    """기본 설정으로 점수 계산."""
    return calculate_score_with_config(dice, category, YachtDiceConfig())


def calculate_total_score(assignment: Dict[int, str], dice_results: List[List[int]],
                          config: YachtDiceConfig) -> int:
    """배정에 대한 총점 계산 (상단 섹션 보너스 포함)."""
    upper_section_score = 0
    total_score = 0

    for dice_idx, category in assignment.items():
        dice = dice_results[dice_idx]
        score = calculate_score_with_config(dice, category, config)
        total_score += score
        if category in UPPER_CATEGORIES:
            upper_section_score += score

    if upper_section_score >= config.bonus_threshold:
        total_score += config.bonus_points

    return total_score


# ============================================================
# 풀이기
# ============================================================

def solve_yacht_dice(dice_results: List[List[int]], config: YachtDiceConfig) -> Tuple[int, Dict[int, str], bool]:
    """
    보너스 인지 완전 탐색 최적 풀이기 + 최적 배정 유일성 검사.

    (best_total, best_assignment, is_unique)를 반환. 상단·하단 섹션의 6! 열거 중
    동점 배정을 수집하여 다중 최적해 존재 여부를 판별.
    """
    categories = get_all_categories()
    upper_cats = categories[:6]
    lower_cats = categories[6:]
    n = len(dice_results)

    # 점수 행렬 사전 계산 (numpy 벡터화).
    upper_scores = np.zeros((n, 6), dtype=np.int32)
    lower_scores = np.zeros((n, 6), dtype=np.int32)
    for i in range(n):
        for j, cat in enumerate(upper_cats):
            upper_scores[i, j] = calculate_score_with_config(dice_results[i], cat, config)
        for j, cat in enumerate(lower_cats):
            lower_scores[i, j] = calculate_score_with_config(dice_results[i], cat, config)

    is_maximize = config.optimization_goal == "maximize"
    best_total = -1 if is_maximize else float('inf')
    optimal_assignments: set = set()

    perms = _PERMS_6              # (720, 6)
    perms_list = perms.tolist()   # 동점 키 수집 시 빠른 Python 순회용

    # C(12, 6) = 924 부분집합 순회. numpy 로 720 perm 일괄 평가.
    for upper_rounds in itertools.combinations(range(n), 6):
        lower_rounds = tuple(i for i in range(n) if i not in upper_rounds)
        upper_list = list(upper_rounds)

        # 부분 행렬 shape (6, 6) — round_in_subset × category
        sub_upper = upper_scores[upper_list, :]
        sub_lower = lower_scores[list(lower_rounds), :]

        # 모든 perm 합 벡터화: sums[k] = sum_j sub[j, perms[k, j]]
        upper_sums = sub_upper[_ROW_IDX_6, perms].sum(axis=1)  # (720,)
        lower_sums = sub_lower[_ROW_IDX_6, perms].sum(axis=1)

        if is_maximize:
            best_upper_score = int(upper_sums.max())
            best_lower_score = int(lower_sums.max())
        else:
            best_upper_score = int(upper_sums.min())
            best_lower_score = int(lower_sums.min())

        bonus = config.bonus_points if best_upper_score >= config.bonus_threshold else 0
        total = best_upper_score + best_lower_score + bonus

        if (is_maximize and total > best_total) or (not is_maximize and total < best_total):
            best_total = total
            optimal_assignments.clear()

        if total == best_total:
            # 이 부분집합이 best_total과 동점 — 동점 permutation만 열거.
            upper_tie_mask = upper_sums == best_upper_score
            lower_tie_mask = lower_sums == best_lower_score
            best_upper_perms = [perms_list[i] for i in np.flatnonzero(upper_tie_mask)]
            best_lower_perms = [perms_list[i] for i in np.flatnonzero(lower_tie_mask)]

            stop = False
            for up in best_upper_perms:
                for lp in best_lower_perms:
                    key = tuple(
                        (upper_list[j], upper_cats[up[j]]) for j in range(6)
                    ) + tuple(
                        (lower_rounds[j], lower_cats[lp[j]]) for j in range(6)
                    )
                    optimal_assignments.add(key)
                    if len(optimal_assignments) > 1:
                        stop = True
                        break
                if stop:
                    break

    is_unique = len(optimal_assignments) == 1
    best_assignment: Dict[int, str] = {}
    if optimal_assignments:
        first = next(iter(optimal_assignments))
        best_assignment = {k: v for k, v in first}
    return best_total, best_assignment, is_unique


def solve_yacht_dice_greedy(
    dice_results: List[List[int]], config: YachtDiceConfig
) -> Tuple[int, Dict[int, str]]:
    """greedy_gap 지표용 근시안(myopic) 그리디 기준 풀이기.

    각 라운드를 순서대로 순회하며 남은 카테고리 중 즉시 점수가 가장 높은
    카테고리를 선택. 미래 탐색이나 보너스 플래닝 없음.
    """
    categories = get_all_categories()
    available = set(categories)
    assignment: Dict[int, str] = {}
    is_maximize = config.optimization_goal == "maximize"

    for idx, dice in enumerate(dice_results):
        best_cat = None
        best_score = -1 if is_maximize else float('inf')
        for cat in available:
            s = calculate_score_with_config(dice, cat, config)
            if (is_maximize and s > best_score) or (not is_maximize and s < best_score):
                best_score = s
                best_cat = cat
        if best_cat is None:
            best_cat = next(iter(available))
        assignment[idx] = best_cat
        available.discard(best_cat)

    total = calculate_total_score(assignment, dice_results, config)
    return total, assignment


def format_solution(dice_results: List[List[int]], assignment: Dict[int, str],
                    config: YachtDiceConfig) -> str:
    """풀이를 읽기 쉬운 형태로 포맷팅."""
    categories = get_all_categories()
    result = []
    upper_section_score = 0
    total_score = 0

    for category in categories:
        dice_idx = None
        for idx, cat in assignment.items():
            if cat == category:
                dice_idx = idx
                break

        if dice_idx is not None:
            dice = dice_results[dice_idx]
            score = calculate_score_with_config(dice, category, config)
            result.append(f"{category}: {dice} => {score}")
            total_score += score
            if category in UPPER_CATEGORIES:
                upper_section_score += score

    bonus = 0
    if upper_section_score >= config.bonus_threshold:
        bonus = config.bonus_points
        total_score += bonus

    output = "\n".join(result)
    output += f"\n\n상단 섹션 합계: {upper_section_score}"
    output += f"\n상단 섹션 보너스: {bonus} (기준: {config.bonus_threshold})"
    output += f"\n총점: {total_score}"
    return output


# ============================================================
# 난이도별 주사위 생성기
# ============================================================

DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    # easy dice 재구조화. roll type diversification 은 acc 17→20% 미세 개선뿐.
    #   four_kind_high 의 "sum of all dice" 규칙이 모델에게 까다로움 → 제거.
    #   full_house 는 (3a+2b vs 25 비교) 판단이 필요 → 제거.
    #   yacht / LS / SS 만으로 압축 → 카테고리 매핑이 거의 결정적.
    #   K=1 + is_unique 필터와 결합하여 ~30-50% 기대.
    "easy": {
        # K=2 합산 정답용 고정 점수 위주 풀.
        # 스팟 라운드가 Yacht/Straight/Full House(고정값)에 배정되어야 하므로
        # 해당 타입에 가중치를 높이고, three_kind 를 소량 추가해 비-스팟 다양성을
        # 확보하되 수락을 막지 않도록 함.
        "roll_types": [
            "yacht", "large_straight", "small_straight", "full_house",
            "three_kind",
        ],
        "weights": [26, 22, 26, 22, 4],
    },
    "medium": {
        # four_kind / full_house 비중 확대 (pair-only 대비 판단 난이도 상승).
        "roll_types": ["four_kind_high", "full_house", "three_kind", "pair", "normal"],
        "weights": [18, 17, 22, 22, 21],
    },
    "hard": {
        "roll_types": ["three_kind", "pair", "normal"],
        "weights": [10, 15, 75],
    },
}


# greedy_gap δ 의 극단적 불일치 밴드 (docs/difficulty_definition.md §2.6).
# yacht_dice 의 단계 수 프록시는 약함 (풀이기가 결정론적이며 대부분의 "작업"은
# 주사위와 무관하게 12! 배정 탐색). 명백한 이상값만 걸러냄 — 예: hard 퍼즐에서
# greedy 가 이미 최적에 근접하거나, easy 퍼즐에서 greedy 를 크게 속이는 경우.
_DIFFICULTY_BANDS = {
    # 보정 (목표 75/50/25; 수정 전 측정 easy 99%, medium 54%, hard 43%).
    "easy": {
        "greedy_gap_abs": {"min": 0, "max": 8},
        "decision_complexity": {"min": 0.0, "max": 3.0},
    },
    "medium": {
        "greedy_gap_abs": {"min": 7, "max": 30},
        "decision_complexity": {"min": 3.5, "max": 7.0},
    },
    "hard": {
        # 100개 정상 생성되던 floor 유지 (gap40/comp10 은 생성 starvation).
        # hard 난이도는 SPOTCHECK_K hard=9 로 강화.
        "greedy_gap_abs": {"min": 30, "max": None},
        "decision_complexity": {"min": 8.0, "max": None},
    },
}


# ============================================================
# Easy 티어 구성적 빌더
# ============================================================
# Easy 과제는 GLOBAL 최적 배정에서 K개 스팟체크 라운드의 합을 채점한다.
# 무작위 주사위로는 스팟 카테고리가 전역적으로 독점되기 어렵고,
# 모델이 예를 들어 Yacht를 다른 라운드에 배정하면 스팟 합이 gold 아래로
# 무너진다 (오차가 모두 음수). 이를 해결하기 위해 주사위를 구성해
# 각 스팟 라운드에 다른 어떤 라운드도 매칭할 수 없는 고점 카테고리를 심고,
# 나머지와 무관하게 전역 최적해가 그 라운드에 반드시 배정하도록 강제한다.

# 구성적 easy 스팟 — 변동 점수만 (Yacht/Straight 제외, K=2 합 난이도 상승).
# easy 는 constructive(자명 ~0.99)와 random(~0.30) 퍼즐의 per-puzzle 믹스.
# 보정: f*0.99 + (1-f)*0.30 = 0.75 -> f ~= 0.69.
EASY_CONSTRUCTIVE_FRACTION = 0.69

EASY_SPOT_CATEGORIES = (
    "full_house", "three_of_a_kind", "four_of_a_kind",
)
# 변동 점수 칸은 per-round margin 기준을 낮춤.
_EASY_VARIABLE_YACHT_CATS = frozenset({"Three-Of-A-Kind", "Four-Of-A-Kind"})


def _pick_easy_spot_cats(k: int, rng: random.Random) -> List[str]:
    """``k``개 easy 스팟 키 샘플."""
    if k <= 0:
        return []
    if k > len(EASY_SPOT_CATEGORIES):
        raise ValueError(f"easy spot k={k} exceeds pool size {len(EASY_SPOT_CATEGORIES)}")
    return rng.sample(EASY_SPOT_CATEGORIES, k)


def _easy_margin_required(yacht_cat: str, default_min: int) -> int:
    if yacht_cat in _EASY_VARIABLE_YACHT_CATS:
        return 2
    return default_min


def _build_easy_spot_dice(cat: str, rng: random.Random) -> List[int]:
    """``cat``을 고립 top1으로 갖고 큰 마진을 확보하는 스팟 라운드 주사위 생성."""
    if cat == "yacht":
        v = rng.randint(1, 6)
        return [v] * 5
    if cat == "large_straight":
        start = rng.choice([1, 2])
        d = list(range(start, start + 5))
        rng.shuffle(d)
        return d
    if cat == "small_straight":
        base = list(rng.choice([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
        # 5번째 주사위를 기존 값으로 중복시켜 라지 스트레이트로 연장되지 않게 함
        # (스팟 라운드가 스몰 스트레이트 30점으로 깔끔하게 유지됨).
        d = base + [rng.choice(base)]
        rng.shuffle(d)
        return d
    if cat == "full_house":
        # a=1로 쓰리 오브 어 카인드 합을 작게 유지(≤15)해
        # 풀 하우스(25점)가 마진 ≥ 10으로 명확한 top1이 되도록 함.
        b = rng.choice([2, 3, 4, 5, 6])
        d = [1, 1, 1, b, b]
        rng.shuffle(d)
        return d
    if cat == "three_of_a_kind":
        v = rng.randint(3, 6)
        others = rng.sample([x for x in range(1, 7) if x != v], 2)
        d = [v, v, v, others[0], others[1]]
        rng.shuffle(d)
        return d
    if cat == "four_of_a_kind":
        v = rng.randint(3, 6)
        other = rng.choice([x for x in range(1, 7) if x != v])
        d = [v, v, v, v, other]
        rng.shuffle(d)
        return d
    raise ValueError(f"알 수 없는 스팟 카테고리: {cat}")


def _build_easy_noise_dice(rng: random.Random) -> List[int]:
    """yacht/straight/full-house/3-4-of-a-kind 를 만들 수 없는 저점 주사위 생성.

    비연속적인 3개 값의 2-1-2 분할을 사용해 스팟 라운드가 의존하는
    카테고리(Yacht/LS/SS/Full House)가 다른 라운드에서 매칭되지 않도록 보장.
    """
    for _ in range(200):
        vals = rng.sample(range(1, 7), 3)
        a, b, c = vals
        dice = [a, a, b, c, c]  # counts 2-2-1 → 쓰리 오브 어 카인드/풀 하우스/요트 불가
        s = set(dice)
        # 스몰 스트레이트(4연속) 거부
        if any({x, x + 1, x + 2, x + 3} <= s for x in (1, 2, 3)):
            continue
        rng.shuffle(dice)
        return dice
    # 안전한 폴백.
    return [1, 1, 3, 3, 5]


def _build_easy_constructive_dice(
    spot_rounds_0based: List[int],
    spot_cats: List[str],
    rng: random.Random,
    num_rounds: int = 12,
) -> List[List[int]]:
    """스팟 라운드에 각 카테고리를 배치하고 나머지를 노이즈로 채움."""
    dice: List[Optional[List[int]]] = [None] * num_rounds
    for idx, cat in zip(spot_rounds_0based, spot_cats):
        dice[idx] = _build_easy_spot_dice(cat, rng)
    for i in range(num_rounds):
        if dice[i] is None:
            dice[i] = _build_easy_noise_dice(rng)
    return [list(d) for d in dice]


class YachtDiceProblemGenerator:
    """난이도 보정 주사위 패턴으로 요트 다이스 문제 생성."""

    def __init__(self):
        self.config = YachtDiceConfig()

    def generate_dice_by_difficulty(self, difficulty: str, seed: int = None) -> List[List[int]]:
        """난이도에 따른 12라운드 주사위 결과 생성."""
        rng = random.Random(seed)
        dice_results: List[List[int]] = []

        if difficulty == "easy":
            cfg = DIFFICULTY_CONFIGS["easy"]
            for _ in range(cfg.get("num_rounds", 12)):
                roll_type = rng.choices(cfg["roll_types"], weights=cfg["weights"], k=1)[0]
                if roll_type == 'yacht':
                    val = rng.randint(1, 6)
                    dice_results.append([val] * 5)
                elif roll_type == 'large_straight':
                    start = rng.choice([1, 2])
                    straight = list(range(start, start + 5))
                    rng.shuffle(straight)
                    dice_results.append(straight)
                elif roll_type == 'small_straight':
                    straight_set = rng.choice(
                        [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
                    )
                    extra = rng.randint(1, 6)
                    dice = list(straight_set) + [extra]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'four_kind_high':
                    val = rng.randint(3, 6)
                    other = rng.choice([x for x in range(1, 7) if x != val])
                    dice = [val] * 4 + [other]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'full_house':
                    nums = rng.sample(range(1, 7), 2)
                    dice = [nums[0]] * 3 + [nums[1]] * 2
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'three_kind':
                    num = rng.randint(1, 6)
                    dice = [num] * 3 + [rng.randint(1, 6) for _ in range(2)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'pair':
                    num = rng.randint(1, 6)
                    dice = [num] * 2 + [rng.randint(1, 6) for _ in range(3)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                else:
                    dice_results.append([rng.randint(1, 6) for _ in range(5)])

        elif difficulty == "medium":
            cfg = DIFFICULTY_CONFIGS["medium"]
            for _ in range(cfg.get("num_rounds", 12)):
                roll_type = rng.choices(cfg["roll_types"], weights=cfg["weights"], k=1)[0]
                if roll_type == 'three_kind':
                    num = rng.randint(1, 6)
                    dice = [num] * 3 + [rng.randint(1, 6) for _ in range(2)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'pair':
                    num = rng.randint(1, 6)
                    dice = [num] * 2 + [rng.randint(1, 6) for _ in range(3)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                else:
                    dice_results.append([rng.randint(1, 6) for _ in range(5)])

        else:  # hard
            cfg = DIFFICULTY_CONFIGS["hard"]
            for _ in range(cfg.get("num_rounds", 12)):
                roll_type = rng.choices(cfg["roll_types"], weights=cfg["weights"], k=1)[0]
                if roll_type == 'three_kind':
                    num = rng.randint(1, 6)
                    dice = [num] * 3 + [rng.randint(1, 6) for _ in range(2)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'pair':
                    num = rng.randint(1, 6)
                    dice = [num] * 2 + [rng.randint(1, 6) for _ in range(3)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                else:
                    dice_results.append([rng.randint(1, 6) for _ in range(5)])

        return dice_results

    def generate_problem(self, difficulty: str, problem_id: int = 1, seed: int = None,
                         max_retries: int = 200, forced_dice: List[List[int]] = None) -> Dict:
        """단일 문제 생성; 지표가 난이도 밴드에 들 때까지 시드 재시도.

        ``forced_dice``가 제공되면 밴드 재시도 루프를 건너뛰고 해당 주사위 구성에
        대한 지표만 계산합니다 (구성적 easy 티어 생성기에서 사용).
        """
        if seed is None:
            difficulty_offset = {"easy": 10000, "medium": 20000, "hard": 30000}
            seed = 1000 + problem_id + difficulty_offset.get(difficulty, 0)

        # 난이도별 설정이 지정된 경우 적용
        diff_cfg = DIFFICULTY_CONFIGS.get(difficulty, {})
        config = YachtDiceConfig(
            bonus_threshold=diff_cfg.get("bonus_threshold", self.config.bonus_threshold),
            bonus_points=self.config.bonus_points,
            full_house_points=self.config.full_house_points,
            small_straight_points=self.config.small_straight_points,
            large_straight_points=self.config.large_straight_points,
            yacht_points=self.config.yacht_points,
            optimization_goal=self.config.optimization_goal,
        )

        band = _DIFFICULTY_BANDS.get(difficulty, {})
        categories = get_all_categories()
        selected = None
        best_complex = None  # 가장 어려운(max-complexity) 후보 (fallback)
        in_band = False
        trial_seed = seed

        effective_retries = 1 if forced_dice is not None else max_retries
        for retry in range(effective_retries):
            trial_seed = seed + retry * 997
            if forced_dice is not None:
                dice_results = [list(r) for r in forced_dice]
            else:
                dice_results = self.generate_dice_by_difficulty(difficulty, trial_seed)
            optimal_score, optimal_assignment, is_unique = solve_yacht_dice(dice_results, config)
            greedy_score, _ = solve_yacht_dice_greedy(dice_results, config)
            greedy_gap = optimal_score - greedy_score
            # abs gap 으로 maximize(gap>=0)와 minimize(gap<=0) 모드를 통일.
            greedy_gap_abs = abs(greedy_gap)

            per_round_margin: List[int] = []
            per_round_top1: List[int] = []
            for dice in dice_results:
                scores = sorted(
                    (calculate_score_with_config(dice, c, config) for c in categories),
                    reverse=True,
                )
                top1, top2 = scores[0], scores[1]
                per_round_top1.append(top1)
                per_round_margin.append(top1 - top2)
            total_decision_complexity = sum(1.0 / max(m, 1) for m in per_round_margin)
            zero_margin_rounds = sum(1 for m in per_round_margin if m == 0)

            upper_sum = sum(
                calculate_score_with_config(dice_results[idx], cat, config)
                for idx, cat in optimal_assignment.items() if cat in UPPER_CATEGORIES
            )
            bonus_applied = upper_sum >= config.bonus_threshold

            selected = {
                'dice_results': dice_results,
                'optimal_score': optimal_score,
                'optimal_assignment': optimal_assignment,
                'is_unique_assignment': is_unique,
                'greedy_score': greedy_score,
                'greedy_gap': greedy_gap,
                'greedy_gap_abs': greedy_gap_abs,
                'per_round_margin': per_round_margin,
                'per_round_top1': per_round_top1,
                'total_decision_complexity': total_decision_complexity,
                'zero_margin_rounds': zero_margin_rounds,
                'bonus_applied': bonus_applied,
            }
            # band_violation fallback 으로 가장 어려운(max-complexity) 후보를 추적
            # (hard 는 out-of-band 를 수락하므로): band-exempt hard dice 가 random
            # last seed 가 아니라 실제로 어려운 dice 가 되도록.
            if best_complex is None or total_decision_complexity > best_complex['total_decision_complexity']:
                best_complex = selected

            # 구성적(forced) 주사위는 밴드 필터링을 완전히 우회.
            if forced_dice is not None:
                in_band = True
                break

            # 밴드 키는 greedy_gap_abs (레거시 'greedy_gap' 도 허용).
            gap_band = band.get('greedy_gap_abs') or band.get('greedy_gap', {})
            complexity_band = band.get('decision_complexity', {})
            ok = True
            if gap_band.get('min') is not None and greedy_gap_abs < gap_band['min']:
                ok = False
            if gap_band.get('max') is not None and greedy_gap_abs > gap_band['max']:
                ok = False
            if complexity_band.get('min') is not None and total_decision_complexity < complexity_band['min']:
                ok = False
            if complexity_band.get('max') is not None and total_decision_complexity > complexity_band['max']:
                ok = False

            if ok:
                in_band = True
                break

        # Fallback: in-band 후보가 없으면 마지막 random seed 가 아니라 가장 어려운
        # 후보를 사용 (band-exempt hard dice 가 어렵게 유지되도록).
        if not in_band and best_complex is not None:
            selected = best_complex
        band_violation = not in_band

        problem = {
            "id": problem_id,
            "difficulty": difficulty,
            "dice_results": selected['dice_results'],
            "answer": selected['optimal_score'],
            "optimal_assignment": {int(k): v for k, v in selected['optimal_assignment'].items()},
            "seed": trial_seed,
            "step_metrics": {
                "per_round_margin": selected['per_round_margin'],
                "per_round_top1": selected['per_round_top1'],
                "total_decision_complexity": selected['total_decision_complexity'],
                "zero_margin_rounds": selected['zero_margin_rounds'],
                "greedy_gap": selected['greedy_gap'],
                "greedy_gap_abs": selected['greedy_gap_abs'],
                "bonus_applied": selected['bonus_applied'],
                "is_unique_assignment": selected['is_unique_assignment'],
                "band_violation": band_violation,
            },
        }
        return problem

    def validate_problem(self, problem: Dict) -> Tuple[bool, str]:
        """문제가 올바르게 구성되었고 풀 수 있는지 검증."""
        try:
            required_fields = ['id', 'difficulty', 'dice_results', 'answer']
            for field in required_fields:
                if field not in problem:
                    return False, f"필수 필드 누락: {field}"

            dice_results = problem['dice_results']
            if len(dice_results) != 12:
                return False, f"12라운드가 필요하지만 {len(dice_results)}라운드가 제공됨"

            for round_idx, dice in enumerate(dice_results):
                if len(dice) != 5:
                    return False, f"라운드 {round_idx+1}: 5개의 주사위가 필요하지만 {len(dice)}개가 제공됨"
                for die in dice:
                    if not (1 <= die <= 6):
                        return False, f"라운드 {round_idx+1}: 유효하지 않은 주사위 값 {die}"

            difficulty = problem.get('difficulty', '')
            diff_cfg = DIFFICULTY_CONFIGS.get(difficulty, {})
            validate_config = YachtDiceConfig(
                bonus_threshold=diff_cfg.get("bonus_threshold", self.config.bonus_threshold),
            )
            optimal_score, _, _ = solve_yacht_dice(dice_results, validate_config)
            if optimal_score != problem['answer']:
                return False, f"정답 불일치: 기대값 {optimal_score}, 실제값 {problem['answer']}"

            return True, "문제가 유효합니다"

        except Exception as e:
            return False, f"검증 오류: {str(e)}"


# ============================================================
# 데이터셋 생성
# ============================================================

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_yacht_solution_ko(
    dice_results: List[List[int]],
    optimal_assignment: Dict[int, str],
    optimal_score: int,
    config: "YachtDiceConfig",
    difficulty: str,
    step_metrics: Dict,
    grading_answer: str,
    spot_rounds_1based: Optional[List[int]] = None,
) -> str:
    """SFT teacher trace: 라운드별 SEG 배정이 있는 요트 다이스."""
    num_rounds = len(dice_results)
    upper_sum = 0
    bonus = 0
    total = 0

    round_to_cat: Dict[int, str] = dict(optimal_assignment)

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {difficulty}",
        f"  - 라운드 수: {num_rounds} · 라운드당 5개 주사위",
        f"  - 상단 섹션 보너스: {config.bonus_points}점 (기준 {config.bonus_threshold})",
        f"  - 탐욕/최적 점수 차: {step_metrics.get('greedy_gap', 0)}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 규칙: 각 라운드의 5개 주사위를 12개 카테고리 중 하나에 정확히 한 번 배정.",
        "  - 규칙: 상단 섹션(Aces~Sixes) 합이 기준 이상이면 보너스.",
        "  - 라운드별 주사위:",
    ]
    for idx, dice in enumerate(dice_results):
        lines.append(f"    R{idx + 1}: {dice}")

    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: DP로 (라운드, 남은 카테고리) 최대 점수 탐색 → 최적 배정 · "
        f"SEG {num_rounds}개"
    )

    for idx in sorted(round_to_cat.keys()):
        dice = dice_results[idx]
        cat = round_to_cat[idx]
        score = calculate_score_with_config(dice, cat, config)
        tag = " (상단)" if cat in UPPER_CATEGORIES else ""
        lines.append(
            f"    [SEG {idx + 1}] R{idx + 1} 주사위 {dice} → {cat}{tag}: {score}점"
        )
        if cat in UPPER_CATEGORIES:
            upper_sum += score
        total += score

    if upper_sum >= config.bonus_threshold:
        bonus = config.bonus_points
        total += bonus

    step3: List[str] = [
        "[STEP 3] 답·검산",
        f'  - 채점용 정답 (데이터셋 "answer"): {grading_answer}',
    ]
    if spot_rounds_1based:
        rlist = ", ".join(str(r) for r in spot_rounds_1based)
        step3.extend([
            f"  - 스팟체크: 라운드 {rlist}의 최적 배정 점수 합만 제출",
            f"  - 전체 최적 총점(참고): {optimal_score}",
        ])
    else:
        step3.append(f"  - 최적 총점: {optimal_score}")
    step3.extend(
        [
            f"  - 상단 섹션 합계: {upper_sum} / 기준 {config.bonus_threshold} → 보너스 {bonus}",
            f"  - 재계산 총점: {total} (위 최적 총점과 일치해야 함)",
            "  - 배정 유일성·보너스 조건이 위 SEG와 일관되는지 확인.",
        ]
    )
    lines.extend(step3)
    return "\n".join(lines)


DATASET_COLUMNS = ("id", "question", "answer", "solution", "difficulty")


def create_dataset_files(
    num_questions: int,
    difficulties: Optional[List[str]] = None,
):
    """요트 다이스 퍼즐 CSV + JSONL 데이터셋 생성.

    난이도별 내부 재시도 예산으로 ``generate_problem``을 호출해 각 시도가
    대체로 빠르게 끝나면서도 in-band dice 를 우선합니다(``_DIFFICULTY_BANDS`` 참고).
    재시도를 소진한 퍼즐은 out-of-band일 수 있으나(step_metrics의 band_violation),
    난이도별 ``YachtDiceConfig`` 기준으로 올바른 최적 점수를 갖습니다.

    ``difficulties``가 단일 티어(예: ``["easy"]``)이면
    ``data/yacht_dice_ko_easy.jsonl``만 작성하고 통합 CSV/JSONL은 건너뜁니다.
    """
    import pandas as pd

    # hard lowered 200->40: max-complexity fallback runs ALL retries per puzzle
    # (in-band is rare), so 200 made hard gen ~50s/puzzle. best-of-40 is still hard.
    fast_retries_by_diff = {"easy": 120, "medium": 150, "hard": 100}

    all_tiers = ["easy", "medium", "hard"]
    if difficulties is None:
        difficulties = list(all_tiers)
    else:
        difficulties = [d.lower() for d in difficulties]
        bad = [d for d in difficulties if d not in all_tiers]
        if bad:
            raise ValueError(f"알 수 없는 난이도 티어: {bad}; 허용값: {all_tiers}")

    print(
        f"{num_questions}개의 요트 다이스 퍼즐을 생성합니다 "
        f"(티어: {', '.join(difficulties)})..."
    )

    generator = YachtDiceProblemGenerator()
    base_cfg = YachtDiceConfig()

    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles: List[Dict] = []
    problem_id = 1

    # Easy: K=2 구성적(변동 스팟) + 전역 전용성 필터.

    def _dice_key(dice_results):
        return tuple(tuple(sorted(r)) for r in dice_results)

    for di, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if di < remainder else 0)
        if count == 0:
            continue

        diff_cfg = DIFFICULTY_CONFIGS.get(difficulty, {})
        config = YachtDiceConfig(
            bonus_threshold=int(diff_cfg.get("bonus_threshold", base_cfg.bonus_threshold)),
            bonus_points=base_cfg.bonus_points,
            full_house_points=base_cfg.full_house_points,
            small_straight_points=base_cfg.small_straight_points,
            large_straight_points=base_cfg.large_straight_points,
            yacht_points=base_cfg.yacht_points,
            optimization_goal=base_cfg.optimization_goal,
        )

        print(f"\n=== {difficulty} 난이도 퍼즐 생성 중 ({count}개 필요) ===")

        seen_keys: set = set()
        diff_success = 0
        retries = 0
        MAX_OUTER_RETRIES = count * 30
        inner_retries = fast_retries_by_diff.get(difficulty, 70)
        easy_margin_min = 10
        easy_value_counts: Dict[int, int] = {}

        while diff_success < count and retries < MAX_OUTER_RETRIES:
            retries += 1
            # 경량 보정: 수락률이 너무 낮으면 스팟 라운드 마진을 소폭 완화해
            # 생성이 멈추지 않도록 함.
            if difficulty == "easy" and retries % 40 == 0:
                accepted_rate = diff_success / retries
                if accepted_rate < 0.15:
                    easy_margin_min = max(8, easy_margin_min - 1)
            # easy 는 PER-PUZZLE 믹스 — EASY_CONSTRUCTIVE_FRACTION 만큼은
            # constructive dominant-category dice(자명, ~0.99), 나머지는 random
            # band-filtered dice(실제 최적화, ~0.30). 평균이 목표 75 에 안착
            # (constructive 단독=0.99 너무 쉽고, random 단독=0.30 너무 어려움).
            if difficulty == "easy" and random.Random(424242 + problem_id).random() < EASY_CONSTRUCTIVE_FRACTION:
                crng = random.Random(1000 + problem_id + 10000)
                k_spot = SPOTCHECK_K.get("easy", 0)
                public_id_pre = f"yacht_dice_ko_easy_{diff_success:04d}"
                spot_rounds_pre = [
                    r - 1 for r in deterministic_round_pick_1based(public_id_pre, k_spot)
                ]
                forced = _build_easy_constructive_dice(
                    spot_rounds_pre, _pick_easy_spot_cats(len(spot_rounds_pre), crng), crng,
                )
                problem = generator.generate_problem("easy", problem_id, forced_dice=forced)
            else:
                problem = generator.generate_problem(
                    difficulty, problem_id, max_retries=inner_retries
                )
            is_valid, _ = generator.validate_problem(problem)
            if not is_valid:
                problem_id += 1
                continue

            sm = problem.get("step_metrics", {})
            # easy/medium 만 band 밖 주사위 거부. hard 는 면제: 빡빡한 band
            # (gap>=30, complexity>=8) 를 random dice 가 거의 못 맞춰 거부 시
            # 생성이 몇 시간씩 멈춤. K=9 가 이미 hard 난이도를 제공하므로
            # best-effort dice 를 수락.
            if difficulty != "hard" and sm.get("band_violation"):
                problem_id += 1
                continue
            # easy 의 constructive spot-독점성 검사 제거 — easy 도 이제
            # medium/hard 처럼 random·band-filtered dice 를 사용 (이 검사들은
            # 심어둔 dominant-category dice 에서만 의미가 있었음).

            key = _dice_key(problem['dice_results'])
            if key in seen_keys:
                problem_id += 1
                continue
            seen_keys.add(key)

            dice_results = problem['dice_results']
            optimal_score = problem['answer']
            optimal_assignment = problem.get('optimal_assignment', {})
            public_id = f"yacht_dice_ko_{difficulty}_{diff_success:04d}"

            k_spot = SPOTCHECK_K.get(difficulty, 0)
            spot_sum = spotcheck_grading_total(
                difficulty,
                public_id,
                dice_results,
                optimal_assignment,
                lambda d, c: calculate_score_with_config(d, c, config),
            )
            if spot_sum is not None:
                answer_str = str(spot_sum)
                spot_rounds = deterministic_round_pick_1based(public_id, k_spot)
            else:
                answer_str = str(optimal_score)
                spot_rounds = None

            solution_str = _build_yacht_solution_ko(
                dice_results=dice_results,
                optimal_assignment=optimal_assignment,
                optimal_score=optimal_score,
                config=config,
                difficulty=difficulty,
                step_metrics=sm,
                grading_answer=answer_str,
                spot_rounds_1based=spot_rounds,
            )

            base_question = config.get_user_prompt(dice_results)
            question = append_spotcheck_user_suffix(
                base_question, difficulty, public_id
            )

            puzzle_data = {
                "id": public_id,
                "question": question,
                "answer": answer_str,
                "solution": solution_str,
                "difficulty": difficulty,
            }
            all_puzzles.append(puzzle_data)
            if difficulty == "easy":
                easy_value_counts[int(answer_str)] = easy_value_counts.get(int(answer_str), 0) + 1
            diff_success += 1
            print(
                f"  [{diff_success}/{count}] 최적총점={optimal_score} "
                f"정답={answer_str} "
                f"gap={sm.get('greedy_gap', '?')} "
                f"complexity={sm.get('total_decision_complexity', 0):.2f} "
                f"band_violation={sm.get('band_violation', '?')}"
                + (
                    f" easy_margin={easy_margin_min}"
                    if difficulty == "easy" else ""
                )
            )
            problem_id += 1

        if diff_success < count:
            print(f"  ⚠️ {difficulty}: {diff_success}/{count}개만 생성됨")
        if difficulty == "easy" and diff_success > 0:
            print(
                "  easy 정답 분포: "
                + ", ".join(f"{k}={v}" for k, v in sorted(easy_value_counts.items()))
            )

    print(f"\n총 {len(all_puzzles)}개의 퍼즐이 생성되었습니다")

    df = pd.DataFrame(all_puzzles, columns=list(DATASET_COLUMNS))
    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("yacht_dice_ko", all_puzzles)
    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="요트 다이스 퍼즐 생성기 (한국어)")
    parser.add_argument(
        "--num",
        type=int,
        default=12,
        help="생성할 문제 수 (--difficulty 지정 시 해당 티어 수; 미지정 시 easy/medium/hard 에 분배)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        nargs="+",
        default=None,
        help="생성할 티어 지정, 예: --difficulty easy",
    )
    parser.add_argument("--workers", type=int, default=0, help="미사용; 셸 호환성을 위해 예약")
    args = parser.parse_args()

    print("=" * 60)
    print("요트 다이스 퍼즐 생성기 (한국어)")
    print("=" * 60)
    create_dataset_files(num_questions=args.num, difficulties=args.difficulty)

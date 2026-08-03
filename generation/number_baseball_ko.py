"""숫자 야구(Bulls and Cows) 퍼즐 생성기 - 한국어 버전

구성적 생성 방식: 정보 가치가 높은 힌트를 선택하여
해를 점진적으로 정확히 1개로 좁혀가는 퍼즐을 구축합니다.

logical-puzzles-me/number_baseball/generator.py 기반 이식:
- 비밀 숫자의 순열을 포함한 후보 힌트 풀 (볼 중심 힌트 생성)
- 중/상 난이도를 위한 2단계 전방 탐색(2-step lookahead) 스코어링
- 상 난이도 전용 볼 중심 체인 전략
- 모든 난이도에서 엄격한 유일 해(MAX_SOLUTIONS = 1) 보장
- 퍼즐 JSONL 에 step_metrics 필드 포함
"""

import itertools
import math
import random
import statistics
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import permutations
from dataclasses import dataclass
from enum import Enum


MAX_SOLUTIONS = 1  # 모든 난이도에서 정확히 1개의 해만 허용


# 모듈 레벨 캐시: num_digits 별 후보 공간 문자열 목록.
# 6-digit 공간 = 151200개 문자열; retry마다 재구축하던 핫루프.
# 반환 list는 공유됨 (caller가 변경하면 안 됨); 현재 caller는 list comprehension으로
# 새 필터링 list에 재바인딩하므로 안전함.
_CANDIDATE_SPACE_CACHE: Dict[int, List[str]] = {}


def _get_candidate_space(num_digits: int) -> List[str]:
    cached = _CANDIDATE_SPACE_CACHE.get(num_digits)
    if cached is None:
        cached = [''.join(p) for p in itertools.permutations('0123456789', num_digits)]
        _CANDIDATE_SPACE_CACHE[num_digits] = cached
    return cached


# 10-bit 자릿수 마스크(0..1023)용 팝카운트 테이블. bit i가 set이면 digit i 존재.
_POPCOUNT = [bin(i).count('1') for i in range(1024)]

# 모듈 레벨 캐시: num_digits 별 후보마다 (string, digit_mask) 쌍.
# 마스크로 매 호출마다 set(secret)을 재구축하지 않고 balls 계산 가능:
# distinct-digit 문자열에서 balls = popcount(mask_s & mask_guess) - strikes.
_CANDIDATE_MASK_CACHE: Dict[int, List[Tuple[str, int]]] = {}


def _digit_mask(s: str) -> int:
    mask = 0
    for ch in s:
        mask |= 1 << (ord(ch) - 48)
    return mask


def _get_candidate_masks(num_digits: int) -> List[Tuple[str, int]]:
    cached = _CANDIDATE_MASK_CACHE.get(num_digits)
    if cached is None:
        cached = [(s, _digit_mask(s)) for s in _get_candidate_space(num_digits)]
        _CANDIDATE_MASK_CACHE[num_digits] = cached
    return cached


# 모듈 레벨 캐시: 전체 공간에 대한 잔여 개수, (strikes, balls) 키, num_digits 별.
# 자릿수/위치 대칭성에 따라 distinct-digit guess에 대해 주어진 (S,B)를
# 산출하는 비밀 숫자 수는 특정 guess와 무관하므로, step 0(current == 전체 공간)에서는
# 힌트별 스캔이 필요 없음.
_RESIDUAL_COUNT_CACHE: Dict[int, Dict[Tuple[int, int], int]] = {}


def _get_residual_counts(num_digits: int) -> Dict[Tuple[int, int], int]:
    cached = _RESIDUAL_COUNT_CACHE.get(num_digits)
    if cached is None:
        masks = _get_candidate_masks(num_digits)
        ref, ref_mask = masks[0]
        counts: Dict[Tuple[int, int], int] = {}
        for s, m in masks:
            st = sum(1 for i in range(num_digits) if s[i] == ref[i])
            key = (st, _POPCOUNT[m & ref_mask] - st)
            counts[key] = counts.get(key, 0) + 1
        _RESIDUAL_COUNT_CACHE[num_digits] = counts
        cached = counts
    return cached


def _filter_by_hint(
    current: List[Tuple[str, int]], guess: str, want_s: int, want_b: int
) -> List[Tuple[str, int]]:
    """힌트의 S/B와 정확히 일치하는 (string, mask) 후보 필터링.

    calculate_strikes_balls(s, guess) == (want_s, want_b)인 s만 남기는 것과 동등하지만,
    사전 계산된 digit mask를 사용해 per-candidate set() 할당을 피함.
    5-6 자리 생성의 핫루프."""
    gmask = _digit_mask(guess)
    n = len(guess)
    out = []
    for s, m in current:
        st = 0
        for i in range(n):
            if s[i] == guess[i]:
                st += 1
        if st == want_s and _POPCOUNT[m & gmask] - st == want_b:
            out.append((s, m))
    return out


@dataclass
class Hint:
    guess: str
    strikes: int
    balls: int

    def __str__(self):
        return f"{self.guess}: {self.strikes}S {self.balls}B"

    def to_dict(self):
        return {"guess": self.guess, "strikes": self.strikes, "balls": self.balls}


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    # 난이도는 주로 num_digits가 결정한다 (gemini-3-flash-preview no-reveal
    # 정확도: 5D~96%, 6D~89%, 7D~62%, 8D~25-35%). 자릿수를 바꾸지 않고 티어를
    # 다시 완화하는 두 개의 부분 공개 손잡이:
    #   revealed_digits (j): 비밀 숫자의 j개 자릿수를 명시(위치는 미상) — 약
    #     7자리까지는 유효한 레버.
    #   revealed_positions: (위치 -> 자릿수) 쌍을 직접 고정. 8D에서는
    #     revealed_digits가 더 이상 정확도를 움직이지 못한다(병목이 배열
    #     자체로 이동 — 아래 "medium" 설정 참고), 그래서 대신 이걸 쓴다.
    #     float 값은 퍼즐당 기대 공개 위치 수(정수부는 항상 공개, 소수부는
    #     그 확률로 공개)로, 티어 평균이 정수 단계 사이에서 매끄럽게 잡힌다.
    #
    # 생성은 정보 탐욕(_select_info_greedy): 후보 집합을 가장 많이 줄이는 힌트를
    # 선택, 1개 후보가 남을 때까지. 100% 신뢰 가능.
    # preferred_strikes/balls는 힌트 풀 프로파일을 제한 (>=1 strike 허용해야
    # 탐욕 선택이 마지막 위치 모호성을 해결 가능).
    "easy": {
        "num_digits": 7,
        "min_hints": 4,
        "max_hints": 8,
        "revealed_digits": 5,
        "preferred_strikes": (0, 2),
        "preferred_balls": (0, 5),
        "pool_size": 50,
    },
    "medium": {
        "num_digits": 8,
        "min_hints": 4,
        "max_hints": 9,
        # 측정된 8D position-reveal 맵: p0=0.28, p1=0.65, p2=0.84 정확도
        # (~+28%p/위치); 0.6은 p0과 p1 사이라 목표 ~50%에 맞음
        "revealed_digits": 0,
        "revealed_positions": 0.6,
        "preferred_strikes": (0, 2),
        "preferred_balls": (0, 6),
        "pool_size": 50,
    },
    "hard": {
        "num_digits": 8,
        "min_hints": 4,
        "max_hints": 9,
        "revealed_digits": 0,
        "preferred_strikes": (0, 2),
        "preferred_balls": (0, 6),
        "pool_size": 50,
    },
}


class BullsAndCows:
    def __init__(self, num_digits: int = 3):
        if num_digits not in [3, 4, 5, 6, 7, 8]:
            raise ValueError("자릿수는 3, 4, 5, 6, 7, 8 중 하나여야 합니다")
        self.num_digits = num_digits

    def generate_number(self) -> str:
        digits = list(range(10))
        random.shuffle(digits)
        return ''.join(str(d) for d in digits[:self.num_digits])

    def calculate_strikes_balls(self, secret: str, guess: str) -> Tuple[int, int]:
        if len(secret) != len(guess):
            raise ValueError("비밀 숫자와 추측의 자릿수가 같아야 합니다")
        # set 조회는 str.__contains__의 O(n) 대비 "in" 검사당 O(1).
        # 핫루프: 퍼즐 생성 retry마다 O(num_digits!)회 호출됨.
        secret_set = set(secret)
        strikes = 0
        balls = 0
        for i, digit in enumerate(guess):
            if digit == secret[i]:
                strikes += 1
            elif digit in secret_set:
                balls += 1
        return strikes, balls

    def check_number_against_hints(self, number: str, hints: List[Hint]) -> bool:
        for hint in hints:
            s, b = self.calculate_strikes_balls(number, hint.guess)
            if s != hint.strikes or b != hint.balls:
                return False
        return True

    def find_all_solutions(self, hints: List[Hint], max_count: int = 0) -> List[str]:
        solutions = []
        for perm in permutations('0123456789', self.num_digits):
            number = ''.join(perm)
            if self.check_number_against_hints(number, hints):
                solutions.append(number)
                if max_count > 0 and len(solutions) >= max_count:
                    break
        return solutions

    def has_unique_solution(self, hints: List[Hint]) -> bool:
        solutions = self.find_all_solutions(hints, max_count=2)
        return len(solutions) == 1

    def generate_hint(self, secret: str, max_attempts: int = 100) -> Optional[Hint]:
        attempts = 0
        while attempts < max_attempts:
            guess = self.generate_number()
            if guess != secret:
                s, b = self.calculate_strikes_balls(secret, guess)
                return Hint(guess, s, b)
            attempts += 1
        return None


class ProblemGenerator:
    """숫자 야구의 구성적 퍼즐 생성기."""

    def __init__(self):
        self.game_3digit = BullsAndCows(3)
        self.game_4digit = BullsAndCows(4)
        self.game_5digit = BullsAndCows(5)
        self.game_6digit = BullsAndCows(6)
        self.game_7digit = BullsAndCows(7)
        self.game_8digit = BullsAndCows(8)
        self._games = {
            3: self.game_3digit, 4: self.game_4digit, 5: self.game_5digit,
            6: self.game_6digit, 7: self.game_7digit, 8: self.game_8digit,
        }

    def _is_duplicate_hint(self, hint: Hint, hints: List[Hint]) -> bool:
        for h in hints:
            if h.guess == hint.guess and h.strikes == hint.strikes and h.balls == hint.balls:
                return True
        return False

    def _hint_matches_difficulty(self, hint: Hint, difficulty: Difficulty) -> bool:
        cfg = DIFFICULTY_CONFIGS[difficulty.name.lower()]
        slo, shi = cfg["preferred_strikes"]
        blo, bhi = cfg["preferred_balls"]
        if not (slo <= hint.strikes <= shi):
            return False
        if not (blo <= hint.balls <= bhi):
            return False
        return True

    def _build_candidate_pool(
        self,
        game: BullsAndCows,
        secret: str,
        difficulty: Difficulty,
        target_size: int = 80,
        enrich_perms: bool = True,
    ) -> List[Hint]:
        """S/B가 포함된 distinct-digit 추측으로 후보 힌트 풀 구성.

        ``enrich_perms``가 True이면 비밀 숫자 자릿수의 순열(full-overlap 힌트,
        strikes+balls == num_digits)로 풀을 시딩 — 레거시 볼 중심 전략에 유용.
        정보 탐욕 경로는 False로 설정: 그런 전체 순열 힌트는 digit set 전체를
        무료로 공개하므로, 비밀 숫자와 더 적은 자릿수를 공유하는 무작위 추측만
        사용(현실적인 Bulls-and-Cows 힌트)."""
        pool: Dict[tuple, Hint] = {}

        if enrich_perms and difficulty != Difficulty.EASY:
            perms = list(itertools.permutations(secret))
            random.shuffle(perms)
            for perm in perms:
                guess = ''.join(perm)
                if guess == secret:
                    continue
                s, b = game.calculate_strikes_balls(secret, guess)
                hint = Hint(guess, s, b)
                if self._hint_matches_difficulty(hint, difficulty):
                    pool[(hint.guess, hint.strikes, hint.balls)] = hint
                if len(pool) >= target_size:
                    break

        attempts = 0
        while len(pool) < target_size and attempts < target_size * 20:
            attempts += 1
            hint = game.generate_hint(secret)
            if hint and self._hint_matches_difficulty(hint, difficulty):
                pool[(hint.guess, hint.strikes, hint.balls)] = hint

        hints = list(pool.values())
        if difficulty == Difficulty.HARD:
            hints.sort(key=lambda h: (h.balls, -h.strikes), reverse=True)
        elif difficulty == Difficulty.MEDIUM:
            hints.sort(key=lambda h: (h.balls, -h.strikes), reverse=True)
        else:
            hints.sort(key=lambda h: (h.strikes, -h.balls), reverse=True)
        return hints[:target_size]

    def _project_two_step_residual(
        self,
        game: BullsAndCows,
        existing_hints: List[Hint],
        current_solutions: List[str],
        candidate: Hint,
        difficulty: Difficulty,
        candidates: List[Hint],
        max_followups: int = 12,
    ) -> int:
        """2단계 전방 탐색: `candidate` 적용 후 다음 힌트로 가능한 최적 잔여를 찾음.
        중/상 난이도에서 사용."""
        best = None
        base_hints = existing_hints + [candidate]
        base_candidates = [
            s for s in current_solutions
            if game.calculate_strikes_balls(s, candidate.guess) == (candidate.strikes, candidate.balls)
        ]
        followups = candidates[:max_followups]
        for nxt in followups:
            if nxt.guess == candidate.guess:
                continue
            if self._is_duplicate_hint(nxt, base_hints):
                continue
            if not self._hint_matches_difficulty(nxt, difficulty):
                continue
            residual = sum(
                1 for s in base_candidates
                if game.calculate_strikes_balls(s, nxt.guess) == (nxt.strikes, nxt.balls)
            )
            if residual < 1:
                continue
            if best is None or residual < best:
                best = residual
                if residual == 1:
                    break
        return best if best is not None else 10**9

    def _select_best_hint(
        self,
        game: BullsAndCows,
        secret: str,
        existing_hints: List[Hint],
        current_solutions: List[str],
        difficulty: Difficulty,
        candidates: List[Hint],
        cfg: Dict[str, int],
    ) -> Optional[Hint]:
        """난이도 프로파일에 따른 최적 다음 힌트 선택."""
        best_hint = None
        best_score = None
        next_index = len(existing_hints) + 1
        min_hints = cfg["min_hints"]
        max_hints = cfg["max_hints"]
        target_lo, target_hi = cfg["target_residual"]

        for hint in candidates:
            if self._is_duplicate_hint(hint, existing_hints):
                continue
            if not self._hint_matches_difficulty(hint, difficulty):
                continue

            residual_candidates = [
                s for s in current_solutions
                if game.calculate_strikes_balls(s, hint.guess) == (hint.strikes, hint.balls)
            ]
            residual = len(residual_candidates)
            if residual < 1:
                continue

            if next_index >= min_hints and residual == 1:
                return hint

            lookahead = self._project_two_step_residual(
                game,
                existing_hints,
                current_solutions,
                hint,
                difficulty,
                candidates,
                max_followups=8 if difficulty == Difficulty.MEDIUM else 10,
            ) if difficulty != Difficulty.EASY else residual

            if difficulty == Difficulty.EASY:
                score = (
                    residual == 1,
                    -residual,
                    hint.strikes,
                    -hint.balls,
                )
            elif difficulty == Difficulty.MEDIUM:
                in_band = target_lo <= residual <= target_hi
                score = (
                    in_band,
                    -(residual == 1),
                    -abs(residual - (target_lo + target_hi) / 2),
                    -(lookahead == 10**9),
                    -abs(lookahead - max(1, target_lo // 2)),
                    hint.balls,
                    -hint.strikes,
                )
            else:
                if next_index < max_hints and residual == 1:
                    continue
                in_band = target_lo <= residual <= target_hi
                ball_heavy = hint.balls >= 2
                low_strike = hint.strikes <= 1
                score = (
                    low_strike,
                    ball_heavy,
                    in_band,
                    -(residual == 1),
                    -abs(residual - (target_lo + target_hi) / 2),
                    -(lookahead == 10**9),
                    -abs(lookahead - max(1, target_lo // 2)),
                    hint.balls,
                    -hint.strikes,
                )

            if best_score is None or score > best_score:
                best_score = score
                best_hint = hint

        return best_hint

    def _select_info_greedy(
        self,
        game: BullsAndCows,
        cfg: Dict[str, int],
        candidate_masks: List[Tuple[str, int]],
        hint_pool: List[Hint],
    ) -> Optional[Tuple[List[Hint], List[int]]]:
        """정보 탐욕 힌트 선택: 매 단계 후보 집합을 가장 많이 줄이는 풀 힌트를 선택,
        정확히 1개의 후보(비밀 숫자)가 남을 때까지 반복.

        볼 중심 "모호성 유지" 전략과 달리 항상 유일 해로 수렴(각 단계가 집합을 엄격히
        줄이고 최소 1개의 스트라이크 힌트로 마지막 위치 모호성을 해결 가능), 5-7 자리
        공간에서 생성 성공률 ~100%.

        max_hints 내에 유일성에 도달할 수 없으면 (hints, residuals) 또는 None 반환
        (드묾; caller가 새 비밀/풀로 재시도)."""
        if not hint_pool:
            return None

        min_hints = cfg["min_hints"]
        max_hints = cfg["max_hints"]
        full_counts = _get_residual_counts(len(candidate_masks[0][0]))

        current = list(candidate_masks)
        hints: List[Hint] = []
        residuals: List[int] = []

        while len(hints) < max_hints:
            # Step 0: current가 전체 공간이므로 잔여는 (S,B)에만 의존
            # -> 조회로 대체(무료); 선택된 힌트에 대해서만
            # 한 번 스캔.
            full_scan = not hints
            best = None
            best_filtered = None
            best_residual = None
            for hint in hint_pool:
                if self._is_duplicate_hint(hint, hints):
                    continue
                if full_scan:
                    residual = full_counts.get((hint.strikes, hint.balls), 0)
                    filtered = None
                else:
                    filtered = _filter_by_hint(
                        current, hint.guess, hint.strikes, hint.balls
                    )
                    residual = len(filtered)
                if residual < 1:
                    continue
                # 최소 힌트 예산이 채워지기 전에는 해결하지 않음.
                if residual == 1 and len(hints) + 1 < min_hints:
                    continue
                if best_residual is None or residual < best_residual:
                    best = hint
                    best_filtered = filtered
                    best_residual = residual

            if best is None:
                break

            if best_filtered is None:
                best_filtered = _filter_by_hint(
                    current, best.guess, best.strikes, best.balls
                )

            hints.append(best)
            current = best_filtered
            residuals.append(len(current))

            if len(current) == 1 and len(hints) >= min_hints:
                return hints, residuals

        if len(current) == 1 and len(hints) >= min_hints:
            return hints, residuals
        return None

    def generate_problem(self, difficulty: Difficulty, max_retries: int = 4000) -> Dict:
        """정확히 1개의 해를 가진 퍼즐을 구성적으로 생성합니다."""
        cfg = DIFFICULTY_CONFIGS[difficulty.name.lower()]
        num_digits = cfg["num_digits"]
        game = self._games.get(num_digits, self.game_6digit)

        min_hints = {difficulty: cfg["min_hints"]}
        max_hints = {difficulty: cfg["max_hints"]}

        for retry in range(max_retries):
            secret = game.generate_number()

            default_pool = 36 if difficulty == Difficulty.EASY else 28
            hint_pool = self._build_candidate_pool(
                game,
                secret,
                difficulty,
                target_size=cfg.get("pool_size", default_pool),
                # 정보 탐욕(num_digits>=5)은 full-permutation 힌트로 digit set이
                # 무료 노출되면 안 됨; 무작위 추측만 사용.
                enrich_perms=num_digits < 5,
            )

            # 캐시: retry마다 순열 list 재구축 방지.
            candidate_space = _get_candidate_space(num_digits)

            # num_digits >= 5: 정보 탐욕 선택 (5-7 자리 공간에서 100% 신뢰 가능).
            # 더 작은 자릿수는 레거시 2단계 전방 탐색 경로
            # (_select_best_hint)로 폴백.
            if num_digits >= 5:
                structured = self._select_info_greedy(
                    game, cfg, _get_candidate_masks(num_digits), hint_pool
                )
                if structured is None:
                    continue
                hints, residuals = structured
                solutions = [secret]
            else:
                hints = []
                # 이후 filter 재바인딩이 캐시에 영향 주지 않도록 복사.
                # (이후 list comprehension이 새 list를 만들지만, 초기 바인딩이
                # 캐시를 alias하면 문제가 됨.)
                solutions = list(candidate_space)
                while len(hints) < max_hints[difficulty]:
                    if len(solutions) == 1 and len(hints) >= min_hints[difficulty]:
                        break

                    best_hint = self._select_best_hint(
                        game, secret, hints, solutions,
                        difficulty, hint_pool, cfg,
                    )

                    if best_hint:
                        hints.append(best_hint)
                        hint_pool.remove(best_hint)
                        solutions = [
                            s for s in solutions
                            if game.calculate_strikes_balls(s, best_hint.guess) == (best_hint.strikes, best_hint.balls)
                        ]
                    else:
                        replenished = [
                            h for h in self._build_candidate_pool(game, secret, difficulty, target_size=24)
                            if not self._is_duplicate_hint(h, hints)
                        ]
                        if replenished:
                            hint_pool.extend(replenished)
                        else:
                            break

                solutions = solutions if hints else [secret]
                residuals = None

            if len(solutions) == 1 and len(hints) >= min_hints[difficulty]:
                # initial_candidates는 순열 개수뿐; 성공마다 (최대 151200 요소) list를
                # 재구축할 필요 없음.
                initial_candidates = len(_get_candidate_space(num_digits))
                if residuals is None:
                    cur = _get_candidate_masks(num_digits)
                    residuals = []
                    for h in hints:
                        cur = _filter_by_hint(cur, h.guess, h.strikes, h.balls)
                        residuals.append(len(cur))
                    # 노이즈 추가가 유일성을 깨면 안 되지만, 예상치 못한 순서
                    # 효과에 대비한 가드.
                    if residuals[-1] != 1:
                        continue

                prev = initial_candidates
                per_hint_bits = []
                for r in residuals:
                    if r <= 0 or prev <= 0:
                        per_hint_bits.append(0.0)
                    else:
                        per_hint_bits.append(math.log2(prev / r))
                    prev = r
                total_deduction_bits = math.log2(initial_candidates) if initial_candidates > 0 else 0.0
                min_per_hint_bits = min(per_hint_bits) if per_hint_bits else 0.0
                ball_heavy_ratio = (
                    sum(1 for h in hints if h.balls >= 2 and h.strikes <= 1) / len(hints)
                    if hints else 0.0
                )
                late_resolution_index = next(
                    (i + 1 for i, r in enumerate(residuals) if r == 1),
                    len(residuals),
                )
                residual_drop_variance = statistics.pvariance(per_hint_bits) if len(per_hint_bits) > 1 else 0.0

                if ball_heavy_ratio < cfg.get("min_ball_heavy_ratio", 0.0):
                    continue

                # 난이도 손잡이: 정답의 j개 자릿수를 명시적으로 공개(위치 미상).
                # digit set을 명시하는 것이 모델에게 실제로 퍼즐을 쉽게 만듦 —
                # DIFFICULTY_CONFIGS 주석 참고.
                answer = solutions[0]
                reveal_n = cfg.get("revealed_digits", 0)
                revealed = sorted(random.sample(answer, reveal_n)) if reveal_n else []

                # 위치 공개: (위치 -> 자릿수) 쌍을 고정해 배열을 직접 완화
                # (digit-set 공개보다 강한 레버).
                # FLOAT = 기대 위치 수를 받음: 정수 부분은 항상
                # 공개, 소수 부분은 그 확률로. 이를 통해
                # 퍼즐별 MIX로 티어 평균이 정수 단계 사이에 오도록 함
                # (각 정수 단계는 8D에서 거친 ~+28%p).
                reveal_pos_cfg = cfg.get("revealed_positions", 0)
                base_pos = int(reveal_pos_cfg)
                reveal_pos_n = base_pos + (
                    1 if random.random() < (reveal_pos_cfg - base_pos) else 0
                )
                revealed_positions = []
                if reveal_pos_n:
                    idxs = sorted(random.sample(range(num_digits), reveal_pos_n))
                    revealed_positions = [[i, answer[i]] for i in idxs]

                return {
                    "difficulty": difficulty.name.lower(),
                    "num_digits": num_digits,
                    "hints": [hint.to_dict() for hint in hints],
                    "hint_text": self._format_hints(hints),
                    "answer": answer,
                    "revealed_digits": revealed,
                    "revealed_positions": revealed_positions,
                    "problem_text": self._create_problem_text(num_digits, hints),
                    "step_metrics": {
                        "initial_candidates": initial_candidates,
                        "residuals": residuals,
                        "per_hint_bits": per_hint_bits,
                        "total_deduction_bits": total_deduction_bits,
                        "min_per_hint_bits": min_per_hint_bits,
                        "hint_count": len(hints),
                        "ball_heavy_ratio": ball_heavy_ratio,
                        "late_resolution_index": late_resolution_index,
                        "residual_drop_variance": residual_drop_variance,
                    },
                }

        raise RuntimeError(
            f"{max_retries}번 재시도 후에도 정확히 1개의 해를 가진 "
            f"{difficulty.name} 난이도 퍼즐을 생성하지 못했습니다"
        )

    def _format_hints(self, hints: List[Hint]) -> List[str]:
        return [str(hint) for hint in hints]

    def _create_problem_text(self, num_digits: int, hints: List[Hint]) -> str:
        hint_strs = [f"[{hint.guess}: {hint.strikes}S {hint.balls}B]" for hint in hints]
        hints_text = ", ".join(hint_strs)
        return (
            f"다음 모든 힌트를 만족하는, 각 자릿수가 서로 다른 "
            f"{num_digits}자리 숫자를 찾으세요: {hints_text}"
        )


SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_baseball_solution_ko(problem: Dict) -> str:
    """SFT teacher trace: 숫자 야구 · 힌트별 후보 축소 SEG."""
    num_digits = problem['num_digits']
    hints = problem['hints']
    answer = problem['answer']
    metrics = problem.get('step_metrics', {})
    initial = metrics.get('initial_candidates', 0)
    residuals = metrics.get('residuals', [])
    per_bits = metrics.get('per_hint_bits', [])

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {problem.get('difficulty', '')}",
        f"  - 자릿수: {num_digits} (서로 다른 숫자)",
        f"  - 힌트 수: {len(hints)} · 초기 후보: {initial}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 규칙: 각 자리 숫자는 서로 다름(0–9).",
        "  - S(스트라이크) = 숫자·위치 모두 일치, B(볼) = 숫자만 일치.",
    ]
    for i, h in enumerate(hints, 1):
        lines.append(
            f"  {i}. 추측 {h['guess']} → {h['strikes']}S {h['balls']}B"
        )

    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: 각 힌트(S/B)로 후보 공간 축소 · 초기 {initial} → "
        f"최종 1 · SEG {len(hints)}개"
    )
    prev = initial
    for i, h in enumerate(hints, 1):
        resid = residuals[i - 1] if i - 1 < len(residuals) else None
        bits = per_bits[i - 1] if i - 1 < len(per_bits) else None
        info_parts = []
        if resid is not None:
            info_parts.append(f"후보 {prev}→{resid}")
            prev = resid
        if bits is not None:
            info_parts.append(f"정보량 {bits:.2f} bits")
        info_text = " · ".join(info_parts) if info_parts else ""
        lines.append(
            f"    [SEG {i}] 힌트 {i} 반영: {h['guess']} → {h['strikes']}S {h['balls']}B · "
            f"{info_text}"
        )

    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답: {answer}",
        "  - 각 힌트에 대해 정답과 S/B를 재계산하여 모두 일치하는지 확인.",
    ])
    return "\n".join(lines)


# ============================================================
# 질문 포맷팅
# ============================================================

def create_question(problem: Dict) -> str:
    num_digits = problem['num_digits']
    hints = problem['hints']

    hints_text = "\n".join([
        f"  {i+1}. 추측: {h['guess']} -> {h['strikes']} 스트라이크(S), {h['balls']} 볼(B)"
        for i, h in enumerate(hints)
    ])

    # 선택적 난이도 보조: 비밀 숫자의 일부 자릿수를 미리 공개
    # (위치 미상), 나머지는 힌트로 추론.
    revealed = problem.get('revealed_digits') or []
    known_block = ""
    if revealed:
        known_list = ", ".join(str(d) for d in revealed)
        if len(revealed) >= num_digits:
            known_block = (
                f"\n알려진 정보: 비밀 숫자는 다음 {num_digits}개 자릿수의 배열입니다 "
                f"(힌트로 순서를 추론하세요):\n  {known_list}\n"
            )
        elif len(revealed) == 1:
            known_block = (
                f"\n알려진 자릿수 (이 숫자가 비밀 숫자에 포함되지만 위치는 모름; "
                f"나머지 {num_digits - 1}개는 스스로 추론):\n  {known_list}\n"
            )
        else:
            known_block = (
                f"\n알려진 자릿수 (이 {len(revealed)}개 숫자가 비밀 숫자에 포함되지만 "
                f"위치는 모름; 나머지 {num_digits - len(revealed)}개는 스스로 추론):\n"
                f"  {known_list}\n"
            )

    # 선택적 난이도 보조: 일부 (위치 -> 자릿수) 쌍을 직접 고정하여
    # 나머지 위치만 힌트로 풀도록 함.
    revealed_positions = problem.get('revealed_positions') or []
    if revealed_positions:
        pos_list = ", ".join(
            f"{int(i) + 1}번째 자리 = {d}" for i, d in revealed_positions
        )
        rest = num_digits - len(revealed_positions)
        known_block += (
            f"\n알려진 위치 (왼쪽부터 1번째 자리 기준으로 해당 자리의 숫자가 확정됨; "
            f"나머지 {rest}개 자리는 추론):\n  {pos_list}\n"
        )

    question = f"""다음 숫자 야구 퍼즐을 풀어보세요.

규칙:
- 비밀 숫자는 {num_digits}자리이며, 각 자릿수는 서로 다릅니다 (0-9)
- "스트라이크(S)"는 숫자가 맞고 위치도 맞음을 의미합니다
- "볼(B)"은 숫자는 맞지만 위치가 틀림을 의미합니다
- 모든 힌트를 만족하는 비밀 숫자를 찾으세요

힌트:
{hints_text}
{known_block}
단계별로 생각하며 유일한 {num_digits}자리 비밀 숫자를 찾으세요.

다음 형식으로 답을 제시하세요:
Answer: [{num_digits}자리 비밀 숫자]"""

    return question


def validate_problem(problem: Dict) -> Tuple[bool, str]:
    try:
        num_digits = problem['num_digits']
        game = BullsAndCows(num_digits)

        hints = [Hint(h['guess'], h['strikes'], h['balls']) for h in problem['hints']]

        answer = problem['answer']
        if len(answer) != num_digits:
            return False, f"정답 길이 {len(answer)}가 자릿수 {num_digits}와 일치하지 않습니다"

        if len(set(answer)) != num_digits:
            return False, f"정답 {answer}에 중복된 숫자가 있습니다"

        if not game.check_number_against_hints(answer, hints):
            return False, f"정답 {answer}이 모든 힌트를 만족하지 않습니다"

        revealed = problem.get('revealed_digits') or []
        if any(str(d) not in answer for d in revealed):
            return False, f"공개된 자릿수 {revealed}가 정답 {answer}에 포함되지 않습니다"

        revealed_positions = problem.get('revealed_positions') or []
        for i, d in revealed_positions:
            if not (0 <= int(i) < num_digits) or answer[int(i)] != str(d):
                return False, f"공개된 위치 {(i, d)}가 정답 {answer}와 일치하지 않습니다"

        # 마스크 기반 유일성: 후보 공간을 반복 필터링(각 힌트가 축소),
        # 힌트마다 모든 순열을 스캔하는 것보다 훨씬 빠름 —
        # 7-8 자리 공간(151K-1.8M 후보)에서 중요.
        cur = _get_candidate_masks(num_digits)
        for h in hints:
            cur = _filter_by_hint(cur, h.guess, h.strikes, h.balls)
            if len(cur) <= 1:
                break
        solutions = [s for s, _ in cur]
        if len(solutions) == 0:
            return False, "주어진 힌트를 만족하는 해가 존재하지 않습니다"
        elif len(solutions) > 1:
            return False, "여러 개의 해가 존재합니다"
        elif solutions[0] != answer:
            return False, f"해 {solutions[0]}가 정답 {answer}과 일치하지 않습니다"

        return True, "유일한 해를 가진 유효한 문제입니다"

    except Exception as e:
        return False, f"검증 오류: {str(e)}"


# ============================================================
# 데이터셋 생성
# ============================================================

def _apply_digit_permutation(problem: Dict, perm: Dict[str, str]) -> Dict:
    """비밀 숫자와 모든 추측에 자릿수 전단사(bijection) 적용.

    자릿수 순열은 Bulls-and-Cows 게임의 대칭성: perm이 '0'..'9'의 전단사이면
    비밀 숫자와 모든 추측의 각 자릿수를 perm[digit]으로 치환해도 동일한 S/B를
    갖는 새 유효 퍼즐이 생성됨."""
    import copy
    new = copy.deepcopy(problem)
    new['answer'] = ''.join(perm[d] for d in problem['answer'])
    new_hints = []
    for h in problem['hints']:
        nh = dict(h)
        nh['guess'] = ''.join(perm[d] for d in h['guess'])
        new_hints.append(nh)
    new['hints'] = new_hints
    if problem.get('revealed_digits'):
        new['revealed_digits'] = sorted(perm[str(d)] for d in problem['revealed_digits'])
    if problem.get('revealed_positions'):
        new['revealed_positions'] = [
            [i, perm[str(d)]] for i, d in problem['revealed_positions']
        ]
    if 'hint_text' in new:
        new.pop('hint_text', None)
    if 'problem_text' in new:
        new.pop('problem_text', None)
    return new


def create_dataset_files(num_questions: int, difficulties: Optional[List[str]] = None):
    """숫자 야구 데이터셋 파일 생성.

    생성 속도 (~20ms easy / ~120ms medium / ~600ms hard/퍼즐)가 빠르고 정보 탐욕
    선택으로 100% 신뢰 가능하므로, 다양성 극대화를 위해 모든 퍼즐을 새로 생성.
    자릿수 순열 파생은 해당 난이도의 신규 생성이 요청 수에 미달할 때만 보충용으로
    유지 (자릿수 순열은 Bulls-and-Cows 게임의 정확한 대칭: S/B 수 불변, 따라서
    순열 퍼즐은 추가 solve 호출 없이 유효).

    ``difficulties``는 생성할 티어를 제한 (예: ``["medium"]``으로 한 티어만 재조정).
    단일 티어이면 ``num_questions``가 해당 티어 개수; 아니면 티어들에 분배.
    합쳐진 JSONL에는 생성된 티어만 포함되므로, downstream split 단계는 해당
    ``*_<tier>.jsonl`` 파일만 덮어씀.
    """
    import pandas as pd

    # 전부 신규 생성: cap >= 현실적인 난이도별 개수이므로 bases_needed == count
    # (신규 생성이 부족할 때만 순열 파생).
    BASE_PER_DIFF = 100000

    _name_to_diff = {
        "easy": Difficulty.EASY, "medium": Difficulty.MEDIUM, "hard": Difficulty.HARD,
    }
    if difficulties is None:
        difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    else:
        difficulties = [_name_to_diff[d.lower()] for d in difficulties]

    print(f"{num_questions}개의 숫자 야구 퍼즐을 생성합니다...")

    generator = ProblemGenerator()

    # 단일 티어 -> num_questions가 해당 티어 개수; 아니면 티어들에 분배.
    if len(difficulties) == 1:
        puzzles_per_diff = num_questions
        remainder = 0
    else:
        puzzles_per_diff = num_questions // len(difficulties)
        remainder = num_questions % len(difficulties)

    all_puzzles = []
    MAX_RETRIES_PER_PUZZLE = 50

    def _hint_key(problem):
        hints = problem.get('hints', [])
        return (
            problem.get('num_digits'),
            tuple(sorted(
                (h.get('guess', ''), h.get('strikes', 0), h.get('balls', 0))
                for h in hints if isinstance(h, dict)
            )),
        )

    for di, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if di < remainder else 0)
        diff_name = difficulty.name.lower()

        if count == 0:
            continue

        print(f"\n=== {diff_name} 난이도 퍼즐 생성 중 ({count}개 필요) ===")

        # --- Phase 1: 기반 퍼즐 생성 ---
        bases_needed = min(BASE_PER_DIFF, count)
        # 내부 retry 예산을 줄여 generate_problem 호출이 빠르게 실패하도록;
        # 외부 루프에서 보상.
        base_inner_retries = 300  # 기본값 4000 대비
        base_problems: List[Dict] = []
        retries = 0
        max_retries_base = bases_needed * MAX_RETRIES_PER_PUZZLE
        while len(base_problems) < bases_needed and retries < max_retries_base:
            try:
                problem = generator.generate_problem(
                    difficulty, max_retries=base_inner_retries
                )
            except RuntimeError as e:
                retries += 1
                print(f"  [기반 시도 {retries}] 실패: {e}")
                continue
            is_valid, msg = validate_problem(problem)
            if not is_valid:
                retries += 1
                continue
            base_problems.append(problem)
            print(f"  [기반 {len(base_problems)}/{bases_needed}] 자릿수={problem['num_digits']}, "
                  f"힌트={len(problem['hints'])}, 정답={problem['answer']}")

        if not base_problems:
            print(f"  {diff_name}: 기반 퍼즐 생성 실패; 건너뜀")
            continue

        # --- Phase 2: 자릿수 순열로 나머지 파생 ---
        diff_success = 0
        seen_keys: set = set()

        for j in range(count):
            base = base_problems[j % len(base_problems)]
            if j < len(base_problems):
                problem = base
            else:
                rng = random.Random(999983 * di + 100003 * j)
                digits = list('0123456789')
                rng.shuffle(digits)
                perm = {str(i): digits[i] for i in range(10)}
                problem = _apply_digit_permutation(base, perm)

            key = _hint_key(problem)
            if key in seen_keys:
                rng2 = random.Random(777777 * di + 13 * j)
                digits2 = list('0123456789')
                rng2.shuffle(digits2)
                perm2 = {str(i): digits2[i] for i in range(10)}
                problem = _apply_digit_permutation(base_problems[0], perm2)
                key = _hint_key(problem)
            seen_keys.add(key)

            puzzle_data = {
                'id': f'number_baseball_ko_{diff_name}_{diff_success:04d}',
                'question': create_question(problem),
                'answer': problem['answer'],
                'solution': _build_baseball_solution_ko(problem),
                'difficulty': diff_name,
            }
            all_puzzles.append(puzzle_data)
            diff_success += 1
            print(f"  [{diff_success}/{count}] 자릿수={problem['num_digits']}, "
                  f"힌트={len(problem['hints'])}, 정답={problem['answer']}")

    print(f"\n총 {len(all_puzzles)}개의 퍼즐이 생성되었습니다")

    df = pd.DataFrame(all_puzzles)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("number_baseball_ko", all_puzzles)

    return df, all_puzzles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="숫자 야구 퍼즐 생성기 (한국어)")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수 (--difficulty 지정 시 해당 티어 수; 미지정 시 티어들에 분배)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], nargs="+", default=None, help="생성할 티어 지정, 예: --difficulty medium")
    parser.add_argument("--workers", type=int, default=0, help="호환성용; fast mode는 template bank 사용")

    args = parser.parse_args()

    print("=" * 60)
    print("숫자 야구(Bulls and Cows) 퍼즐 생성기 (한국어)")
    print("=" * 60)

    create_dataset_files(num_questions=args.num, difficulties=args.difficulty)

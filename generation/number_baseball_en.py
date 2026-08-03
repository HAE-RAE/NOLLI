"""Number Baseball (Bulls and Cows) Puzzle Generator (EN)

Constructive generation: builds puzzles by selecting high-information
hints that progressively narrow solutions to exactly 1.

Ported from logical-puzzles-me/number_baseball/generator.py:
- Permutation-based candidate pool for ball-heavy hints
- 2-step lookahead scoring for medium/hard
- Hard-specific ball-heavy chain strategy
- Strict uniqueness (MAX_SOLUTIONS = 1) for all difficulties
- step_metrics exported in puzzle JSONL
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


MAX_SOLUTIONS = 1  # Only allow exactly 1 solution for ALL difficulties


# Module-level cache: candidate space strings per num_digits.
# 6-digit space = 151200 strings; rebuilding each retry was a hot loop.
# Returned list is shared (callers must not mutate); they currently rebind to a
# new filtered list via list-comprehension, so this is safe.
_CANDIDATE_SPACE_CACHE: Dict[int, List[str]] = {}


def _get_candidate_space(num_digits: int) -> List[str]:
    cached = _CANDIDATE_SPACE_CACHE.get(num_digits)
    if cached is None:
        cached = [''.join(p) for p in itertools.permutations('0123456789', num_digits)]
        _CANDIDATE_SPACE_CACHE[num_digits] = cached
    return cached


# Popcount table for 10-bit digit masks (0..1023). bit i set => digit i present.
_POPCOUNT = [bin(i).count('1') for i in range(1024)]

# Module-level cache: (string, digit_mask) per candidate, per num_digits.
# The mask lets us compute balls without rebuilding set(secret) on every call:
# for distinct-digit strings, balls = popcount(mask_s & mask_guess) - strikes.
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


# Module-level cache: residual count over the FULL space, keyed by (strikes,
# balls), per num_digits. By digit/position symmetry the number of secrets that
# yield a given (S,B) against a distinct-digit guess is independent of the
# specific guess, so step 0 (current == full space) needs no per-hint scan.
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
    """Filter (string, mask) candidates matching a hint's S/B exactly.

    Equivalent to keeping s where calculate_strikes_balls(s, guess) ==
    (want_s, want_b), but avoids a per-candidate set() allocation by using a
    precomputed digit mask. Hot loop for 5-6 digit generation."""
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
    # Difficulty is dominated by num_digits (no-reveal accuracy on
    # gemini-3-flash-preview: 5D~96%, 6D~89%, 7D~62%, 8D~25-35%), with two
    # partial-reveal knobs that ease a tier back up without changing digit count:
    #   revealed_digits (j): state j of the secret's digits explicitly
    #     (position unknown) — the effective lever up to ~7 digits.
    #   revealed_positions: pin (position -> digit) pairs directly. On 8D,
    #     revealed_digits stops moving accuracy at all (the bottleneck shifts
    #     to the positional arrangement — see the "medium" config below), so
    #     this is used instead. A float value is the expected number of
    #     revealed positions per puzzle (integer part always revealed,
    #     fractional part probabilistically), giving a smooth per-tier average.
    #
    # Generation is information-greedy (_select_info_greedy): pick the hint that
    # shrinks the candidate set most, until one candidate remains. 100% reliable.
    # preferred_strikes/balls bound the hint-pool profile (>=1 strike must be
    # allowed so the greedy can resolve the final positional ambiguity).
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
        # measured 8D position-reveal map: p0=0.28, p1=0.65, p2=0.84 accuracy
        # (~+28%p/position); 0.6 lands between p0 and p1 to hit the ~50% target
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
            raise ValueError("Number of digits must be 3, 4, 5, 6, 7, or 8")
        self.num_digits = num_digits

    def generate_number(self) -> str:
        digits = list(range(10))
        random.shuffle(digits)
        return ''.join(str(d) for d in digits[:self.num_digits])

    def calculate_strikes_balls(self, secret: str, guess: str) -> Tuple[int, int]:
        if len(secret) != len(guess):
            raise ValueError("Secret and guess must have the same length")
        # Set lookup is O(1) per "in" check vs O(n) for str.__contains__.
        # Hot loop: invoked O(num_digits!) times per puzzle generate retry.
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
    """Constructive puzzle generator for Bulls and Cows."""

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
        """Build a candidate hint pool of distinct-digit guesses with their S/B.

        When ``enrich_perms`` is True the pool is seeded with permutations of
        the secret's digits (full-overlap hints, strikes+balls == num_digits) —
        useful for the legacy ball-heavy strategy. The information-greedy path
        sets it False: those full-permutation hints would reveal the entire
        digit set for free, so we use only random guesses, which naturally
        share fewer digits with the secret (a realistic Bulls-and-Cows hint)."""
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
        """2-step lookahead: after applying `candidate`, find the best possible
        residual from a next hint. Used for medium/hard."""
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
        """Select best next hint given difficulty profile."""
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
        """Information-greedy hint selection: at each step pick the pool hint
        that shrinks the candidate set the most, until exactly one candidate
        (the secret) remains.

        Unlike the ball-heavy "stay ambiguous" strategy, this always converges
        to a unique solution (each step strictly reduces the set and at least
        one strike-bearing hint can resolve the last positional ambiguity), so
        generation succeeds ~100% of the time across 5-7 digit spaces.

        Returns (hints, residuals) or None if the pool cannot reach uniqueness
        within max_hints (rare; caller retries with a fresh secret/pool)."""
        if not hint_pool:
            return None

        min_hints = cfg["min_hints"]
        max_hints = cfg["max_hints"]
        full_counts = _get_residual_counts(len(candidate_masks[0][0]))

        current = list(candidate_masks)
        hints: List[Hint] = []
        residuals: List[int] = []

        while len(hints) < max_hints:
            # Step 0: current is the full space, so residual depends only on
            # (S,B) -> look it up instead of scanning (free); scan once for the
            # chosen hint afterwards.
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
                # Don't resolve before the minimum hint budget is met.
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
        """Constructively generate a puzzle with exactly 1 solution."""
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
                # Info-greedy (num_digits>=5) must not reveal the digit set via
                # full-permutation hints; use random guesses only.
                enrich_perms=num_digits < 5,
            )

            # Cached: avoids rebuilding the permutation list per retry.
            candidate_space = _get_candidate_space(num_digits)

            # num_digits >= 5: information-greedy selection (100% reliable for
            # 5-7 digit spaces). Smaller digit counts fall through to the legacy
            # 2-step-lookahead path (_select_best_hint).
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
                # Copy so subsequent filter rebinds don't affect cache. (List
                # comprehensions later create new lists; initial bind would
                # otherwise alias the cache.)
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
                # initial_candidates is just the permutation count; no need to
                # rebuild the (up to 151200-element) list on every success.
                initial_candidates = len(_get_candidate_space(num_digits))
                if residuals is None:
                    cur = _get_candidate_masks(num_digits)
                    residuals = []
                    for h in hints:
                        cur = _filter_by_hint(cur, h.guess, h.strikes, h.balls)
                        residuals.append(len(cur))
                    # Adding noise should never break uniqueness, but guard
                    # against any unexpected order effect.
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

                # Difficulty knob: reveal j of the answer's digits explicitly
                # (positions unknown). Stating the digit set is what actually
                # eases the puzzle for the model — see DIFFICULTY_CONFIGS notes.
                answer = solutions[0]
                reveal_n = cfg.get("revealed_digits", 0)
                revealed = sorted(random.sample(answer, reveal_n)) if reveal_n else []

                # Position reveal: pin (position -> digit) pairs to ease the
                # arrangement directly (stronger lever than digit-set reveal).
                # Accepts a FLOAT = expected #positions: the integer part is always
                # revealed, the fractional part with that probability. This gives a
                # per-puzzle MIX so the tier average lands between integer steps
                # (each integer step is a coarse ~+28%p on 8D).
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
            f"Failed to generate {difficulty.name} puzzle with exactly 1 solution "
            f"after {max_retries} retries"
        )

    def _format_hints(self, hints: List[Hint]) -> List[str]:
        return [str(hint) for hint in hints]

    def _create_problem_text(self, num_digits: int, hints: List[Hint]) -> str:
        hint_strs = [f"[{hint.guess}: {hint.strikes}S {hint.balls}B]" for hint in hints]
        hints_text = ", ".join(hint_strs)
        return (
            f"Find the {num_digits}-digit number with distinct digits that satisfies "
            f"all the following hints: {hints_text}"
        )


SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


def _build_baseball_solution_en(problem: Dict) -> str:
    """SFT teacher trace: number baseball with per-hint SEG shrinkage."""
    num_digits = problem['num_digits']
    hints = problem['hints']
    answer = problem['answer']
    metrics = problem.get('step_metrics', {})
    initial = metrics.get('initial_candidates', 0)
    residuals = metrics.get('residuals', [])
    per_bits = metrics.get('per_hint_bits', [])

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {problem.get('difficulty', '')}",
        f"  - Digits: {num_digits} (all distinct)",
        f"  - Hints: {len(hints)} · initial candidates: {initial}",
        "  - Final answer is confirmed in [STEP 3]",
        "[STEP 1] Given",
        "  - Rule: each digit of the secret is distinct (0-9).",
        "  - S(trike) = right digit + right position; B(all) = right digit only.",
    ]
    for i, h in enumerate(hints, 1):
        lines.append(
            f"  {i}. guess {h['guess']} -> {h['strikes']}S {h['balls']}B"
        )

    lines.append("[STEP 2] Worked solution")
    lines.append(
        f"  · Summary: shrink the candidate set by each S/B hint · "
        f"{initial} -> 1 · {len(hints)} SEGs"
    )
    prev = initial
    for i, h in enumerate(hints, 1):
        resid = residuals[i - 1] if i - 1 < len(residuals) else None
        bits = per_bits[i - 1] if i - 1 < len(per_bits) else None
        info_parts = []
        if resid is not None:
            info_parts.append(f"candidates {prev}->{resid}")
            prev = resid
        if bits is not None:
            info_parts.append(f"info {bits:.2f} bits")
        info_text = " · ".join(info_parts) if info_parts else ""
        lines.append(
            f"    [SEG {i}] apply hint {i}: {h['guess']} -> {h['strikes']}S {h['balls']}B · "
            f"{info_text}"
        )

    lines.extend([
        "[STEP 3] Answer and verification",
        f"  - Final answer: {answer}",
        "  - Recompute S/B of each hint against the answer; all must match exactly.",
    ])
    return "\n".join(lines)


# ============================================================
# Question formatting
# ============================================================

def create_question(problem: Dict) -> str:
    num_digits = problem['num_digits']
    hints = problem['hints']

    hints_text = "\n".join([
        f"  {i+1}. Guess: {h['guess']} -> {h['strikes']} Strike(s), {h['balls']} Ball(s)"
        for i, h in enumerate(hints)
    ])

    # Optional difficulty aid: some of the secret's digits stated up front
    # (positions unknown), the rest must be deduced from the hints.
    revealed = problem.get('revealed_digits') or []
    known_block = ""
    if revealed:
        known_list = ", ".join(str(d) for d in revealed)
        if len(revealed) >= num_digits:
            known_block = (
                f"\nKnown: the secret is an arrangement of exactly these "
                f"{num_digits} digits (work out their order from the hints):\n"
                f"  {known_list}\n"
            )
        elif len(revealed) == 1:
            known_block = (
                f"\nKnown digit (this digit appears in the secret, position "
                f"unknown; the other {num_digits - 1} are for you to deduce):\n"
                f"  {known_list}\n"
            )
        else:
            known_block = (
                f"\nKnown digits (these {len(revealed)} digits appear in the secret, "
                f"positions unknown; the other {num_digits - len(revealed)} are for you "
                f"to deduce):\n  {known_list}\n"
            )

    # Optional difficulty aid: some (position -> digit) pairs pinned outright,
    # so only the remaining positions must be worked out from the hints.
    revealed_positions = problem.get('revealed_positions') or []
    if revealed_positions:
        pos_list = ", ".join(
            f"position {int(i) + 1} = {d}" for i, d in revealed_positions
        )
        rest = num_digits - len(revealed_positions)
        known_block += (
            f"\nKnown positions (these digits are fixed at the given positions, "
            f"counting from the left starting at 1; the other {rest} positions are "
            f"for you to deduce):\n  {pos_list}\n"
        )

    question = f"""Solve this Number Baseball (Bulls and Cows) puzzle.

Rules:
- The secret number has {num_digits} digits, each digit is unique (0-9)
- "Strike" means a digit is correct AND in the correct position
- "Ball" means a digit is correct BUT in the wrong position
- Your task: Find the secret number that satisfies ALL hints

Hints:
{hints_text}
{known_block}
Think step by step and find the unique {num_digits}-digit secret number.

Provide your answer in this format:
Answer: [the {num_digits}-digit secret number]"""

    return question


def validate_problem(problem: Dict) -> Tuple[bool, str]:
    try:
        num_digits = problem['num_digits']
        game = BullsAndCows(num_digits)

        hints = [Hint(h['guess'], h['strikes'], h['balls']) for h in problem['hints']]

        answer = problem['answer']
        if len(answer) != num_digits:
            return False, f"Answer length {len(answer)} doesn't match num_digits {num_digits}"

        if len(set(answer)) != num_digits:
            return False, f"Answer {answer} doesn't have unique digits"

        if not game.check_number_against_hints(answer, hints):
            return False, f"Answer {answer} doesn't satisfy all hints"

        revealed = problem.get('revealed_digits') or []
        if any(str(d) not in answer for d in revealed):
            return False, f"Revealed digits {revealed} not all in answer {answer}"

        revealed_positions = problem.get('revealed_positions') or []
        for i, d in revealed_positions:
            if not (0 <= int(i) < num_digits) or answer[int(i)] != str(d):
                return False, f"Revealed position {(i, d)} doesn't match answer {answer}"

        # Mask-based uniqueness: iteratively filter the candidate space (each
        # hint shrinks it), far faster than scanning all permutations per hint —
        # important for 7-8 digit spaces (151K-1.8M candidates).
        cur = _get_candidate_masks(num_digits)
        for h in hints:
            cur = _filter_by_hint(cur, h.guess, h.strikes, h.balls)
            if len(cur) <= 1:
                break
        solutions = [s for s, _ in cur]
        if len(solutions) == 0:
            return False, "No solution exists for the given hints"
        elif len(solutions) > 1:
            return False, f"Multiple solutions exist"
        elif solutions[0] != answer:
            return False, f"Solution {solutions[0]} doesn't match answer {answer}"

        return True, "Problem is valid with unique solution"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ============================================================
# Dataset generation
# ============================================================

def _apply_digit_permutation(problem: Dict, perm: Dict[str, str]) -> Dict:
    """Apply a digit bijection to a baseball problem's secret and all guesses.

    Digit permutation is a symmetry of the Bulls-and-Cows game: if perm is a
    bijection on '0'..'9', then replacing every digit in the secret and every
    guess with perm[digit] yields a new valid puzzle with the same S/B counts.
    """
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
    """Create number baseball dataset files.

    Generation is now fast (~20ms easy / ~120ms medium / ~600ms hard per
    puzzle) and 100% reliable via information-greedy selection, so every puzzle
    is generated fresh for maximum diversity. Digit-permutation derivation is
    kept only as a fallback to top up a difficulty whose fresh generation falls
    short of the requested count (digit permutation is an exact symmetry of the
    Bulls-and-Cows game: S/B counts are invariant, so the permuted puzzle is
    valid with no extra solve calls).

    ``difficulties`` restricts which tiers to generate (e.g. ``["medium"]`` to
    re-tune one tier without touching the others). When a single tier is given,
    ``num_questions`` is that tier's count; otherwise it is split across tiers.
    The combined JSONL only contains the generated tiers, so the downstream
    split step overwrites just those ``*_<tier>.jsonl`` files.
    """
    import pandas as pd

    # All-fresh: cap >= any realistic per-difficulty count so bases_needed ==
    # count (no permutation derivation unless fresh generation underperforms).
    BASE_PER_DIFF = 100000

    _name_to_diff = {
        "easy": Difficulty.EASY, "medium": Difficulty.MEDIUM, "hard": Difficulty.HARD,
    }
    if difficulties is None:
        difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    else:
        difficulties = [_name_to_diff[d.lower()] for d in difficulties]

    print(f"Generating {num_questions} number baseball puzzles (perm-fast mode)...")

    generator = ProblemGenerator()

    # Single tier -> num_questions is that tier's count; else split across tiers.
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

        print(f"\n=== Generating {diff_name} puzzles ({count} needed) ===")

        # --- Phase 1: generate base puzzles ---
        bases_needed = min(BASE_PER_DIFF, count)
        # Use a reduced inner-retry budget so each generate_problem call
        # fails fast; we compensate by looping in the outer loop.
        base_inner_retries = 300  # vs default 4000
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
                print(f"  [base attempt {retries}] Failed: {e}")
                continue
            is_valid, msg = validate_problem(problem)
            if not is_valid:
                retries += 1
                continue
            base_problems.append(problem)
            print(f"  [base {len(base_problems)}/{bases_needed}] digits={problem['num_digits']}, "
                  f"hints={len(problem['hints'])}, answer={problem['answer']}")

        if not base_problems:
            print(f"  No base puzzles generated for {diff_name}; skipping")
            continue

        # --- Phase 2: derive remaining via digit permutation ---
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
                'id': f'number_baseball_en_{diff_name}_{diff_success:04d}',
                'question': create_question(problem),
                'answer': problem['answer'],
                'solution': _build_baseball_solution_en(problem),
                'difficulty': diff_name,
            }
            all_puzzles.append(puzzle_data)
            diff_success += 1
            print(f"  [{diff_success}/{count}] digits={problem['num_digits']}, "
                  f"hints={len(problem['hints'])}, answer={problem['answer']}")

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("number_baseball_en", all_puzzles)

    return df, all_puzzles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Number Baseball Puzzle Generator (EN)")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate (per tier if --difficulty is set; else split across tiers)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], nargs="+", default=None, help="Generate only these tier(s), e.g. --difficulty medium")
    parser.add_argument("--workers", type=int, default=0, help="Accepted for compatibility; fast mode uses template bank")

    args = parser.parse_args()

    print("=" * 60)
    print("Number Baseball (Bulls and Cows) Puzzle Generator")
    print("=" * 60)

    create_dataset_files(num_questions=args.num, difficulties=args.difficulty)

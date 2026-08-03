"""
Inequality Evaluator

Evaluates inequality puzzle responses with constraint-based fallback validation.
Answer format: digit sequence for size<=9 (e.g., "3142"); for size>9, one char
per cell using lowercase letters for values 10+ (a=10 ... g=16, e.g.
"123456789abcdefg"). Space-separated numbers are also accepted when grading.
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class InequalityEvaluator(BaseEvaluator):
    """
    Inequality puzzle evaluator.

    Supports concatenated digits (size<=9), letter notation a-g for values
    10-16 (size>9), and space-separated numbers as a grading fallback.
    Falls back to constraint-based validation when the answer doesn't match
    the pre-computed value but still satisfies all inequality constraints.
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert at inequality-grid (Futoshiki-style) constraint puzzles.

### Rules
1. Fill blanks with a permutation of 1 through N; each of `<`, `>`, and hidden `?` between neighbors must be satisfiable (for `?`, some choice of `<` or `>`).
2. Keep given hints fixed; `_` marks empty cells in the alternating value–symbol row (e.g. `_ < 2 > _ < _`).
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: <left to right>
(If puzzle size <= 9, use concatenated digits only, e.g. Answer: 53241; if size > 9, write one character per cell, using lowercase letters for values 10 and above (a=10, b=11, ... g=16), concatenated without spaces, e.g. Answer: 123456789abcdefg.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 부등호 제약 그리드 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 1부터 N까지 순열로 빈칸을 채우고, 인접 `<`·`>`·`?`(숨은 부등호)를 모두 만족하는 배치를 제시하세요.
2. 주어진 힌트 숫자는 고정이며, `_`는 빈칸(값·기호 교차 나열, 예: `_ < 2 > _ < _`)입니다.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: <왼쪽에서 오른쪽>
(크기 9 이하는 숫자만 이어 쓰기, 예: Answer: 53241; 크기 9 초과는 10 이상의 값을 소문자(a=10, b=11, ... g=16)로 표기하여 공백 없이 이어 쓰기, 예: Answer: 123456789abcdefg.)
"""

    @staticmethod
    def _infer_size(expected_answer: str) -> int:
        """Infer puzzle size from the gold answer string.

        Letter-notation answers (values 10-16 written as a-g, one char per
        cell, e.g. "194c26b7d5g8af3e") have size == len(answer). Otherwise
        fall back to counting space-separated numbers, or the string length
        for concatenated digits.
        """
        s = str(expected_answer).strip()
        if not s:
            return 0
        if re.fullmatch(r'[0-9a-g]+', s, re.IGNORECASE) and re.search(
            r'[a-g]', s, re.IGNORECASE
        ):
            return len(s)
        nums = re.findall(r'\d+', s)
        return len(nums) if len(nums) > 1 else len(s)

    @staticmethod
    def _extract_letter_token(text: str, size: int) -> Optional[str]:
        """Find a size-length [0-9a-g] token (case-insensitive) in text.

        Prefers the last standalone occurrence; a second pass strips common
        separators (spaces, commas, hyphens) to rescue answers written as
        "1 9 4 c ...". Returns the token lowercased, or None.
        """
        if not text or not size:
            return None
        pattern = r'(?<![0-9A-Za-z])[0-9a-gA-G]{%d}(?![0-9A-Za-z])' % size
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].lower()
        compact = re.sub(r'[\s,\-]+', '', text)
        matches = re.findall(pattern, compact)
        if matches:
            return matches[-1].lower()
        return None

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract number sequence from LLM response."""
        expected_answer = puzzle.get("answer", "")
        size = puzzle.get("size", 0)
        answer_text = self._extract_final_answer_text(response) or response
        if not size and expected_answer:
            size = self._infer_size(expected_answer)

        uses_letters = bool(
            re.fullmatch(r'[0-9a-g]+', str(expected_answer), re.IGNORECASE)
            and re.search(r'[a-g]', str(expected_answer), re.IGNORECASE)
        )
        if uses_letters and size:
            token = self._extract_letter_token(answer_text, size)
            if token is None:
                tail = response[-200:] if len(response) > 200 else response
                token = self._extract_letter_token(tail, size)
            if token is not None:
                return token
            # fall through: the model may have used space-separated numbers,
            # which the digit-based logic below (with the corrected size)
            # can still recover and _check_answer can grade.

        patterns = [
            r'Answer\s*[:：]\s*([\d\s]+)',
            r'answer\s*[:：]\s*([\d\s]+)',
            r'정답\s*[:：]\s*([\d\s]+)',
            r'solution[:\s]+([\d\s]+)',
            r'sequence[:\s]+([\d\s]+)',
        ]

        def _normalize(nums):
            # Single concatenated run whose length == cell count -> single-digit
            # grid (e.g. size 25 "2514..."); _to_int_list splits it char-by-char.
            if len(nums) == 1 and size and len(nums[0]) == size:
                return nums[0]
            # Single-digit grid in ANY grouping (spaced singles "2 5 1 ...", or
            # row-grouped "31254 25143 ..."): flattening all digit runs yields
            # exactly `size` digit characters, one per cell.
            joined = ''.join(nums)
            if size and len(joined) == size:
                return joined
            # Multi-digit cells given as >= size separate tokens: keep `size`.
            if len(nums) >= size:
                return ' '.join(nums[:size])
            return None

        for pattern in patterns:
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                nums = re.findall(r'\d+', raw)
                out = _normalize(nums)
                if out is not None:
                    return out

        last_part = answer_text[-200:] if len(answer_text) > 200 else answer_text
        all_nums = re.findall(r'\d+', last_part)
        # Reuse the tagged-answer normalization: handles a single length==size
        # run, row-grouped tokens totaling `size` digits ("31254 25143 ..." when
        # the "Answer:" tag was stripped by _extract_final_answer_text), and the
        # >=size case.
        out = _normalize(all_nums)
        if out is not None:
            return out
        # Last resort: any single run whose length == cell count.
        for cand in reversed(all_nums):
            if size and len(cand) == size:
                return cand

        return None

    def _to_int_list(self, s: str, size: int):
        """Convert answer string to list of ints, handling all formats:
        letter notation (a=10 ... g=16), concatenated digits, and
        space-separated numbers."""
        if not s:
            return []
        s = str(s).strip()
        low = s.lower()
        if re.fullmatch(r'[0-9a-g]+', low) and re.search(r'[a-g]', low):
            return [
                int(ch) if ch.isdigit() else ord(ch) - ord('a') + 10
                for ch in low
            ]
        nums = re.findall(r'\d+', s)
        # Single concatenated run of single-digit cells (len == cell count),
        # e.g. a 5x5 grid answered as "2514314532423153125453421" (size 25).
        # (No size<=9 restriction: grid answers have size = number of cells.)
        if len(nums) == 1 and size and len(nums[0]) == size:
            return [int(d) for d in nums[0]]
        return [int(n) for n in nums]

    def _validate_inequality_solution(self, answer_str: str, puzzle: Dict) -> bool:
        """Validate an answer against inequality constraints.

        Prefers the schema fields `inequalities` (full list) + `hidden_inequalities`
        (indices that are hidden in display) when present; falls back to parsing
        the compact `problem` string.
        """
        size = puzzle.get("size", 0)

        parsed = self._to_int_list(answer_str, size)
        if len(parsed) != size:
            return False

        if sorted(parsed) != list(range(1, size + 1)):
            return False

        given_positions = puzzle.get("given_positions", [])
        given_values = puzzle.get("given_values", [])
        for pos, val in zip(given_positions, given_values):
            if pos < len(parsed) and parsed[pos] != val:
                return False

        inequalities = puzzle.get("inequalities")

        if inequalities:
            for i, ineq in enumerate(inequalities):
                if i + 1 >= len(parsed):
                    break
                if ineq == "<" and parsed[i] >= parsed[i + 1]:
                    return False
                if ineq == ">" and parsed[i] <= parsed[i + 1]:
                    return False
            return True

        problem_str = puzzle.get("problem", puzzle.get("solution", ""))
        if not problem_str:
            return False

        parts = problem_str.split()
        fallback_ineqs = parts[1::2]
        for i, ineq in enumerate(fallback_ineqs):
            if i + 1 >= len(parsed):
                break
            if ineq == "?":
                continue
            if ineq == "<" and parsed[i] >= parsed[i + 1]:
                return False
            if ineq == ">" and parsed[i] <= parsed[i + 1]:
                return False

        return True

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str],
        puzzle: Optional[Dict] = None,
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        size = self._infer_size(str(expected))

        expected_list = self._to_int_list(str(expected), size)
        predicted_list = self._to_int_list(str(predicted), size)

        correct = expected_list == predicted_list
        if correct:
            return True, 1.0

        # Fallback: accept if answer satisfies visible inequality constraints
        # (handles puzzles with hidden `?` inequalities where multiple visible
        # permutations could be consistent, though by construction visible
        # uniqueness should hold — this keeps the grader robust).
        if puzzle is not None and self._validate_inequality_solution(str(predicted), puzzle):
            return True, 1.0

        return False, 0.0

    def _process_response(
        self,
        puzzle: Dict[str, Any],
        response: str,
        latency_ms: float,
        usage: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Override base _process_response so _check_answer can see the puzzle
        (needed for constraint-based fallback validation).
        """
        usage = usage or {}

        if "error" in usage:
            return self._create_error_result(
                puzzle,
                response if response else "",
                latency_ms,
                usage["error"],
            )

        try:
            predicted = self._parse_answer(response, puzzle)
            correct, partial_score = self._check_answer(
                puzzle["answer"], predicted, puzzle
            )
            return EvaluationResult(
                puzzle_id=puzzle["id"],
                difficulty=puzzle.get("difficulty", "Unknown"),
                correct=correct,
                partial_score=partial_score,
                expected=puzzle["answer"],
                predicted=predicted,
                raw_response=response,
                latency_ms=latency_ms,
                thinking_content=usage.get("thinking_content", "") if isinstance(usage, dict) else "",
                finish_reason=usage.get("finish_reason", "") if isinstance(usage, dict) else "",
            )
        except Exception as e:
            return self._create_error_result(
                puzzle, response, latency_ms, str(e),
                finish_reason=usage.get("finish_reason") or "error",
            )
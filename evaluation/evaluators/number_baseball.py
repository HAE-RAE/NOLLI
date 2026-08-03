"""
Number Baseball Evaluator

Evaluates number baseball (Bulls and Cows) puzzle responses
with constraint-based fallback validation.
Answer format: digit string (e.g., "1234" or "012345")
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


class NumberBaseballEvaluator(BaseEvaluator):
    """
    Number Baseball puzzle evaluator.

    Falls back to constraint-based validation (checking all hints)
    when predicted answer doesn't match expected answer exactly.
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert at Bulls and Cows (Number Baseball) puzzles.

### Rules
1. Infer the secret digit string: unique digits 0–9 per position, length fixed by the puzzle; leading zeros are allowed.
2. Every hint’s strike (correct digit, correct place) and ball (correct digit, wrong place) counts must be satisfied by your answer.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: <digits>
(Digits only; no spaces or other characters; nothing after the answer on the Answer: line.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 숫자 야구(Bulls and Cows) 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 비밀 숫자는 서로 다른 0–9 자리로 이루어지며 길이는 문제에서 정해집니다. 앞자리 0이 허용됩니다.
2. 각 힌트의 스트라이크·볼 개수를 모두 만족하는 유일한 수를 찾으세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: <숫자열>
(숫자만; 공백·부가 텍스트 없이 Answer: 줄에 답만 적으세요.)
"""

    def _extract_hint_numbers(self, puzzle: Dict) -> set:
        """Extract hint guess numbers to use as blacklist for parsing."""
        hint_nums = set()
        hints = puzzle.get("hints", [])
        for h in hints:
            if isinstance(h, dict):
                hint_nums.add(h.get("guess", ""))
            elif isinstance(h, str):
                nums = re.findall(r'\d+', h)
                hint_nums.update(nums)
        return hint_nums

    def _infer_num_digits(self, puzzle: Dict) -> int:
        """Infer expected digit length from puzzle metadata/question when missing."""
        num_digits = puzzle.get("num_digits")
        if isinstance(num_digits, int) and num_digits > 0:
            return num_digits

        expected = str(puzzle.get("answer", "")).strip()
        if re.fullmatch(r"\d+", expected):
            return len(expected)

        question = str(puzzle.get("question", ""))
        m = re.search(r"(\d+)\s*-\s*digit|(\d+)\s*digit", question, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g]
            if groups:
                return int(groups[0])

        m_ko = re.search(r"(\d+)\s*자리", question)
        if m_ko:
            return int(m_ko.group(1))

        # Fallback: infer from hint guess token length in question text
        hint_guess = re.search(r"\[(\d+)\s*:\s*\d+\s*[sS]\s*\d+\s*[bB]\]", question)
        if hint_guess:
            return len(hint_guess.group(1))

        return 3

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract digit sequence from LLM response, filtering out hint numbers."""
        num_digits = self._infer_num_digits(puzzle)
        hint_numbers = self._extract_hint_numbers(puzzle)
        answer_text = self._extract_final_answer_text(response) or response

        # Priority 1: "Answer:" pattern
        patterns = [
            r'Answer:\s*(\d+)',
            r'answer:\s*(\d+)',
            r'secret number[:\s]+(\d+)',
            r'number is[:\s]+(\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            for m in reversed(matches):
                if len(m) == num_digits and m not in hint_numbers:
                    return m

        # Priority 2: last N-digit number not in hints
        last_part = answer_text[-500:] if len(answer_text) > 500 else answer_text
        numbers = re.findall(rf'\b(\d{{{num_digits}}})\b', last_part)
        for n in reversed(numbers):
            if n not in hint_numbers:
                return n

        # Priority 3: any N-digit number
        if numbers:
            return numbers[-1]

        return None

    def _calculate_strikes_balls(self, secret: str, guess: str) -> Tuple[int, int]:
        """Calculate strikes and balls."""
        strikes = sum(1 for s, g in zip(secret, guess) if s == g)
        balls = sum(1 for i, g in enumerate(guess) if g != secret[i] and g in secret)
        return strikes, balls

    def _validate_solution(self, answer: str, puzzle: Dict) -> bool:
        """Validate if answer satisfies all hints (constraint-based fallback)."""
        num_digits = puzzle.get("num_digits", 0)
        if not answer or len(answer) != num_digits:
            return False
        if len(set(answer)) != num_digits:
            return False
        if not all(c.isdigit() for c in answer):
            return False

        hints = puzzle.get("hints", [])
        for hint in hints:
            if isinstance(hint, dict):
                guess = hint["guess"]
                expected_s = hint["strikes"]
                expected_b = hint["balls"]
                s, b = self._calculate_strikes_balls(answer, guess)
                if s != expected_s or b != expected_b:
                    return False

        return True

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        correct = str(predicted) == str(expected)
        return correct, 1.0 if correct else 0.0

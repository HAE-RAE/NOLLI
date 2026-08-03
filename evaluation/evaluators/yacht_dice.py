"""
Yacht Dice Evaluator

Answer format: integer (total score or spotcheck round sum). Spotcheck wording
(if any) lives in ``puzzle["question"]`` from the generator.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


# must stay in sync with the generator's YachtDiceConfig
@dataclass
class YachtDiceConfig:
    bonus_threshold: int = 63
    bonus_points: int = 35
    full_house_points: int = 25
    small_straight_points: int = 30
    large_straight_points: int = 40
    yacht_points: int = 50
    optimization_goal: Literal["maximize", "minimize"] = "maximize"


class YachtDiceEvaluator(BaseEvaluator):
    """
    Yacht Dice puzzle evaluator.

    Sends the stored system prompt and ``question`` field unchanged; compares the
    parsed integer to ``puzzle["answer"]`` (full optimal total or spotcheck sum,
    depending on how the dataset was built).
    """

    CONFIG = YachtDiceConfig()

    SYSTEM_PROMPT = """### Instructions
You are an expert at Yacht Dice (Yahtzee-style) score assignment optimization.

### Rules
1. Follow the dice rolls, scoring categories, and point values exactly as given in the user message (including bonuses and category caps).
2. Assign each of the 12 rounds to a category at most once so the objective (usually maximize total or a stated spotcheck sum) is met optimally unless the puzzle specifies otherwise.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: [number]
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 요트 다이스(Yacht Dice, 야추 스타일) 점수 배정 최적화 문제를 푸는 전문가입니다.

### 규칙
1. 사용자 메시지에 제시된 주사위·점수 칸·보너스 규칙(카테고리별 상한 포함)을 그대로 따르세요.
2. 12라운드를 각 칸에 최대 한 번씩 배정하여, 목표(총점 또는 별도로 제시된 스팟체크 합 등)에 맞게 최적으로 배치하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: [숫자]
"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """Extract integer answer from LLM response (multi-step fallback)."""
        response = self._strip_code_fences(response).strip()
        answer_text = self._extract_final_answer_text(response) or response

        # Priority 1: "Answer:" pattern
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
            answer_text, re.IGNORECASE
        )
        if answer_matches:
            return int(answer_matches[-1])

        # Priority 2: Total/sum patterns
        total_patterns = [
            r'[Tt]otal[:\s]*[=\s]*(\d+)',
            r'[Ss]um[:\s]*[=\s]*(\d+)',
        ]
        for pattern in total_patterns:
            matches = re.findall(pattern, answer_text)
            if matches:
                return int(matches[-1])

        # Priority 3: last number in last 5 lines (largest on line)
        lines = answer_text.strip().split('\n')
        for line in reversed(lines[-5:]):
            nums = re.findall(r'\b(\d+)\b', line.strip())
            if nums:
                return int(max(nums, key=int))

        # Priority 4: last number anywhere
        all_nums = re.findall(r'\b(\d+)\b', answer_text)
        if all_nums:
            return int(all_nums[-1])

        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int],
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0
        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            return False, 0.0
        correct = int(predicted) == expected_num
        return correct, 1.0 if correct else 0.0

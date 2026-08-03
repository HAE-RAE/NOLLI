"""
Korean Traditional Units Evaluator (KO)

전통 단위 환산 퍼즐 평가기. 정답 = 기준 단위로 환산한 정수(예: "373돈").
응답에서 정수를 추출해 비교한다.
"""

import re
from typing import Dict, Any, Optional, Tuple

from ..core.base import BaseEvaluator


class KoreanUnitsEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = """### Instructions
You are an expert at traditional unit conversion.

### Rules
1. Use ONLY the conversion table given in the problem (not real-world knowledge of traditional units).
2. Convert each unit to the base unit (chain the table if needed) and multiply to get each item's value.
3. Apply the rules stated in the problem (per-item multiplier 'k배', [더함]/[뺌] signs, etc.) and compute the final sum.

### Output format
Your final line must be:
Answer: N<unit>
(digits exact, e.g. `Answer: 373돈`)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 도량형 환산 전문가입니다.

### 규칙
1. 문제에 함께 제시된 환산표만 사용합니다(실제 전통 단위 지식이 아니라 표 기준).
2. 각 단위를 기준 단위로 바꿔(필요하면 표를 연쇄 적용) 곱해 항목 값을 구합니다.
3. 문제에 명시된 규칙(항목별 배수 'k배', [더함]/[뺌] 부호 등)을 그대로 적용해 최종 합을 구합니다.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: N<단위>
(숫자만 정확히, 예: `Answer: 373돈`)
"""

    def _expected_int(self, expected: str) -> Optional[int]:
        m = re.search(r'(-?\d+)', str(expected))
        return int(m.group(1)) if m else None

    def _parse_answer(self, response: str, puzzle: Dict[str, Any]) -> Optional[int]:
        if not response:
            return None
        labeled = re.findall(r'(?:정답|answer)\s*[:：]?\s*([\d,]+)', response, re.IGNORECASE)
        if labeled:
            return int(labeled[-1].replace(',', ''))
        tail = response[-200:]
        nums = re.findall(r'[\d,]+', tail)
        for n in reversed(nums):
            digits = n.replace(',', '')
            if digits.isdigit():
                return int(digits)
        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0
        exp = self._expected_int(expected)
        if exp is None:
            return False, 0.0
        correct = int(predicted) == exp
        return correct, 1.0 if correct else 0.0

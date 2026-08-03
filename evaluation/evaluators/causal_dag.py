import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


class CausalDAGEvaluator(BaseEvaluator):    
    SYSTEM_PROMPT = """### Instructions
You are an expert at causal DAG and quantitative reasoning puzzles.

### Rules
1. Use the given causal structure and all stated constraints in order.
2. Derive the single numeric answer the problem asks for (e.g. minutes or counts) with correct units implied by the prompt.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: [number]
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 인과 DAG·정량 추론 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 주어진 인과 구조와 모든 제약을 순서대로 반영하세요.
2. 문제가 요구하는 단일 수치 답(분, 횟수 등)을 정확히 구하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: [숫자]
"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """Extract the numeric (minutes) answer from the LLM response."""
        response = response.strip()
        answer_text = self._extract_final_answer_text(response) or response
        
        # Pattern 1: Just a number
        if answer_text.isdigit():
            return int(answer_text)
        
        # Pattern 2: "Answer: 45" or "answer: 45"
        match = re.search(r'[Aa]nswer\s*[:：]\s*(\d+)', answer_text)
        if match:
            return int(match.group(1))
        
        # Pattern 3: "event X first occurs at minute 45" or "occurs at minute 45"
        matches = list(re.finditer(r'(?:first\s+)?occurs?\s+at\s+minute\s+(\d+)', answer_text, re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 4: "at minute 45" (last occurrence)
        matches = list(re.finditer(r'at\s+minute\s+(\d+)', answer_text, re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 5: "minute 45" (last occurrence)
        matches = list(re.finditer(r'[Mm]inute\s+(\d+)', answer_text))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 6: "45 minutes"
        match = re.search(r'(\d+)\s+[Mm]inutes?', answer_text)
        if match:
            return int(match.group(1))
        
        # Pattern 7: Last number in response
        numbers = re.findall(r'\b(\d+)\b', answer_text)
        if numbers:
            return int(numbers[-1])
        
        return None
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            return False, 0.0
        
        correct = predicted == expected_num
        return correct, 1.0 if correct else 0.0

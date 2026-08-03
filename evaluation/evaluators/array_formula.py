"""Array Formula Evaluator (spreadsheet/array-formula puzzles, EN/KO)."""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


class ArrayFormulaEvaluator(BaseEvaluator):
    """Answer is numeric or text; system prompt branches on locale."""

    SYSTEM_PROMPT = """### Instructions
You are an expert at spreadsheet and array-formula puzzles.

### Rules
1. Read the given table and the question carefully, then compute or infer the required value.
2. For numbers, reply with digits only (no units, commas, or symbols); truncate decimals unless the puzzle says otherwise. For text, give the exact string only.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: [answer]
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 스프레드시트·배열 수식 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 주어진 표와 질문을 꼼꼼히 읽고 필요한 값을 계산하거나 추론하세요.
2. 숫자는 숫자만(단위·쉼표·기호 없이), 별도 지시가 없으면 소수는 버림; 텍스트는 정확한 문자열만 제시하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: [답]
"""

    @staticmethod
    def _infer_answer_type(puzzle: Dict) -> str:
        """Use the row's `answer_type` if present, else infer from the gold `answer`."""
        answer_type = puzzle.get("answer_type")
        if answer_type in {"number", "text"}:
            return answer_type

        expected = str(puzzle.get("answer", "")).strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", expected):
            return "number"
        return "text"

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Any]:
        """Try 'Final answer:'/'Answer:'/'최종 답:' in order, else fall back to the last line."""
        answer_type = self._infer_answer_type(puzzle)

        answer_text = self._extract_final_answer_text(response)

        patterns = [
            r"[Ff]inal\s*[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)",
            r"[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)",
            r"최종\s*답\s*[:：]\s*(.+?)(?:\n|$)",
        ]

        if answer_text is None:
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer_text = match.group(1).strip()
                    break

        if answer_text is None:
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            if lines:
                answer_text = lines[-1]

        if answer_text is None:
            return None

        if answer_type == "number":
            number_match = re.search(r"-?[\d,]+\.?\d*", answer_text.replace(",", ""))
            if number_match:
                try:
                    num_str = number_match.group().replace(",", "")
                    if "." in num_str:
                        return float(num_str)
                    return int(num_str)
                except ValueError:
                    pass
            return None

        # Text type
        answer_text = answer_text.strip("'\"")
        return answer_text

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Any]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        answer_type = "number" if isinstance(predicted, (int, float)) else "text"

        if answer_type == "number":
            try:
                expected_num = float(expected)
                predicted_num = float(predicted)
                exact = abs(expected_num - predicted_num) < 0.001
                return exact, 1.0 if exact else 0.0
            except (ValueError, TypeError):
                return False, 0.0
        else:
            expected_str = str(expected).strip().lower()
            predicted_str = str(predicted).strip().lower()
            correct = expected_str == predicted_str
            return correct, 1.0 if correct else 0.0


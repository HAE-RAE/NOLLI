"""SAT Puzzle Evaluator (propositional-logic team puzzles, EN/KO)."""

import logging
import json
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


class SATPuzzleEvaluator(BaseEvaluator):
    """Answer format: JSON mapping variable name -> bool."""


    SYSTEM_PROMPT = """### Instructions
You are an expert at propositional logic and SAT-style team puzzles.

### Rules
1. Satisfy every clue in the user message and assign each named variable true or false consistently.
2. Put one-line JSON on the final line with lowercase boolean literals true/false; do not wrap it in markdown code fences.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: {"K team": false, "L team": true}
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 명제 논리·SAT형 팀 추론 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 사용자 메시지의 모든 제약을 만족하도록 각 변수에 참/거짓을 일관되게 부여하세요.
2. 마지막 줄은 한 줄 JSON이며 불리언은 소문자 true/false만 사용하고, 마크다운 코드블록으로 감싸지 마세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: {"K팀": false, "L팀": true}
"""

    @staticmethod
    def _as_assignment(value: Any) -> Dict[str, bool]:
        """Return the gold answer as a dict.

        The dataset stores `answer` as a JSON string (so the column has one
        stable type for CSV/HuggingFace); older files stored a raw dict. Accept
        both so evaluation works either way.
        """
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return {}
        return value if isinstance(value, dict) else {}

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Dict[str, bool]]:
        """Extract the JSON variable->bool answer from the LLM response."""
        variables = puzzle.get("variables") or list(self._as_assignment(puzzle.get("answer")).keys())
        answer_text = self._extract_final_answer_text(response)

        def _validate(answer_obj: Any) -> Optional[Dict[str, bool]]:
            if not isinstance(answer_obj, dict):
                return None
            if not variables:
                return answer_obj

            # Preserve the dataset's canonical variable names while accepting
            # harmless casing changes from the model, e.g. TeamA -> teama.
            lower_key_map = {str(key).lower(): key for key in answer_obj}
            canonical_answer: Dict[str, bool] = {}
            for var in variables:
                key = var if var in answer_obj else lower_key_map.get(str(var).lower())
                if key is None or not isinstance(answer_obj[key], bool):
                    return None
                canonical_answer[var] = answer_obj[key]
            return canonical_answer

        def _parse_json_like(text: str) -> Optional[Dict[str, bool]]:
            if not text:
                return None
            candidate = text.strip()
            candidate = re.sub(r'//.*', '', candidate)
            # Accept Python-style bools too: True/False -> true/false
            candidate = re.sub(r'\bTrue\b', 'true', candidate)
            candidate = re.sub(r'\bFalse\b', 'false', candidate)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return None
            return _validate(parsed)
        
        try:
            if answer_text and answer_text.strip().startswith("{"):
                parsed = _parse_json_like(answer_text)
                if parsed is not None:
                    return parsed

            # multiline fenced style: Answer:\n```json\n{ ... }\n```
            answer_block = re.search(
                r'(?is)answer\s*[:：]\s*```(?:json)?\s*(\{.*?\})\s*```',
                response,
            )
            if answer_block:
                parsed = _parse_json_like(answer_block.group(1))
                if parsed is not None:
                    return parsed

            # multiline unfenced style: Answer:\n{ ... }
            answer_multiline = re.search(
                r'(?is)answer\s*[:：]\s*(\{.*?\})',
                response,
            )
            if answer_multiline:
                parsed = _parse_json_like(answer_multiline.group(1))
                if parsed is not None:
                    return parsed

            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                parsed = _parse_json_like(json_match.group(1))
                if parsed is not None:
                    return parsed

            json_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*(?:true|false|True|False)[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = _parse_json_like(json_match.group(0))
                if parsed is not None:
                    return parsed

            # line-by-line fallback, e.g. "Alice: True"
            answer = {}
            for line in response.split('\n'):
                for var in variables:
                    if var in line:
                        if 'true' in line.lower() or ': true' in line.lower():
                            answer[var] = True
                            break
                        elif 'false' in line.lower() or ': false' in line.lower():
                            answer[var] = False
                            break
            
            if len(answer) == len(variables) and all(var in answer for var in variables):
                return answer
            
            return None
        
        except (json.JSONDecodeError, AttributeError):
            return None
    
    def _check_answer(
        self,
        expected: Dict[str, bool],
        predicted: Optional[Dict[str, bool]]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        expected = self._as_assignment(expected)
        if not expected:
            return False, 0.0

        # exact match only — no partial credit
        for var in expected:
            if var in predicted:
                pred_value = predicted[var]
            else:
                pred_key = next(
                    (key for key in predicted if str(key).lower() == str(var).lower()),
                    None,
                )
                if pred_key is None:
                    return False, 0.0
                pred_value = predicted[pred_key]

            if pred_value != expected[var]:
                return False, 0.0
        
        return True, 1.0

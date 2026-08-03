import re
from typing import Dict, Tuple, Optional

from ..core.base import BaseEvaluator


class JamoEvaluator(BaseEvaluator):
    """한글 자모 합성 task 평가기 (jamo_ko, 2축). 답 = 변환된 한글 문자열. KO 전용."""

    SYSTEM_PROMPT = """### Instructions
You decompose Korean syllable blocks into 초성/중성/종성, shift components per the
rule, and recompose. Treat each double final consonant (겹받침: ㄺ, ㅄ, ...) as a
single 종성.

### Rules
1. Decompose each syllable into 초성/중성/종성, shift the components per the rule, then recompose.
2. Reason step by step, then output the final string.

### Output format
Your final line must be:
Answer: <string>
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 한글 자모(초성·중성·종성) 분해와 재조합 전문가입니다. 겹받침(ㄺ, ㅄ 등)도
정확히 한 종성으로 처리하세요.

### 규칙
1. 각 글자를 초성·중성·종성으로 분해하고, 규칙대로 자모를 이동한 뒤 다시 합치세요.
2. 단계적으로 풀이한 뒤 최종 문자열을 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: 결과문자열
(평가기는 가장 마지막 Answer: 줄의 한글 문자열만 채점에 사용합니다.)
"""

    @staticmethod
    def _trim(raw: str) -> str:
        if not raw:
            return raw or ""
        m = list(re.finditer(r"answer\s*[:：]", raw, re.IGNORECASE))
        if m:
            return raw[m[-1].start():]
        for label in ("정답", "답"):
            mm = list(re.finditer(label + r"\s*[:：]", raw))
            if mm:
                return raw[mm[-1].start():]
        return raw

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        text = self._extract_final_answer_text(response) or response or ""
        text = self._trim(text)
        runs = re.findall(r"[가-힣]+", text)
        return runs[-1] if runs else None

    def _check_answer(self, expected: str, predicted: Optional[str]) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0
        ok = expected.strip() == predicted.strip()
        return ok, 1.0 if ok else 0.0

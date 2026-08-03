import re
from typing import Dict, Tuple, Optional

from ..core.base import BaseEvaluator

_GAN = "갑을병정무기경신임계"
_JI = "자축인묘진사오미신유술해"
_GANJI_RE = re.compile(rf"[{_GAN}][{_JI}]")


class SajuEvaluator(BaseEvaluator):
    """사주(四柱) 만세력 task 평가기. 답은 간지 2글자(예: 갑자). KO 전용."""

    SYSTEM_PROMPT = """### Instructions
You are an expert at the Korean Four Pillars (saju / manseryeok).

### Rules
1. Compute the requested pillar (year/month/day/hour) step by step.
2. Give the final answer as a single 2-character 간지.

### Output format
Your final line must be:
Answer: 간지
(e.g. 갑자. The evaluator uses only the last Answer: line.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 사주명리·만세력 전문가입니다. 연주(年柱, 입춘 경계), 월주(月柱, 절기 월지 + 월두법),
일주(日柱, 연속 60갑자 일진), 시주(時柱, 일간 + 시두법)를 정확히 계산합니다.

### 규칙
1. 묻는 기둥을 단계적으로 계산하세요.
2. 최종 답을 간지 2글자로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: 간지
(예: 갑자. 평가기는 가장 마지막 Answer: 줄만 채점에 사용합니다.)
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
        m = _GANJI_RE.search(text)
        return m.group(0) if m else None

    def _check_answer(self, expected: str, predicted: Optional[str]) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0
        ok = expected.strip() == predicted.strip()
        return ok, 1.0 if ok else 0.0

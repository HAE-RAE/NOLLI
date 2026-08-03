import re
from typing import Dict, Optional, Tuple

from ..core.base import BaseEvaluator

_GAN = "갑을병정무기경신임계"
_JI = "자축인묘진사오미신유술해"
_GANJI_RE = re.compile(rf"[{_GAN}][{_JI}]")
_WEEKDAY_RE = re.compile(r"[월화수목금토일]요일")
_DATE_RE = re.compile(r"(\d{3,4})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*(\(?\s*윤달\s*\)?)?")


class TimeEvaluator(BaseEvaluator):
    """한국어 날짜/달력(양력↔음력) 추론 task 평가기.

    답 유형은 인스턴스마다 다르다(난이도 = 일진 문제 혼합비):
      - 날짜  : 'YYYY.M.D' (음력 윤달이면 '(윤달)')
      - 요일  : 'X요일'
      - 일진  : 60갑자 2글자(예: 갑자)
    기대 답(puzzle['answer'])의 형식으로 유형을 판별해 해당 토큰만 채점한다. KO 전용.
    """

    SYSTEM_PROMPT = """### Instructions
You solve Korean calendar reasoning puzzles (solar/lunar dates, day offsets,
solar<->lunar conversion, and the sexagenary day-pillar 일진).

### Rules
1. Reason step by step.
2. Give the final answer in the requested form:
   - date as YYYY.M.D (add (윤달) for a lunar leap month)
   - weekday as 'X요일'
   - 일진 as a 2-character 간지

### Output format
Your final line must be:
Answer: <answer>
(The evaluator uses only the last Answer: line.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 한국 달력(양력·음력) 추론 전문가입니다. 공휴일 날짜, 상대일(오늘/내일/어제 등),
일수 가산, 양력↔음력 변환, 그리고 일진(日辰, 연속 60갑자)을 정확히 계산합니다.

### 규칙
1. 단계적으로 계산하세요.
2. 묻는 형식에 맞춰 최종 답을 제시하세요.
   - 날짜: 'YYYY.M.D' (음력 윤달이면 '(윤달)' 표기)
   - 요일: 'X요일' (예: 월요일)
   - 일진: 간지 2글자 (예: 갑자)

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: <답>
(평가기는 가장 마지막 Answer: 줄만 채점에 사용합니다.)
"""

    @staticmethod
    def _answer_type(expected: str) -> str:
        e = (expected or "").strip()
        if _GANJI_RE.fullmatch(e):
            return "ganji"
        if e.endswith("요일"):
            return "weekday"
        return "date"

    @staticmethod
    def _norm_date(m: re.Match) -> str:
        leap = "(윤달)" if (m.group(4) and "윤" in m.group(4)) else ""
        return f"{int(m.group(1))}.{int(m.group(2))}.{int(m.group(3))}{leap}"

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        # 마지막 '정답:/Answer:' 라인 우선, 없으면 전체 응답에서 마지막 토큰.
        text = self._extract_final_answer_text(response) or response or ""
        atype = self._answer_type(str(puzzle.get("answer", "")))

        if atype == "ganji":
            ms = _GANJI_RE.findall(text) or _GANJI_RE.findall(response or "")
            return ms[-1] if ms else None
        if atype == "weekday":
            ms = _WEEKDAY_RE.findall(text) or _WEEKDAY_RE.findall(response or "")
            return ms[-1] if ms else None
        ms = list(_DATE_RE.finditer(text)) or list(_DATE_RE.finditer(response or ""))
        return self._norm_date(ms[-1]) if ms else None

    def _check_answer(self, expected: str, predicted: Optional[str]) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0
        exp = (expected or "").strip()
        if self._answer_type(exp) == "date":
            m = _DATE_RE.search(exp)
            if m:
                exp = self._norm_date(m)
        ok = exp == predicted.strip()
        return ok, 1.0 if ok else 0.0

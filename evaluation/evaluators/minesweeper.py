"""Minesweeper Evaluator.

Answer format: coordinate list "(r1,c1), (r2,c2), ..." (exact set match).
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional, Set

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|except|exclude|excluding|without|isn't|aren't|don't|cannot|wrong)\b",
    re.IGNORECASE,
)
_COORD_PAIR_RE = re.compile(
    r"\(\s*(?:r|row)?\s*(\d+)\s*,\s*(?:c|col)?\s*(\d+)\s*\)",
    re.IGNORECASE,
)


class MinesweeperEvaluator(BaseEvaluator):
    """Minesweeper evaluator with coordinate-set exact match scoring."""

    SYSTEM_PROMPT = """### Instructions
You are an expert at logical Minesweeper deduction.

### Rules
1. Treat '#' as hidden and digits 0–8 as revealed counts of mines among eight neighbors; use the stated total mine count.
2. Assume a unique satisfying mine layout consistent with the given clues.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: (0,1), (0,3), (2,4), ...
(List every mine cell as "(row,col)" with 0-based row and column; sort by row then column; digits only, no prefixes.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 지뢰찾기(Minesweeper) 논리 추론 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. '#'은 미공개, 0–8은 인접 8칸 중 지뢰 개수이며, 문제에 주어진 지뢰 총개수를 반영하세요.
2. 단서와 일치하는 지뢰 배치가 유일하다고 가정하고 풀어야 합니다.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: (0,1), (0,3), (2,4), ...
(모든 지뢰 칸을 0부터 행·열 좌표로 "(행,열)" 형식에 맞게 나열; 행 우선 정렬; 숫자만, 접두어 없음.)
"""

    @staticmethod
    def _parse_coord_set(text: str) -> Optional[Set[Tuple[int, int]]]:
        """Extract a set of (r,c) pairs from an output string."""
        text = re.sub(r"```[a-z]*\n?", "", text)
        text = re.sub(r"```", "", text).strip()

        answer_matches = list(re.finditer(
            r"(?:Answer|Output|Final\s*Answer|답)\s*[:\s]*(.*)",
            text,
            re.IGNORECASE,
        ))
        if answer_matches:
            coords_str = answer_matches[-1].group(1)
            if not _NEGATION_RE.search(coords_str):
                pairs = _COORD_PAIR_RE.findall(coords_str)
                if pairs:
                    return {(int(r), int(c)) for r, c in pairs}

        lines = text.strip().split("\n")
        for line in reversed(lines[-3:]):
            stripped = line.strip()
            if _NEGATION_RE.search(stripped):
                continue
            pairs = _COORD_PAIR_RE.findall(stripped)
            if pairs:
                cleaned = _COORD_PAIR_RE.sub("", stripped)
                cleaned = re.sub(r"[,\s\[\]\-:.]", "", cleaned)
                if len(cleaned) <= 6:
                    return {(int(r), int(c)) for r, c in pairs}

        return None

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Set[Tuple[int, int]]]:
        answer_text = self._extract_final_answer_text(response)
        if answer_text:
            parsed = self._parse_coord_set(answer_text)
            if parsed is not None:
                return parsed
        return self._parse_coord_set(response)

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Set[Tuple[int, int]]],
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        truth = self._parse_coord_set(str(expected)) if not isinstance(expected, set) else expected
        if truth is None:
            return False, 0.0

        correct = truth == predicted
        return correct, 1.0 if correct else 0.0

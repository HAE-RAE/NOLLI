"""Sudoku evaluator.

Answer format: K space-separated digits at spotcheck (row, col) positions.
- _prepare_puzzle_for_eval: reads puzzle["spotcheck"]["positions"], appends
  spotcheck instruction to user content.
- _parse_answer: extracts K space-separated digits (each 1-9) from
  "Answer: d1 d2 ..." line; falls back to last 5 lines for a matching
  digit sequence of correct length.
- _check_answer: whitespace-normalized string comparison against expected
  answer ("d1 d2 ... dK").
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class SudokuEvaluator(BaseEvaluator):
    """Sudoku puzzle evaluator using spotcheck digit-tuple scoring."""

    SYSTEM_PROMPT = """### Instructions
You are an expert at standard 9×9 Sudoku.

### Rules
1. Each row, column, and 3×3 box must contain digits 1–9 exactly once; '.' or '0' marks a given empty cell.
2. Solve the full grid implied by the puzzle, then read the spotcheck (row, col) list the user appends and take digits from your completed solution in that order.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: d1 d2 d3 ...
(each d_i is 1–9 for the i-th spotcheck position, single spaces; e.g. Answer: 5 3 4 6 7 8 for six checks.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 표준 9×9 스도쿠 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 각 행·열·3×3 박스에 1–9가 정확히 한 번씩; '.' 또는 '0'은 빈 칸입니다.
2. 퍼즐을 완전히 푼 뒤, 사용자가 덧붙인 스팟체크 (행, 열) 순서대로 완성 격자에서 숫자를 읽으세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: d1 d2 d3 ...
(각 d_i는 i번째 스팟체크 위치의 1–9, 공백 한 칸 구분. 예: 스팟체크 6개면 Answer: 5 3 4 6 7 8)
"""

    @staticmethod
    def _parse_position(pos: str) -> Tuple[int, int]:
        parts = pos[1:].split('c')
        return int(parts[0]) - 1, int(parts[1]) - 1

    def _format_rc_pairs(self, positions: List[str]) -> str:
        pairs = []
        for pos in positions:
            r, c = self._parse_position(pos)
            pairs.append(f"({r+1}, {c+1})")
        return ", ".join(pairs)

    def _spotcheck_suffix(self, positions: List[str], is_ko: bool) -> str:
        rc_str = self._format_rc_pairs(positions)
        k = len(positions)
        if is_ko:
            return (
                f"\n\n스팟체크 좌표 (행, 열) (1-기반): {rc_str}\n"
                f"위 {k}개 좌표의 값을 같은 순서로 제시하세요.\n"
                f"마지막 줄은 반드시 다음 형식이어야 합니다:\n"
                f"Answer: [{k}개 숫자를 공백으로 구분]"
            )
        return (
            f"\n\nThe spotcheck positions (row, col) are (1-based): {rc_str}\n"
            f"Report the values at those {k} positions in the same order.\n"
            f"Your last line MUST be:\n"
            f"Answer: [{k} digits separated by spaces]"
        )

    def _prepare_puzzle_for_eval(self, puzzle: Dict) -> Tuple[Dict, str]:
        """Return (puzzle, user_content) with spotcheck instruction appended.

        If puzzle['spotcheck']['positions'] is present, append the spotcheck
        instruction suffix to the question text.
        """
        puzzle = dict(puzzle)
        user_content = puzzle.get("question", "")
        spotcheck = puzzle.get("spotcheck") or {}
        positions = spotcheck.get("positions") if isinstance(spotcheck, dict) else None
        if positions:
            is_ko = self._is_korean(puzzle)
            # Only append if the suffix isn't already in the question body.
            if "spotcheck" not in user_content.lower() and "스팟체크" not in user_content:
                user_content = user_content + self._spotcheck_suffix(positions, is_ko)
        return puzzle, user_content

    def _expected_k(self, puzzle: Dict) -> int:
        spotcheck = puzzle.get("spotcheck") or {}
        if isinstance(spotcheck, dict):
            if "k" in spotcheck and isinstance(spotcheck["k"], int):
                return spotcheck["k"]
            positions = spotcheck.get("positions")
            if isinstance(positions, list):
                return len(positions)
        expected = puzzle.get("answer", "")
        if isinstance(expected, str) and expected:
            return len(expected.split())
        return 0

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract K space-separated digits from response."""
        if not response:
            return None

        cleaned = self._strip_code_fences(response)
        answer_text = self._extract_final_answer_text(cleaned)

        k = self._expected_k(puzzle)

        lines = cleaned.split('\n')

        # Priority 1: "Answer:" line near the end.
        answer_line = answer_text
        if answer_line is None:
            for line in lines:
                stripped = line.strip()
                if re.match(r'(?i)^\s*answer\s*[:\-]', stripped):
                    answer_line = stripped

        if answer_line:
            nums = re.findall(r'[1-9]', answer_line)
            if k > 0 and len(nums) == k:
                return ' '.join(nums)
            # If K unknown or mismatched, still try if we have >=1 digit.
            if k == 0 and nums:
                return ' '.join(nums)

        # Priority 2: scan last 5 lines for a matching-length digit sequence.
        if k > 0:
            for line in reversed(lines[-5:]):
                nums = re.findall(r'[1-9]', line)
                if len(nums) == k:
                    return ' '.join(nums)

        # Priority 3: any digit sequence of length K across the whole response.
        if k > 0:
            all_nums = re.findall(r'[1-9]', cleaned)
            if len(all_nums) >= k:
                return ' '.join(all_nums[-k:])

        return None

    @staticmethod
    def _normalize(s: Any) -> str:
        return ' '.join(str(s).split())

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[str],
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0
        exp_norm = self._normalize(expected)
        pred_norm = self._normalize(predicted)
        correct = exp_norm == pred_norm
        return correct, 1.0 if correct else 0.0

    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient",
    ) -> EvaluationResult:
        enriched, user_content = self._prepare_puzzle_for_eval(puzzle)
        messages = [
            {"role": "system", "content": self._get_system_prompt(enriched)},
            {"role": "user", "content": user_content},
        ]
        start = time.time()
        try:
            response, usage = llm_client.generate(messages)
            latency = (time.time() - start) * 1000
            return self._process_response(enriched, response, latency, usage)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return self._process_response(enriched, "", latency, {"error": str(e)})

    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10,
    ) -> List[EvaluationResult]:
        from ..core.base import logger as core_logger

        enriched_puzzles: List[Dict[str, Any]] = []
        messages_list = []
        for puzzle in puzzles:
            enriched, user_content = self._prepare_puzzle_for_eval(puzzle)
            enriched_puzzles.append(enriched)
            messages_list.append([
                {"role": "system", "content": self._get_system_prompt(enriched)},
                {"role": "user", "content": user_content},
            ])

        total_puzzles = len(enriched_puzzles)
        task_name = getattr(self, "_task_name", None)
        task_prefix = f"[{task_name}] " if task_name else ""

        if verbose:
            core_logger.info(
                f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, "
                f"max_concurrent={max_concurrent}"
            )

        start_time = time.time()

        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    core_logger.info(
                        f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)"
                    )

        responses = await llm_client.async_batch_generate(
            messages_list,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None,
        )
        total_latency = (time.time() - start_time) * 1000

        if verbose:
            core_logger.info(
                f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in "
                f"{total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)"
            )

        results = []
        correct_count = 0
        error_count = 0
        for puzzle, (response, usage) in zip(enriched_puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            result = self._process_response(puzzle, response, latency_ms, usage)
            if result.correct:
                correct_count += 1
            if result.error:
                error_count += 1
            results.append(result)

        if verbose:
            incorrect_count = total_puzzles - correct_count - error_count
            core_logger.info(
                f"Processing completed: {correct_count} correct, "
                f"{incorrect_count} incorrect, {error_count} errors"
            )

        return results

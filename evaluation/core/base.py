import logging
import time
import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

from ..task_names import locale_from_task_name

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    puzzle_id: str
    difficulty: str
    correct: bool
    partial_score: float
    expected: Any
    predicted: Any
    raw_response: str
    latency_ms: float
    error: Optional[str] = None
    thinking_content: str = ""
    finish_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "difficulty": self.difficulty,
            "correct": self.correct,
            "partial_score": self.partial_score,
            "expected": str(self.expected),
            "predicted": str(self.predicted) if self.predicted is not None else None,
            "raw_response": self.raw_response,
            "thinking_content": self.thinking_content,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "finish_reason": self.finish_reason,
        }


class TaskEvaluator(Protocol):
    """Minimum interface a task evaluator must implement (structural typing via Protocol)."""

    def evaluate(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        use_async: bool = False,
        max_concurrent: int = 10
    ) -> List[EvaluationResult]:
        ...


class BaseEvaluator(ABC):
    """Common eval loop; subclasses define SYSTEM_PROMPT(S) and _parse_answer/_check_answer."""

    SYSTEM_PROMPT: str
    KOREAN_SYSTEM_PROMPT: str = ""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """Detect Korean locale from task_name, then question/answer text."""
        task = getattr(self, "_task_name", None) or ""
        hint = locale_from_task_name(task)
        if hint is not None:
            return hint
        if puzzle is not None:
            text = str(puzzle.get("question", "")) + str(puzzle.get("answer", ""))
            return bool(re.search(r"[가-힣]", text))
        return False

    def _get_system_prompt(self, puzzle: Dict) -> str:
        """Return the appropriate system prompt for the given puzzle."""
        if self._is_korean(puzzle) and self.KOREAN_SYSTEM_PROMPT:
            return self.KOREAN_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences from model response."""
        cleaned = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", text)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned

    def _extract_final_answer_text(self, response: str) -> Optional[str]:
        """
        Extract final answer payload from the last labeled answer line.

        Matches: Answer / Final answer / 최종 답 / 정답 / 원문 / Plaintext (case-insensitive).
        """
        cleaned = self._strip_code_fences(response or "").strip()
        if not cleaned:
            return None

        label_pattern = re.compile(
            r"(?:^|\n)\s*"
            r"(?:final\s*answer|answer|최종\s*답|정답|원문|plaintext)\s*[:：]\s*(.+)",
            re.IGNORECASE,
        )
        labels = label_pattern.findall(cleaned)
        if labels:
            return labels[-1].strip()

        return None
    
    def evaluate(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        use_async: bool = False,
        max_concurrent: int = 10,
        task_name: Optional[str] = None
    ) -> List[EvaluationResult]:
        """Evaluate all puzzles, dispatching to async or sync execution."""
        self._task_name = task_name
        if use_async:
            if verbose:
                logger.info("Starting async evaluation...")
            try:
                try:
                    loop = asyncio.get_running_loop()
                    logger.warning("Event loop already running, falling back to synchronous mode")
                    use_async = False
                except RuntimeError:
                    pass

                if use_async:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            results = loop.run_until_complete(
                                self._evaluate_async(puzzles, llm_client, verbose, max_concurrent)
                            )
                        finally:
                            try:
                                loop.close()
                            except Exception as e:
                                if verbose:
                                    logger.warning(f"Error closing loop: {e}")
                            finally:
                                asyncio.set_event_loop(None)

                        return results
                    except Exception as e:
                        logger.error(f"Error in async evaluation: {e}")
                        import traceback
                        if verbose:
                            logger.debug(traceback.format_exc())
                        raise
                else:
                    results = []
                    for i, puzzle in enumerate(puzzles):
                        result = self._evaluate_single(puzzle, llm_client)
                        results.append(result)
                        
                        if verbose:
                            status = "CORRECT" if result.correct else "INCORRECT"
                            logger.info(f"[{i+1}/{len(puzzles)}] {puzzle['id']}... {status}")
                    
                    return results
            except Exception as e:
                logger.error(f"Error in async evaluation: {e}")
                import traceback
                if verbose:
                    logger.debug(traceback.format_exc())
                raise
        else:
            results = []
            for i, puzzle in enumerate(puzzles):
                result = self._evaluate_single(puzzle, llm_client)
                results.append(result)
                
                if verbose:
                    status = "CORRECT" if result.correct else "INCORRECT"
                    logger.info(f"[{i+1}/{len(puzzles)}] {puzzle['id']}... {status}")
            
            return results
    
    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient"
    ) -> EvaluationResult:
        """
        Evaluate a single puzzle
        
        Args:
            puzzle: Puzzle data
            llm_client: LLM client
            
        Returns:
            Evaluation result
        """
        messages = [
            {"role": "system", "content": self._get_system_prompt(puzzle)},
            {"role": "user", "content": puzzle["question"]}
        ]
        
        start = time.time()
        try:
            response, usage = llm_client.generate(messages)
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, response, latency, usage)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, "", latency, {"error": str(e)})
    
    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10
    ) -> List[EvaluationResult]:
        """Async-batch variant of evaluate(), run via llm_client.async_batch_generate."""
        messages_list = []
        for puzzle in puzzles:
            messages = [
                {"role": "system", "content": self._get_system_prompt(puzzle)},
                {"role": "user", "content": puzzle["question"]}
            ]
            messages_list.append(messages)
        
        total_puzzles = len(puzzles)
        task_name = getattr(self, '_task_name', None)
        task_prefix = f"[{task_name}] " if task_name else ""
        
        if verbose:
            logger.info(f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, max_concurrent={max_concurrent}")
        
        start_time = time.time()
        
        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    logger.info(f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)")
        
        responses = await llm_client.async_batch_generate(
            messages_list, 
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None
        )
        total_latency = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in {total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)")
        
        results = []
        correct_count = 0
        error_count = 0
        
        for puzzle, (response, usage) in zip(puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            result = self._process_response(puzzle, response, latency_ms, usage)
            
            if result.correct:
                correct_count += 1
            if result.error:
                error_count += 1
            
            results.append(result)
        
        if verbose:
            incorrect_count = total_puzzles - correct_count - error_count
            logger.info(f"Processing completed: {correct_count} correct, {incorrect_count} incorrect, {error_count} errors")
        
        return results
    
    def _process_response(
        self,
        puzzle: Dict[str, Any],
        response: str,
        latency_ms: float,
        usage: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Response processing shared by sync/async: parse, grade, wrap as EvaluationResult."""
        usage = usage or {}

        if "error" in usage:
            return self._create_error_result(
                puzzle, 
                response if response else "", 
                latency_ms, 
                usage["error"]
            )
        
        try:
            predicted = self._parse_answer(response, puzzle)
            correct, partial_score = self._check_answer(
                puzzle["answer"],
                predicted
            )
            
            return EvaluationResult(
                puzzle_id=puzzle["id"],
                difficulty=puzzle.get("difficulty", "Unknown"),
                correct=correct,
                partial_score=partial_score,
                expected=puzzle["answer"],
                predicted=predicted,
                raw_response=response,
                latency_ms=latency_ms,
                thinking_content=usage.get("thinking_content", "") if isinstance(usage, dict) else "",
                finish_reason=usage.get("finish_reason", "") if isinstance(usage, dict) else "",
            )
        except Exception as e:
            return self._create_error_result(
                puzzle, response, latency_ms, str(e),
                finish_reason=usage.get("finish_reason") or "error",
            )
    
    def _create_error_result(
        self,
        puzzle: Dict[str, Any],
        response: str,
        latency_ms: float,
        error: str,
        finish_reason: str = "error"
    ) -> EvaluationResult:
        """Wrap a failed call/parse as an EvaluationResult(correct=False)."""
        return EvaluationResult(
            puzzle_id=puzzle["id"],
            difficulty=puzzle.get("difficulty", "Unknown"),
            correct=False,
            partial_score=0.0,
            expected=puzzle["answer"],
            predicted=None,
            raw_response=response if response else "",
            latency_ms=latency_ms,
            error=error,
            finish_reason=finish_reason
        )
    
    @abstractmethod
    def _parse_answer(self, response: str, puzzle: Dict[str, Any]) -> Optional[Any]:
        """Extract the predicted answer from the raw LLM response. Implemented per task."""
        pass

    @abstractmethod
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Any]
    ) -> Tuple[bool, float]:
        """Grade predicted against expected -> (is_correct, partial_score). Implemented per task."""
        pass

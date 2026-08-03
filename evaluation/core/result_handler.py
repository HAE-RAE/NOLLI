import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict

from .base import EvaluationResult

logger = logging.getLogger(__name__)


class ResultHandler:
    """Evaluation result storage and aggregation"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        task_name: str,
        results: List[EvaluationResult],
        model: str,
        puzzles: Optional[List[Dict[str, Any]]] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        gen_kwargs_cli: Optional[str] = None,
    ) -> Optional[Dict[str, Path]]:
        """Save results as {output_dir}/{model}/{task_name}/*.{csv,json}."""
        if not results:
            logger.warning(f"No results to save for {task_name}")
            return None
        
        model_safe = model.replace("/", "_").replace(":", "_")
        
        model_dir = self.output_dir / model_safe
        task_dir = model_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        summary = self._calculate_summary(task_name, results, model)
        accuracy = summary["accuracy"]
        
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        base_filename = f"{model_safe}_{task_name}_{timestamp}__{accuracy:.2f}"
        
        csv_path = task_dir / f"{base_filename}.csv"
        self._save_csv(csv_path, results, puzzles)

        json_path = task_dir / f"{base_filename}.json"
        self._save_json_summary(
            json_path,
            task_name,
            model,
            summary,
            timestamp,
            gen_kwargs=gen_kwargs,
            gen_kwargs_cli=gen_kwargs_cli,
        )
        
        logger.info(f"Results saved (CSV): {csv_path.name}")
        logger.info(f"Results saved (JSON): {json_path.name}")
        
        return {"csv": csv_path, "json": json_path}
    
    def _save_csv(
        self,
        filepath: Path,
        results: List[EvaluationResult],
        puzzles: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Save per-item results to CSV, joined against the original puzzle data."""
        puzzle_map = {}
        if puzzles:
            for puzzle in puzzles:
                puzzle_map[puzzle["id"]] = puzzle
        
        rows = []
        for result in results:
            puzzle = puzzle_map.get(result.puzzle_id, {})
            
            row = {
                "id": result.puzzle_id,
                "question": puzzle.get("question", ""),
                "answer": str(result.expected),
                "thinking_content": result.thinking_content,  # <think> block split out by the vLLM reasoning parser
                "resps": result.raw_response,
                "filtered_resps": str(result.predicted) if result.predicted is not None else "",  # parsed/extracted answer
                "exact_match": 1 if result.correct else 0,
                "difficulty": result.difficulty,
                "finish_reason": result.finish_reason
            }
            rows.append(row)

        if rows:
            fieldnames = ["id", "question", "answer", "thinking_content", "resps", "filtered_resps", "exact_match", "difficulty", "finish_reason"]
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    def _save_json_summary(
        self,
        filepath: Path,
        task_name: str,
        model: str,
        summary: Dict[str, Any],
        timestamp: str,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        gen_kwargs_cli: Optional[str] = None,
    ) -> None:
        """Save overall + per-difficulty summary stats and run metadata to JSON."""
        metadata: Dict[str, Any] = {
            "task": task_name,
            "model": model,
            "timestamp": timestamp,
            "total_puzzles": summary["total_count"],
            "gen_kwargs": gen_kwargs if gen_kwargs is not None else {},
        }
        if gen_kwargs_cli is not None:
            metadata["gen_kwargs_cli"] = gen_kwargs_cli

        output = {
            "metadata": metadata,
            "summary": {
                "overall": {
                    "accuracy": summary["accuracy"],
                    "correct_count": summary["correct_count"],
                    "total_count": summary["total_count"],
                    "avg_latency_ms": summary["avg_latency_ms"],
                    "finish_reason_counts": summary["finish_reason_counts"]
                },
                "by_difficulty": summary["by_difficulty"]
            }
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
    def _calculate_summary(
        self,
        task_name: str,
        results: List[EvaluationResult],
        model: str
    ) -> Dict[str, Any]:
        """Aggregate accuracy overall, per-difficulty, and per-finish_reason."""
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
        
        DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]
        by_difficulty = {}
        difficulties = set(r.difficulty.lower() if r.difficulty else "unknown" for r in results)
        ordered_diffs = [d for d in DIFFICULTY_ORDER if d in difficulties]
        ordered_diffs += sorted(d for d in difficulties if d not in DIFFICULTY_ORDER)
        
        for diff in ordered_diffs:
            diff_results = [r for r in results if (r.difficulty or "").lower() == diff]
            if diff_results:
                diff_correct = sum(1 for r in diff_results if r.correct)
                
                by_difficulty[diff] = {
                    "total": len(diff_results),
                    "correct": diff_correct,
                    "accuracy": diff_correct / len(diff_results)
                }
        
        errors = [r.error for r in results if r.error]

        # empty finish_reason (provider didn't report one) is counted as "unknown"
        finish_reason_counts: Dict[str, int] = {}
        for r in results:
            reason = getattr(r, "finish_reason", "") or "unknown"
            finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1
        finish_reason_counts = dict(
            sorted(finish_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )

        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct_count": correct,
            "total_count": total,
            "avg_latency_ms": avg_latency,
            "by_difficulty": by_difficulty,
            "error_count": len(errors),
            "finish_reason_counts": finish_reason_counts,
        }
    
    def load(self, filepath: Path) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def aggregate_results(self, pattern: str = "*.json") -> Dict[str, List[Dict[str, Any]]]:
        """Group saved result-file summaries by task name."""
        aggregated = defaultdict(list)
        
        for filepath in self.output_dir.glob(pattern):
            data = self.load(filepath)
            task = data["metadata"]["task"]
            aggregated[task].append(data["summary"])
        
        return dict(aggregated)

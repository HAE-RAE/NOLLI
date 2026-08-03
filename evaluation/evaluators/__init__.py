from typing import Dict, Type

from ..task_names import resolve_registry_task_key


def _get_registry() -> Dict[str, Type]:
    """Dynamically create the evaluator registry."""
    from .kinship import KinshipEvaluator
    from .cipher import CipherEvaluator
    from .array_formula import ArrayFormulaEvaluator
    from .causal_dag import CausalDAGEvaluator
    from .cryptarithmetic import CryptarithmeticEvaluator
    from .inequality import InequalityEvaluator
    from .minesweeper import MinesweeperEvaluator
    from .number_baseball import NumberBaseballEvaluator
    from .sat_puzzle import SATPuzzleEvaluator
    from .sudoku import SudokuEvaluator
    from .yacht_dice import YachtDiceEvaluator
    from .saju import SajuEvaluator
    from .jamo import JamoEvaluator
    from .time import TimeEvaluator
    from .korean_units import KoreanUnitsEvaluator

    return {
        "array_formula_en": ArrayFormulaEvaluator,
        "array_formula_ko": ArrayFormulaEvaluator,
        "causal_dag_en": CausalDAGEvaluator,
        "causal_dag_ko": CausalDAGEvaluator,
        "cipher_en": CipherEvaluator,
        "cipher_ko": CipherEvaluator,
        "cryptarithmetic_en": CryptarithmeticEvaluator,
        "cryptarithmetic_ko": CryptarithmeticEvaluator,
        "inequality_en": InequalityEvaluator,
        "inequality_ko": InequalityEvaluator,
        "kinship_ko": KinshipEvaluator,
        "saju_ko": SajuEvaluator,
        "jamo_ko": JamoEvaluator,
        "time_ko": TimeEvaluator,
        "korean_units_ko": KoreanUnitsEvaluator,
        "minesweeper_en": MinesweeperEvaluator,
        "minesweeper_ko": MinesweeperEvaluator,
        "number_baseball_en": NumberBaseballEvaluator,
        "number_baseball_ko": NumberBaseballEvaluator,
        "sat_puzzles_en": SATPuzzleEvaluator,
        "sat_puzzles_ko": SATPuzzleEvaluator,
        "sudoku_en": SudokuEvaluator,
        "sudoku_ko": SudokuEvaluator,
        "yacht_dice_en": YachtDiceEvaluator,
        "yacht_dice_ko": YachtDiceEvaluator,
    }


def get_evaluator(task_name: str):
    """Return an evaluator instance for the given task name."""
    registry = _get_registry()
    key = resolve_registry_task_key(task_name, set(registry.keys()))

    if key not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")

    return registry[key]()


def list_tasks() -> list:
    """Dynamically return the list of all registered tasks."""
    return list(_get_registry().keys())


__all__ = ["get_evaluator", "list_tasks"]

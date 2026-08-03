#!/usr/bin/env python3
"""
Boolean SAT (Satisfiability) Puzzle Generator

Generates logic puzzles in CNF (Conjunctive Normal Form) with natural language.
Uses SAT solver to ensure unique solutions.
"""

import random
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from itertools import combinations


SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class SATClause:
    """Represents a single clause (disjunction of literals)"""
    literals: List[Tuple[str, bool]]  # [(var_name, is_positive), ...]
    
    def __str__(self):
        parts = []
        for var, is_positive in self.literals:
            if is_positive:
                parts.append(var)
            else:
                parts.append(f"NOT {var}")
        return f"({' OR '.join(parts)})"


@dataclass
class SATPuzzle:
    """Represents a complete SAT puzzle"""
    id: str
    difficulty: str
    domain: str
    variables: List[str]
    clauses: List[SATClause]
    natural_constraints: List[str]
    question: str
    answer: Dict[str, bool]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'difficulty': self.difficulty,
            'domain': self.domain,
            'variables': self.variables,
            'clauses': [[[lit[0], lit[1]] for lit in clause.literals] for clause in self.clauses],
            'constraints': self.natural_constraints
        }
    
    def to_prompt(self) -> str:
        """Generate the puzzle prompt for LLM evaluation"""
        prompt = "You are given a logic puzzle. Determine which statements are true or false.\n\n"
        
        domain_contexts = {
            'crime': "A crime has been committed. Based on the evidence, determine who is guilty.",
            'meeting': "A meeting is being scheduled. Determine who will attend.",
            'task': "Tasks are being assigned to teams. Determine which teams are assigned.",
            'restaurant': "A group is ordering at a restaurant. Determine what will be ordered."
        }
        
        if self.domain in domain_contexts:
            prompt += f"**Context:** {domain_contexts[self.domain]}\n\n"
        
        prompt += f"**Variables:** {', '.join(self.variables)}\n\n"

        prompt += "**Constraints:**\n"
        for i, constraint in enumerate(self.natural_constraints, 1):
            prompt += f"  {i}. {constraint}\n"
        prompt += "\n"

        prompt += "**Rules:**\n"
        prompt += "  - Each variable is either True or False\n"
        prompt += "  - All constraints must be satisfied simultaneously\n\n"

        prompt += f"**Question:** {self.question}\n\n"
        
        prompt += "**Instructions:**\n"
        prompt += "Provide your answer in the following format:\n"
        for var in self.variables:
            prompt += f"- {var}: True/False\n"
        prompt += "\nOr as JSON:\n"
        prompt += "```json\n{\n"
        for i, var in enumerate(self.variables):
            comma = "," if i < len(self.variables) - 1 else ""
            prompt += f'  "{var}": true{comma}  // or false\n'
        prompt += "}\n```\n"
        
        return prompt


def _clause_satisfying_literal_en(clause, assignment) -> str:
    """Pick one literal in the clause that is True under the answer assignment."""
    try:
        literals = getattr(clause, "literals", None)
        if literals is not None:
            for var, is_positive in literals:
                val = assignment.get(var)
                if val is None:
                    continue
                lit_val = val if is_positive else (not val)
                if lit_val:
                    return var if is_positive else f"NOT {var}"
    except Exception:
        pass
    return "?"


def _build_sat_solution_en(puzzle: SATPuzzle) -> str:
    """SFT teacher trace: CNF + satisfying assignment."""
    n_clauses = len(puzzle.clauses)
    n_vars = len(puzzle.variables)
    n_cons = len(puzzle.natural_constraints)
    n_true = sum(1 for v in puzzle.answer.values() if v)
    lines = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {puzzle.difficulty.name.lower()}",
        f"  - Domain: {puzzle.domain}",
        f"  - Variables: {', '.join(puzzle.variables)}",
        f"  - Clauses: {n_clauses} (CNF: conjunction of disjunctions)",
        f"  - Question: {puzzle.question}",
        "  - Final answer is confirmed in [STEP 3]",
        "[STEP 1] Given",
    ]
    for i, c in enumerate(puzzle.natural_constraints, 1):
        lines.append(f"  {i}. {c}")
    lines.append("[STEP 2] Worked solution")
    lines.append(
        f"  · Summary: {n_vars} vars · {n_cons} NL constraints -> {n_clauses} CNF clauses · "
        f"AND(clauses)/OR(literals) structure, find the unique satisfying assignment · "
        f"{n_clauses} SEGs"
    )
    for i, cl in enumerate(puzzle.clauses, 1):
        sat_lit = _clause_satisfying_literal_en(cl, puzzle.answer)
        lines.append(
            f"    [SEG {i}] clause {cl} -> literal '{sat_lit}' is True, so the clause holds."
        )
    lines.append("[STEP 3] Answer and verification")
    lines.append(f"  - Final answer (unique model, True={n_true}/{n_vars}):")
    for v, t in sorted(puzzle.answer.items()):
        lines.append(f"    · {v}: {t}")
    lines.append(
        "  - For each clause, at least one literal should evaluate to True under this assignment."
    )
    for i, cl in enumerate(puzzle.clauses, 1):
        lines.append(f"  - CNF {i}: {cl}")
    return "\n".join(lines)


class SATPuzzleGenerator:
    """Generates SAT puzzles with guaranteed unique solutions"""
    
    # Domain templates
    DOMAINS = {
        'crime': {
            'names': ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 
                     'Iris', 'Jack', 'Kate', 'Leo', 'Mary', 'Nick', 'Olivia'],
            'predicate_true': 'guilty',
            'predicate_false': 'innocent',
            'question_template': 'Who is guilty and who is innocent?'
        },
        'meeting': {
            'names': ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry',
                     'Iris', 'Jack', 'Kate', 'Leo', 'Mary', 'Nick'],
            'predicate_true': 'attending',
            'predicate_false': 'not attending',
            'question_template': 'Who is attending the meeting?'
        },
        'task': {
            'names': ['TeamA', 'TeamB', 'TeamC', 'TeamD', 'TeamE', 'TeamF', 'TeamG', 'TeamH',
                     'TeamI', 'TeamJ', 'TeamK', 'TeamL', 'TeamM', 'TeamN'],
            'predicate_true': 'assigned',
            'predicate_false': 'not assigned',
            'question_template': 'Which teams are assigned to the project?'
        },
        'restaurant': {
            'names': ['Pizza', 'Pasta', 'Salad', 'Burger', 'Soup', 'Steak', 'Sandwich', 'Tacos',
                     'Sushi', 'Curry', 'Noodles', 'Rice', 'Fish', 'Chicken'],
            'predicate_true': 'ordered',
            'predicate_false': 'not ordered',
            'question_template': 'What items will be ordered?'
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def generate(self, difficulty: Difficulty, max_retries: int = 100) -> SATPuzzle:
        """Generate a SAT puzzle of specified difficulty"""
        config = self._get_difficulty_config(difficulty)
        domain = random.choice(list(self.DOMAINS.keys()))

        for attempt in range(max_retries):
            variables, solution = self._generate_solution(config, domain)
            clauses = self._generate_clauses(variables, solution, config)

            # simplified check, not exhaustive
            if self._verify_unique_solution(variables, clauses, solution, config):
                break
        else:
            raise RuntimeError(f"Failed to generate a unique {difficulty.value} SAT puzzle")

        natural_constraints = self._clauses_to_natural_language(clauses, domain)
        puzzle_id = f"sat_{difficulty.lower()}_{random.randint(1000, 9999)}"

        temp_puzzle = SATPuzzle(
            id=puzzle_id,
            difficulty=difficulty,
            domain=domain,
            variables=variables,
            clauses=clauses,
            natural_constraints=natural_constraints,
            question=self.DOMAINS[domain]['question_template'],
            answer=solution
        )
        
        complete_prompt = temp_puzzle.to_prompt()
        temp_puzzle.question = complete_prompt
        
        return temp_puzzle
    
    def _get_difficulty_config(self, difficulty: Difficulty) -> dict:
        """Get configuration parameters for each difficulty level"""
        configs = {
            Difficulty.EASY: {
                # Target: ~75%. Previous setting scored 83%, so nudge the state
                # space up while keeping prompts shorter than medium.
                'num_vars': 9,
                'min_clauses': 36,
                'max_clauses': 54,
                'clause_length': (3, 4),
                'unit_clause_rate': 0.0,
                'negation_ratio': 0.51,
            },
            Difficulty.MEDIUM: {
                # Target: ~50%. Previous setting scored 46%, so keep 11 variables
                # but trim noise slightly to recover a few points.
                'num_vars': 11,
                'min_clauses': 56,
                'max_clauses': 88,
                'clause_length': (3, 4),
                'unit_clause_rate': 0.0,
                'negation_ratio': 0.54,
            },
            Difficulty.HARD: {
                # Target: ~25%. The 14-variable / 130+ clause setting scored 16%,
                # so keep the variable count and trim clause noise slightly.
                'num_vars': 14,
                'min_clauses': 115,
                'max_clauses': 170,
                'clause_length': (3, 4),
                'unit_clause_rate': 0.0,
                'negation_ratio': 0.57,
            }
        }
        return configs[difficulty]
    
    def _generate_solution(self, config: dict, domain: str) -> Tuple[List[str], Dict[str, bool]]:
        """Generate a random solution (variable assignment)"""
        num_vars = config['num_vars']
        available_names = self.DOMAINS[domain]['names']
        
        variables = random.sample(available_names, num_vars)
        solution = {var: random.choice([True, False]) for var in variables}
        
        return variables, solution
    
    def _generate_clauses(
        self,
        variables: List[str],
        solution: Dict[str, bool],
        config: dict
    ) -> List[SATClause]:
        """Generate random satisfied CNF clauses with a unique target solution."""
        clauses: List[SATClause] = []
        used_clauses = set()

        def append_clause(literals: List[Tuple[str, bool]]) -> bool:
            clause_sig = tuple(sorted(literals))
            if clause_sig in used_clauses:
                return False
            if not self._eval_clause(literals, solution):
                return False
            used_clauses.add(clause_sig)
            clauses.append(SATClause(literals=literals))
            return True

        attempts = 0
        max_attempts = config['max_clauses'] * 500

        while attempts < max_attempts and len(clauses) < config['max_clauses']:
            attempts += 1
            literals = self._generate_random_satisfied_clause(variables, solution, config)
            append_clause(literals)
            if len(clauses) >= config['min_clauses'] and self._count_solutions(variables, clauses, stop_after=2) == 1:
                break
        
        return clauses

    def _solution_literal(self, var: str, solution: Dict[str, bool]) -> Tuple[str, bool]:
        """Return the literal that is true for var under the target solution."""
        return (var, solution[var])

    def _negate_literal(self, literal: Tuple[str, bool]) -> Tuple[str, bool]:
        """Return the logical negation of a literal."""
        var, is_positive = literal
        return (var, not is_positive)

    def _generate_random_satisfied_clause(
        self,
        variables: List[str],
        solution: Dict[str, bool],
        config: dict,
    ) -> List[Tuple[str, bool]]:
        """Generate one random clause that is satisfied by the target solution."""
        min_len, max_len = config['clause_length']
        if random.random() < config['unit_clause_rate']:
            clause_len = 1
        else:
            clause_len = random.randint(min_len, min(max_len, len(variables)))

        selected_vars = random.sample(variables, clause_len)
        literals = []
        has_true_literal = False

        for var in selected_vars:
            is_positive = not solution[var] if random.random() < config['negation_ratio'] else solution[var]
            if self._eval_literal(var, is_positive, solution):
                has_true_literal = True
            literals.append((var, is_positive))

        if not has_true_literal:
            idx = random.randrange(len(literals))
            var, _ = literals[idx]
            literals[idx] = self._solution_literal(var, solution)

        random.shuffle(literals)
        return literals
    
    def _eval_literal(self, var: str, is_positive: bool, solution: Dict[str, bool]) -> bool:
        """Evaluate a literal given the solution"""
        var_value = solution[var]
        return var_value if is_positive else not var_value
    
    def _eval_clause(self, literals: List[Tuple[str, bool]], solution: Dict[str, bool]) -> bool:
        """Evaluate a clause (OR of literals)"""
        return any(self._eval_literal(var, is_pos, solution) 
                  for var, is_pos in literals)

    def _count_solutions(
        self,
        variables: List[str],
        clauses: List[SATClause],
        stop_after: Optional[int] = None,
    ) -> int:
        """Count satisfying assignments, optionally stopping early."""
        num_solutions = 0

        for i in range(2 ** len(variables)):
            assignment = {}
            for j, var in enumerate(variables):
                assignment[var] = bool((i >> j) & 1)

            if all(self._eval_clause(clause.literals, assignment) for clause in clauses):
                num_solutions += 1
                if stop_after is not None and num_solutions >= stop_after:
                    return num_solutions

        return num_solutions
    
    def _verify_unique_solution(
        self,
        variables: List[str],
        clauses: List[SATClause],
        expected_solution: Dict[str, bool],
        config: dict
    ) -> bool:
        """
        Verify that the clauses lead to a unique solution.
        Brute-force solution counting is feasible because calibrated SAT puzzles
        are capped at fourteen variables.
        """
        return self._count_solutions(variables, clauses, stop_after=2) == 1
    
    def _clauses_to_natural_language(
        self,
        clauses: List[SATClause],
        domain: str
    ) -> List[str]:
        """Convert logical clauses to natural language"""
        domain_info = self.DOMAINS[domain]
        pred_true = domain_info['predicate_true']
        pred_false = domain_info['predicate_false']
        
        natural = []
        
        for clause in clauses:
            nl_clause = self._clause_to_english(clause, pred_true, pred_false)
            natural.append(nl_clause)
        
        return natural
    
    def _clause_to_english(
        self,
        clause: SATClause,
        pred_true: str,
        pred_false: str
    ) -> str:
        """Convert a single clause to English"""
        literals = clause.literals
        
        # Special case: single literal
        if len(literals) == 1:
            var, is_pos = literals[0]
            if is_pos:
                return f"{var} is {pred_true}"
            else:
                return f"{var} is {pred_false}"
        
        # Special case: two literals with negations (implication pattern)
        if len(literals) == 2:
            var1, is_pos1 = literals[0]
            var2, is_pos2 = literals[1]
            
            # Pattern: (NOT A OR B) = "If A then B"
            if not is_pos1 and is_pos2:
                return f"If {var1} is {pred_true}, then {var2} is {pred_true}"
            
            # Pattern: (A OR NOT B) = "If B then A"
            if is_pos1 and not is_pos2:
                return f"If {var2} is {pred_true}, then {var1} is {pred_true}"
            
            # Pattern: (NOT A OR NOT B) = "A and B cannot both be true"
            if not is_pos1 and not is_pos2:
                return f"{var1} and {var2} cannot both be {pred_true}"
            
            # Pattern: (A OR B) = "At least one is true"
            if is_pos1 and is_pos2:
                return f"At least one of {var1} or {var2} is {pred_true}"
        
        # General case: multiple literals
        positive_vars = [var for var, is_pos in literals if is_pos]
        negative_vars = [var for var, is_pos in literals if not is_pos]
        
        if len(positive_vars) > 0 and len(negative_vars) == 0:
            # All positive: "At least one of X, Y, Z is true"
            if len(positive_vars) == 2:
                return f"At least one of {positive_vars[0]} or {positive_vars[1]} is {pred_true}"
            else:
                vars_str = ', '.join(positive_vars[:-1]) + f', or {positive_vars[-1]}'
                return f"At least one of {vars_str} is {pred_true}"
        
        # Mixed or all negative: describe as disjunction
        parts = []
        for var, is_pos in literals:
            if is_pos:
                parts.append(f"{var} is {pred_true}")
            else:
                parts.append(f"{var} is {pred_false}")
        
        if len(parts) == 2:
            return f"Either {parts[0]}, or {parts[1]}"
        else:
            return f"At least one of the following is true: {', or '.join(parts)}"


def generate_dataset(
    num_samples: int,
    seed: Optional[int] = None
):
    """Generate a dataset of SAT puzzles"""
    import os
    from pathlib import Path
    
    generator = SATPuzzleGenerator(seed=seed)
    puzzles = []
    
    # Generate balanced dataset - 쉬운 난이도부터 어려운 순서로 생성
    per_difficulty = num_samples // 3
    remaining = num_samples - (per_difficulty * 3)
    
    # 쉬운 순서로 생성 (EASY -> MEDIUM -> HARD)
    difficulties = [Difficulty.EASY] * per_difficulty + \
                  [Difficulty.MEDIUM] * per_difficulty + \
                  [Difficulty.HARD] * (per_difficulty + remaining)
    
    print(f"Generating {num_samples} SAT puzzles...")
    
    for i, difficulty in enumerate(difficulties, 1):
        puzzle = generator.generate(difficulty)
        puzzles.append(puzzle)
        
        if i % 10 == 0:
            print(f"Generated {i}/{num_samples} puzzles...")
    
    # Re-assign ids to follow per-difficulty naming convention
    diff_counters = {}
    for puzzle in puzzles:
        diff_name = getattr(puzzle.difficulty, "value", puzzle.difficulty)
        diff_name = str(diff_name).lower()
        diff_idx = diff_counters.get(diff_name, 0)
        diff_counters[diff_name] = diff_idx + 1
        puzzle.id = f'sat_puzzle_en_{diff_name}_{diff_idx:04d}'
    
    def _row(p: SATPuzzle) -> dict:
        return {
            "id": p.id,
            "question": p.question,
            # answer 는 변수→진리값 매핑이지만, 변수명이 문제마다 달라
            # dict 그대로 두면 CSV/Arrow 스키마가 폭발한다. JSON 문자열로 고정.
            "answer": json.dumps(p.answer, ensure_ascii=False, sort_keys=True),
            "solution": _build_sat_solution_en(p),
            "difficulty": p.difficulty,
            "variables": p.variables,
            "clauses": [[[lit[0], lit[1]] for lit in clause.literals] for clause in p.clauses],
        }

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("sat_puzzles_en", [_row(p) for p in puzzles])
    print(f"\n✅ Dataset created successfully!")
    print(f"   Total puzzles: {num_samples}")
    
    easy_count = sum(1 for p in puzzles if p.difficulty == Difficulty.EASY)
    medium_count = sum(1 for p in puzzles if p.difficulty == Difficulty.MEDIUM)
    hard_count = sum(1 for p in puzzles if p.difficulty == Difficulty.HARD)
    
    print(f"   Difficulty breakdown:")
    print(f"     - Easy: {easy_count}")
    print(f"     - Medium: {medium_count}")
    print(f"     - Hard: {hard_count}")


def main():
    parser = argparse.ArgumentParser(description="Generate SAT Puzzles")
    parser.add_argument('--num-samples', type=int, default=150,
                       help='Number of puzzles to generate')
    parser.add_argument('--output-dir', type=str, default='data/sat',
                       help='Output directory for the dataset')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--example', action='store_true',
                       help='Generate and print example puzzles')
    
    args = parser.parse_args()
    
    if args.example:
        print("\n" + "="*70)
        print("SAT PUZZLE EXAMPLES")
        print("="*70 + "\n")
        
        generator = SATPuzzleGenerator(seed=42)
        
        for difficulty in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
            puzzle = generator.generate(difficulty)
            
            print(f"\n{'='*70}")
            print(f"{difficulty.upper()} EXAMPLE")
            print(f"{'='*70}")
            print(puzzle.question)
            print(f"✅ **Correct Answer:**")
            for var, value in puzzle.answer.items():
                print(f"   {var}: {value}")
            print()
    else:
        generate_dataset(args.num_samples, args.seed)


if __name__ == "__main__":
    main()

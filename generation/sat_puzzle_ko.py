#!/usr/bin/env python3
"""
Boolean SAT (Satisfiability) Puzzle Generator

CNF(Conjunctive Normal Form) 형식의 논리 퍼즐을 자연어로 생성합니다.
SAT 솔버로 유일 해를 보장합니다.
"""

import random
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from itertools import combinations


SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class SATClause:
    """단일 절(리터럴의 논리합)을 나타냄"""
    literals: List[Tuple[str, bool]]  # [(변수명, 긍정인지), ...]
    
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
    """완전한 SAT 퍼즐을 나타냄"""
    id: str
    difficulty: str
    domain: str
    variables: List[str]
    clauses: List[SATClause]
    natural_constraints: List[str]
    question: str
    answer: Dict[str, bool]
    
    def to_dict(self) -> dict:
        """JSON 직렬화를 위한 딕셔너리 변환"""
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
        """LLM 평가를 위한 퍼즐 프롬프트 생성"""
        prompt = "논리 퍼즐이 주어집니다. 어떤 진술이 참이고 거짓인지 판단하세요.\n\n"
        
        domain_contexts = {
            'crime': "범죄가 발생했습니다. 증거를 바탕으로 누가 유죄인지 판단하세요.",
            'meeting': "회의 일정이 잡히고 있습니다. 누가 참석하는지 판단하세요.",
            'task': "팀에게 작업이 할당되고 있습니다. 어떤 팀이 할당되는지 판단하세요.",
            'restaurant': "단체가 식당에서 주문하고 있습니다. 무엇이 주문될지 판단하세요."
        }
        
        if self.domain in domain_contexts:
            prompt += f"**상황:** {domain_contexts[self.domain]}\n\n"
        
        prompt += f"**변수:** {', '.join(self.variables)}\n\n"

        prompt += "**제약 조건:**\n"
        for i, constraint in enumerate(self.natural_constraints, 1):
            prompt += f"  {i}. {constraint}\n"
        prompt += "\n"

        prompt += "**규칙:**\n"
        prompt += "  - 각 변수는 참(True) 또는 거짓(False)입니다\n"
        prompt += "  - 모든 제약 조건은 동시에 만족되어야 합니다\n\n"

        prompt += f"**질문:** {self.question}\n\n"
        
        prompt += "**지시사항:**\n"
        prompt += "답변을 다음 형식으로 제공하세요:\n"
        for var in self.variables:
            prompt += f"- {var}: 참/거짓\n"
        prompt += "\n또는 JSON으로:\n"
        prompt += "```json\n{\n"
        for i, var in enumerate(self.variables):
            comma = "," if i < len(self.variables) - 1 else ""
            prompt += f'  "{var}": true{comma}  // 또는 false\n'
        prompt += "}\n```\n"
        
        return prompt


def _clause_satisfying_literal_ko(clause, assignment) -> str:
    """절에서 정답을 참으로 만드는 리터럴 하나를 골라 반환."""
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


def _build_sat_solution_ko(puzzle: SATPuzzle) -> str:
    """SFT teacher trace: CNF + 만족 할당."""
    n_clauses = len(puzzle.clauses)
    n_vars = len(puzzle.variables)
    n_cons = len(puzzle.natural_constraints)
    n_true = sum(1 for v in puzzle.answer.values() if v)
    lines = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {puzzle.difficulty.name.lower()}",
        f"  - 도메인(시나리오): {puzzle.domain}",
        f"  - 변수: {', '.join(puzzle.variables)}",
        f"  - 절(clause) 개수: {n_clauses} (CNF: 모든 절이 동시에 참이어야 함)",
        f"  - 질문: {puzzle.question}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
    ]
    for i, c in enumerate(puzzle.natural_constraints, 1):
        lines.append(f"  {i}. {c}")
    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: 변수 {n_vars} · 자연어 제약 {n_cons} → CNF 절 {n_clauses} · "
        f"AND(절) / OR(리터럴) 구조에서 유일 할당 찾기 · SEG {n_clauses}개"
    )
    for i, cl in enumerate(puzzle.clauses, 1):
        sat_lit = _clause_satisfying_literal_ko(cl, puzzle.answer)
        lines.append(
            f"    [SEG {i}] 절 {cl} → 리터럴 '{sat_lit}'가 참이므로 절 성립."
        )
    lines.append("[STEP 3] 답·검산")
    lines.append(f"  - 최종 답(유일 모델, True={n_true}/{n_vars}):")
    for v, t in sorted(puzzle.answer.items()):
        lines.append(f"    · {v}: {t} ({'참' if t else '거짓'})")
    lines.append(
        "  - 각 절에 정답을 대입: 절에 속한 리터럴 중 최소 하나는 참이어야 한다."
    )
    for i, cl in enumerate(puzzle.clauses, 1):
        lines.append(f"  - CNF {i}: {cl}")
    return "\n".join(lines)


class SATPuzzleGenerator:
    """유일 해를 보장하는 SAT 퍼즐 생성"""
    
    # 도메인 템플릿
    DOMAINS = {
        'crime': {
            'names': ['지민', '태희', '준호', '수연', '민석', '하연', '동욱', '서현', 
                     '영호', '지우', '현수', '나영', '재훈', '소희', '진우'],
            'predicate_true': '유죄',
            'predicate_false': '무죄',
            'question_template': '누가 유죄이고 누가 무죄입니까?'
        },
        'meeting': {
            'names': ['지민', '태희', '준호', '수연', '민석', '하연', '동욱', '서현',
                     '영호', '지우', '현수', '나영', '재훈', '소희'],
            'predicate_true': '참석',
            'predicate_false': '불참',
            'question_template': '누가 회의에 참석합니까?'
        },
        'task': {
            'names': ['A팀', 'B팀', 'C팀', 'D팀', 'E팀', 'F팀', 'G팀', 'H팀',
                     'I팀', 'J팀', 'K팀', 'L팀', 'M팀', 'N팀'],
            'predicate_true': '할당됨',
            'predicate_false': '미할당',
            'question_template': '어떤 팀이 프로젝트에 할당됩니까?'
        },
        'restaurant': {
            'names': ['피자', '파스타', '샐러드', '버거', '국', '스테이크', '샌드위치', '타코',
                     '초밥', '카레', '국수', '밥', '생선', '치킨'],
            'predicate_true': '주문됨',
            'predicate_false': '주문안됨',
            'question_template': '어떤 음식이 주문됩니까?'
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def generate(self, difficulty: Difficulty, max_retries: int = 100) -> SATPuzzle:
        """지정된 난이도의 SAT 퍼즐 생성"""
        config = self._get_difficulty_config(difficulty)
        domain = random.choice(list(self.DOMAINS.keys()))

        for attempt in range(max_retries):
            variables, solution = self._generate_solution(config, domain)
            clauses = self._generate_clauses(variables, solution, config)

            # 단순화된 검사, 완전 검증 아님
            if self._verify_unique_solution(variables, clauses, solution, config):
                break
        else:
            raise RuntimeError(f"유일한 해를 갖는 {difficulty.value} SAT 퍼즐 생성 실패")

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
        """각 난이도에 대한 설정 매개변수 가져오기"""
        configs = {
            Difficulty.EASY: {
                # 목표: 약 75%. 이전 설정이 83%였으므로, 프롬프트는 medium보다
                # 짧게 유지하면서 상태 공간을 조금 키운다.
                'num_vars': 9,
                'min_clauses': 36,
                'max_clauses': 54,
                'clause_length': (3, 4),
                'unit_clause_rate': 0.0,
                'negation_ratio': 0.51,
            },
            Difficulty.MEDIUM: {
                # 목표: 약 50%. 이전 설정이 46%였으므로, 변수 11개는 유지하고
                # 노이즈를 약간 줄여 몇 점 회복한다.
                'num_vars': 11,
                'min_clauses': 56,
                'max_clauses': 88,
                'clause_length': (3, 4),
                'unit_clause_rate': 0.0,
                'negation_ratio': 0.54,
            },
            Difficulty.HARD: {
                # 목표: 약 25%. 14변수 / 130개 이상 절 설정이 16%였으므로,
                # 변수 수는 유지하고 절 노이즈만 약간 줄인다.
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
        """무작위 해답(변수 할당) 생성"""
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
        """목표 해답을 만족하는 무작위 CNF 절 생성"""
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
        """목표 해답에서 해당 변수를 참으로 만드는 리터럴 반환"""
        return (var, solution[var])

    def _negate_literal(self, literal: Tuple[str, bool]) -> Tuple[str, bool]:
        """리터럴의 논리 부정 반환"""
        var, is_positive = literal
        return (var, not is_positive)

    def _generate_random_satisfied_clause(
        self,
        variables: List[str],
        solution: Dict[str, bool],
        config: dict,
    ) -> List[Tuple[str, bool]]:
        """목표 해답을 만족하는 무작위 절 하나 생성"""
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
        """해답이 주어졌을 때 리터럴 평가"""
        var_value = solution[var]
        return var_value if is_positive else not var_value
    
    def _eval_clause(self, literals: List[Tuple[str, bool]], solution: Dict[str, bool]) -> bool:
        """절(리터럴의 논리합) 평가"""
        return any(self._eval_literal(var, is_pos, solution) 
                  for var, is_pos in literals)

    def _count_solutions(
        self,
        variables: List[str],
        clauses: List[SATClause],
        stop_after: Optional[int] = None,
    ) -> int:
        """만족하는 할당 수를 세되, 필요하면 조기 중단"""
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
        절이 유일 해로 이어지는지 검증.
        보정된 SAT 퍼즐은 변수 14개로 제한되므로 전수 해 개수 세기가 가능합니다.
        """
        return self._count_solutions(variables, clauses, stop_after=2) == 1
    
    def _clauses_to_natural_language(
        self,
        clauses: List[SATClause],
        domain: str
    ) -> List[str]:
        """논리 절을 자연어로 변환"""
        domain_info = self.DOMAINS[domain]
        pred_true = domain_info['predicate_true']
        pred_false = domain_info['predicate_false']
        
        natural = []
        
        for clause in clauses:
            nl_clause = self._clause_to_korean(clause, pred_true, pred_false)
            natural.append(nl_clause)
        
        return natural
    
    def _clause_to_korean(
        self,
        clause: SATClause,
        pred_true: str,
        pred_false: str
    ) -> str:
        """단일 절을 한국어로 변환"""
        literals = clause.literals
        
        # 특수 케이스: 단일 리터럴
        if len(literals) == 1:
            var, is_pos = literals[0]
            if is_pos:
                return f"{var}은(는) {pred_true}이다"
            else:
                return f"{var}은(는) {pred_false}이다"
        
        # 특수 케이스: 부정이 있는 두 리터럴 (함의 패턴)
        if len(literals) == 2:
            var1, is_pos1 = literals[0]
            var2, is_pos2 = literals[1]
            
            # 패턴: (NOT A OR B) = "A이면 B이다"
            if not is_pos1 and is_pos2:
                return f"{var1}이(가) {pred_true}이면, {var2}도 {pred_true}이다"
            
            # 패턴: (A OR NOT B) = "B이면 A이다"
            if is_pos1 and not is_pos2:
                return f"{var2}이(가) {pred_true}이면, {var1}도 {pred_true}이다"
            
            # 패턴: (NOT A OR NOT B) = "A와 B가 둘 다 참일 수 없다"
            if not is_pos1 and not is_pos2:
                return f"{var1}과(와) {var2}이(가) 둘 다 {pred_true}일 수는 없다"
            
            # 패턴: (A OR B) = "최소 하나는 참이다"
            if is_pos1 and is_pos2:
                return f"{var1} 또는 {var2} 중 최소 하나는 {pred_true}이다"
        
        # 일반 케이스: 다중 리터럴
        positive_vars = [var for var, is_pos in literals if is_pos]
        negative_vars = [var for var, is_pos in literals if not is_pos]
        
        if len(positive_vars) > 0 and len(negative_vars) == 0:
            # 모두 긍정: "X, Y, Z 중 최소 하나는 참이다"
            if len(positive_vars) == 2:
                return f"{positive_vars[0]} 또는 {positive_vars[1]} 중 최소 하나는 {pred_true}이다"
            else:
                vars_str = ', '.join(positive_vars[:-1]) + f' 또는 {positive_vars[-1]}'
                return f"{vars_str} 중 최소 하나는 {pred_true}이다"
        
        # 혼합 또는 모두 부정: 논리합으로 설명
        parts = []
        for var, is_pos in literals:
            if is_pos:
                parts.append(f"{var}은(는) {pred_true}")
            else:
                parts.append(f"{var}은(는) {pred_false}")
        
        if len(parts) == 2:
            return f"{parts[0]}이거나, {parts[1]}이다"
        else:
            return f"다음 중 최소 하나는 참이다: {' 또는 '.join(parts)}"


def generate_dataset(
    num_samples: int,
    seed: Optional[int] = None
):
    """SAT 퍼즐 데이터셋 생성"""
    import os
    from pathlib import Path
    
    generator = SATPuzzleGenerator(seed=seed)
    puzzles = []
    
    # 균형 잡힌 데이터셋 생성 - 쉬운 난이도부터 어려운 순서로 생성
    per_difficulty = num_samples // 3
    remaining = num_samples - (per_difficulty * 3)
    
    # 쉬운 순서로 생성 (EASY -> MEDIUM -> HARD)
    difficulties = [Difficulty.EASY] * per_difficulty + \
                  [Difficulty.MEDIUM] * per_difficulty + \
                  [Difficulty.HARD] * (per_difficulty + remaining)
    
    print(f"{num_samples}개의 SAT 퍼즐 생성 중...")
    
    for i, difficulty in enumerate(difficulties, 1):
        puzzle = generator.generate(difficulty)
        puzzles.append(puzzle)
        
        if i % 10 == 0:
            print(f"{i}/{num_samples} 퍼즐 생성 완료...")
    
    # 난이도별 명명 규칙에 맞게 id 재할당
    diff_counters = {}
    for puzzle in puzzles:
        diff_name = getattr(puzzle.difficulty, "value", puzzle.difficulty)
        diff_name = str(diff_name).lower()
        diff_idx = diff_counters.get(diff_name, 0)
        diff_counters[diff_name] = diff_idx + 1
        puzzle.id = f'sat_puzzle_ko_{diff_name}_{diff_idx:04d}'
    
    def _row(p: SATPuzzle) -> dict:
        return {
            "id": p.id,
            "question": p.question,
            # answer 는 변수→진리값 매핑이지만, 변수명이 문제마다 달라
            # dict 그대로 두면 CSV/Arrow 스키마가 폭발한다. JSON 문자열로 고정.
            "answer": json.dumps(p.answer, ensure_ascii=False, sort_keys=True),
            "solution": _build_sat_solution_ko(p),
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

    write_release_jsonl("sat_puzzles_ko", [_row(p) for p in puzzles])
    print(f"\n✅ 데이터셋 생성 완료!")
    print(f"   총 퍼즐 수: {num_samples}")
    
    easy_count = sum(1 for p in puzzles if p.difficulty == Difficulty.EASY)
    medium_count = sum(1 for p in puzzles if p.difficulty == Difficulty.MEDIUM)
    hard_count = sum(1 for p in puzzles if p.difficulty == Difficulty.HARD)
    
    print(f"   난이도 분포:")
    print(f"     - Easy: {easy_count}")
    print(f"     - Medium: {medium_count}")
    print(f"     - Hard: {hard_count}")


def main():
    parser = argparse.ArgumentParser(description="SAT 퍼즐 생성기 (한국어)")
    parser.add_argument('--num-samples', type=int, default=150,
                       help='생성할 퍼즐 수')
    parser.add_argument('--output-dir', type=str, default='data/sat_ko',
                       help='데이터셋 출력 디렉터리')
    parser.add_argument('--seed', type=int, default=None,
                       help='재현성을 위한 랜덤 시드')
    parser.add_argument('--example', action='store_true',
                       help='예제 퍼즐 생성 및 출력')
    
    args = parser.parse_args()
    
    if args.example:
        print("\n" + "="*70)
        print("SAT 퍼즐 예제")
        print("="*70 + "\n")
        
        generator = SATPuzzleGenerator(seed=42)
        
        for difficulty in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
            puzzle = generator.generate(difficulty)
            
            print(f"\n{'='*70}")
            print(f"{difficulty.upper()} 예제")
            print(f"{'='*70}")
            print(puzzle.question)
            print(f"✅ **정답:**")
            for var, value in puzzle.answer.items():
                print(f"   {var}: {value}")
            print()
    else:
        generate_dataset(args.num_samples, args.seed)


if __name__ == "__main__":
    main()

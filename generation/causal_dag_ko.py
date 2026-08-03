"""Causal DAG (Directed Acyclic Graph) Reasoning Puzzle Generator

사건 사슬과 시간 전파를 기반으로 인과 추론 퍼즐을 생성합니다.
LLM의 시간에 따른 인과관계 추론 능력을 평가합니다.

주요 기능:
1. DAG 기반 사건 그래프 (순환 없음)
2. 시간 지연 인과 관계
3. 최단 경로 추론 (Dijkstra)
4. 유일 해 보장 (결정적 그래프)
"""

import random
import heapq
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class EventType(Enum):
    """퍼즐 생성용 사건 카테고리"""
    TECHNICAL = "기술"
    BUSINESS = "비즈니스"
    ENVIRONMENTAL = "환경"
    OPERATIONAL = "운영"


@dataclass
class CausalEdge:
    """사건 간 인과 관계를 나타냄"""
    from_event: str
    to_event: str
    delay: int  # 분 단위 시간 지연
    from_events: Optional[List[str]] = None  # 다중 전제 조건 (AND용)
    condition: str = 'OR'  # 'OR' 또는 'AND'
    
    def __repr__(self):
        if self.from_events and len(self.from_events) > 1:
            cond = ' 그리고 ' if self.condition == 'AND' else ' 또는 '
            return f"[{cond.join(self.from_events)}] → {self.to_event} (+{self.delay}분)"
        return f"{self.from_event} → {self.to_event} (+{self.delay}분)"


@dataclass
class Event:
    """인과 그래프의 사건을 나타냄"""
    id: str
    name: str
    description: str
    event_type: EventType
    
    def __repr__(self):
        return f"{self.id}: {self.name}"


@dataclass
class CausalPuzzle:
    """완전한 인과 추론 퍼즐"""
    events: Dict[str, Event]
    edges: List[CausalEdge]
    trigger: str
    trigger_time: int
    target_event: str
    answer: int
    difficulty: str
    query_type: str = 'occurrence_time'
    compare_event: Optional[str] = None
    shuffle_edges: bool = False
    
    def to_dict(self):
        """JSON 직렬화를 위한 딕셔너리 변환"""
        return {
            'events': {k: {
                'id': v.id,
                'name': v.name,
                'description': v.description,
                'event_type': v.event_type.value
            } for k, v in self.events.items()},
            'edges': [{
                'from_event': e.from_event,
                'to_event': e.to_event,
                'delay': e.delay,
                'from_events': e.from_events,
                'condition': e.condition
            } for e in self.edges],
            'trigger': self.trigger,
            'trigger_time': self.trigger_time,
            'target': self.target_event,
            'answer': self.answer,
            'difficulty': self.difficulty,
            'query_type': self.query_type,
            'compare_event': self.compare_event,
            'shuffle_edges': self.shuffle_edges,
        }


class CausalPuzzleGenerator:
    """인과 DAG 추론 퍼즐 생성"""
    
    def __init__(self):
        """사건 템플릿으로 초기화"""
        self.event_templates = {
            EventType.TECHNICAL: [
                ('전력차단', '주 전력망 장애 발생'),
                ('서버다운', '애플리케이션 서버 중단'),
                ('데이터베이스장애', '데이터베이스 서비스 응답 중지'),
                ('네트워크장애', '네트워크 연결 끊김'),
                ('디스크포화', '저장 공간 100% 도달'),
                ('메모리누수', '애플리케이션 메모리 사용량 급증'),
                ('백업실패', '자동 백업 프로세스 실패'),
                ('보안침해', '무단 접근 탐지'),
                ('API타임아웃', '외부 API 응답 중지'),
                ('캐시만료', '캐시 무효화 발생'),
            ],
            EventType.BUSINESS: [
                ('주문접수', '고객 신규 주문 발생'),
                ('결제완료', '결제 트랜잭션 완료'),
                ('재고부족', '재고 수준이 임계치 이하로 하락'),
                ('배송지연', '배송 일정 연기'),
                ('고객불만', '고객 지원 티켓 생성'),
                ('환불처리', '고객에게 금액 반환'),
                ('가격변경', '제품 가격 업데이트'),
                ('프로모션시작', '마케팅 캠페인 시작'),
                ('계약체결', '법적 계약 최종 확정'),
                ('청구서발송', '청구 문서 생성'),
            ],
            EventType.ENVIRONMENTAL: [
                ('폭우', '시간당 50mm 이상 강수량'),
                ('도로침수', '수위 상승으로 교통 차단'),
                ('교통체증', '차량 정체 발생'),
                ('전력급등', '전력망 전압 급등'),
                ('지진', '지진 활동 감지'),
                ('폭염', '기온 35도 이상'),
                ('폭풍경보', '기상 경보 발령'),
                ('폭설', '적설량 누적 시작'),
                ('강풍피해', '강풍으로 인한 기반 시설 손상'),
                ('가뭄', '수자원 부족'),
            ],
            EventType.OPERATIONAL: [
                ('정기점검', '계획된 시스템 유지보수 시작'),
                ('인력부족', '가용 인력이 최소치 이하'),
                ('설비고장', '핵심 설비 작동 중단'),
                ('품질문제', '제품 결함 발견'),
                ('공급망중단', '협력사 납품 지연'),
                ('용량초과', '최대 처리량 초과'),
                ('정책변경', '신규 운영 규정 시행'),
                ('검사불합격', '규정 준수 검사 미통과'),
                ('교육완료', '직원 자격 인증 취득'),
                ('시스템업그레이드', '소프트웨어 버전 업데이트'),
            ]
        }
    
    def generate_puzzle(self, difficulty: str, seed: Optional[int] = None) -> CausalPuzzle:
        """
        인과관계 추론 퍼즐 생성
        
        Args:
            difficulty: 'easy', 'medium', 또는 'hard'
            seed: 재현성을 위한 난수 시드
        
        Returns:
            유일 해를 갖는 CausalPuzzle
        """
        if seed is not None:
            random.seed(seed)
        
        # gemini-3-flash-preview 실행 결과에 맞춰 보정된 난이도 설정.
        # 주요 난이도 축은 관계 제시 순서 섞기, 대체 인과 규칙,
        # 더 깊은 타겟, 단일 경로보다 넓은 전파를 요구하는 질의 유형이다.
        config = {
            'easy': {
                'num_events': random.randint(25, 31),
                'edge_density': 0.62,
                'delay_range': (10, 60),
                'max_out_degree': 3,
                'and_probability': 0.47,
                'target_quantile': (0.56, 0.86),
                'query_weights': {
                    'occurrence_time': 0.30,
                    'time_gap': 0.25,
                    'count_by_target_time': 0.45,
                },
                'shuffle_edges': True,
            },
            'medium': {
                'num_events': random.randint(40, 46),
                'edge_density': 0.88,
                'delay_range': (20, 140),
                'max_out_degree': 4,
                'and_probability': 0.63,
                'target_quantile': (0.70, 0.94),
                'query_weights': {
                    'occurrence_time': 0.03,
                    'time_gap': 0.22,
                    'count_by_target_time': 0.75,
                },
                'shuffle_edges': True,
            },
            'hard': {
                'num_events': random.randint(46, 58),
                'edge_density': 0.95,
                'delay_range': (30, 240),
                'max_out_degree': 5,
                'and_probability': 0.64,
                'target_quantile': (0.64, 0.94),
                'query_weights': {
                    'time_gap': 0.20,
                    'count_by_target_time': 0.80,
                },
                'shuffle_edges': True,
            }
        }[difficulty]
        
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                events = self._generate_events(config['num_events'])
                edges = self._generate_causal_graph(events, config)
                
                if not edges:
                    continue
                
                # 트리거 선택 (진입 차수 0인 노드)
                in_degree = self._calculate_in_degree(events, edges)
                possible_triggers = [e_id for e_id, degree in in_degree.items() 
                                    if degree == 0]
                
                if not possible_triggers:
                    continue
                
                trigger = random.choice(possible_triggers)
                trigger_time = random.randint(0, 60)
                
                reach_times = self._calculate_reach_times(events, edges, trigger, trigger_time)
                
                # 타겟/질의 선택 (도달 가능한 사건, 트리거 제외)
                reachable = [e_id for e_id, time in reach_times.items() 
                           if time < float('inf') and e_id != trigger]
                
                if not reachable:
                    continue
                
                query_type, target_event, answer, compare_event = self._select_query(
                    reachable,
                    reach_times,
                    config
                )
                
                return CausalPuzzle(
                    events=events,
                    edges=edges,
                    trigger=trigger,
                    trigger_time=trigger_time,
                    target_event=target_event,
                    answer=answer,
                    difficulty=difficulty,
                    query_type=query_type,
                    compare_event=compare_event,
                    shuffle_edges=config.get('shuffle_edges', False),
                )
            
            except Exception as e:
                continue
        
        # 모든 시도 실패 시 단순 선형 체인 생성
        return self._generate_simple_puzzle(difficulty)

    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        """작은 가중치 딕셔너리에서 키 하나를 선택한다."""
        total = sum(weights.values())
        threshold = random.random() * total
        cumulative = 0.0
        for key, weight in weights.items():
            cumulative += weight
            if threshold <= cumulative:
                return key
        return next(reversed(weights))

    def _select_query(
        self,
        reachable: List[str],
        reach_times: Dict[str, int],
        config: Dict,
    ) -> Tuple[str, str, int, Optional[str]]:
        """질의 유형을 선택하고 해당 답을 계산한다."""
        target_candidates = sorted(reachable, key=lambda e: (reach_times[e], e))
        q_start, q_end = config.get('target_quantile', (1 / 3, 2 / 3))
        start_idx = int(len(target_candidates) * q_start)
        end_idx = max(start_idx + 1, int(len(target_candidates) * q_end))
        target_pool = target_candidates[start_idx:end_idx] or target_candidates
        target_event = random.choice(target_pool)
        query_type = self._weighted_choice(config.get('query_weights', {'occurrence_time': 1.0}))

        if query_type == 'time_gap':
            earlier = [
                e for e in target_candidates
                if e != target_event and reach_times[e] < reach_times[target_event]
            ]
            if earlier:
                compare_event = random.choice(earlier)
                answer = reach_times[target_event] - reach_times[compare_event]
                return query_type, target_event, int(answer), compare_event
            query_type = 'occurrence_time'

        if query_type == 'count_by_target_time':
            cutoff = reach_times[target_event]
            answer = sum(
                1 for time in reach_times.values()
                if time < float('inf') and time <= cutoff
            )
            return query_type, target_event, int(answer), None

        if query_type == 'latest_event_time':
            target_event = max(target_candidates, key=lambda e: (reach_times[e], e))
            return query_type, target_event, int(reach_times[target_event]), None

        return 'occurrence_time', target_event, int(reach_times[target_event]), None
    
    def _generate_events(self, num_events: int) -> Dict[str, Event]:
        """사건 노드 생성"""
        events = {}
        
        # 사건 타입별 분배
        event_types = list(EventType)
        selected = []
        
        for i in range(num_events):
            event_type = event_types[i % len(event_types)]
            available = [t for t in self.event_templates[event_type] 
                        if t[0] not in [s[0] for s in selected]]
            
            if not available:
                available = self.event_templates[event_type]
            
            name, description = random.choice(available)
            selected.append((name, description))
            
            events[f"E{i+1}"] = Event(
                id=f"E{i+1}",
                name=name,
                description=description,
                event_type=event_type
            )
        
        return events
    
    def _generate_causal_graph(self, events: Dict[str, Event], 
                               config: Dict) -> List[CausalEdge]:
        """AND/OR 조건을 갖는 DAG 인과 관계 생성"""
        edges = []
        event_ids = sorted(events.keys())
        and_prob = config.get('and_probability', 0.0)
        extra_edge_rate = config.get('edge_density', 0.0)
        
        # 각 타겟 노드에 대해 간선 생성
        for i, to_id in enumerate(event_ids):
            if i == 0:
                continue  # 첫 노드 건너뜀 (트리거 후보)

            # DAG를 유지하기 위해 이전 노드에서만 전제 사건 선택
            possible_from = event_ids[:i]
            if not possible_from:
                continue

            base_extra = int(extra_edge_rate)
            fractional_extra = 1 if random.random() < (extra_edge_rate - base_extra) else 0
            rules_for_target = 1 + base_extra + fractional_extra
            seen_prereq_sets: Set[Tuple[str, ...]] = set()

            for _ in range(rules_for_target):
                max_prereqs = min(config['max_out_degree'], len(possible_from))
                num_prereqs = random.randint(1, max(1, max_prereqs))
                from_events = tuple(sorted(
                    random.sample(possible_from, min(num_prereqs, len(possible_from)))
                ))

                # 같은 타겟에 대해 완전히 같은 전제 집합이 중복되지 않게 한다.
                if from_events in seen_prereq_sets and len(seen_prereq_sets) < len(possible_from):
                    continue
                seen_prereq_sets.add(from_events)

                condition = 'OR'
                if len(from_events) > 1 and random.random() < and_prob:
                    condition = 'AND'

                delay = random.randint(*config['delay_range'])
                edges.append(CausalEdge(
                    from_event=from_events[0],
                    to_event=to_id,
                    delay=delay,
                    from_events=list(from_events),
                    condition=condition if len(from_events) > 1 else 'OR'
                ))
        
        return edges
    
    def _calculate_in_degree(self, events: Dict[str, Event], 
                            edges: List[CausalEdge]) -> Dict[str, int]:
        """각 노드의 진입 차수 계산"""
        in_degree = {e_id: 0 for e_id in events}
        for edge in edges:
            in_degree[edge.to_event] += 1
        return in_degree
    
    def _calculate_reach_times(self, events: Dict[str, Event],
                               edges: List[CausalEdge],
                               trigger: str,
                               trigger_time: int) -> Dict[str, int]:
        """
        AND/OR 조건으로 각 사건의 최초 발생 시간 계산
        
        Returns:
            사건 ID -> 최초 발생 시간 매핑 딕셔너리
        """
        # 각 사건의 최초 발생 시간 추적
        earliest_time = {e_id: float('inf') for e_id in events}
        earliest_time[trigger] = trigger_time
        
        # 인과 규칙별 전제 도달 시각 추적. 같은 사건을 향하는 여러
        # 대체 규칙이 있을 수 있으며 지연 시간이 다를 수 있다.
        prereq_arrival_times = {idx: {} for idx in range(len(edges))}
        
        # 우선순위 큐: (시간, 사건 ID)
        pq = [(trigger_time, trigger)]
        processed = set()
        
        while pq:
            current_time, current_event = heapq.heappop(pq)
            
            if current_event in processed:
                continue
            processed.add(current_event)
            
            # current_event에 의존하는 모든 사건 찾기
            for edge_idx, edge in enumerate(edges):
                from_events = edge.from_events if edge.from_events else [edge.from_event]
                if current_event not in from_events:
                    continue
                
                to_event = edge.to_event
                arrival_time = current_time + edge.delay
                arrivals = prereq_arrival_times[edge_idx]
                
                # 이 전제 조건의 도달 시간 기록
                if current_event not in arrivals:
                    arrivals[current_event] = arrival_time
                else:
                    # 이 전제 조건의 최초 도달 시각 유지
                    arrivals[current_event] = min(
                        arrivals[current_event],
                        arrival_time
                    )
                
                # 모든 전제 조건이 도달했는지 확인
                all_prereqs_arrived = all(
                    prereq in arrivals
                    for prereq in from_events
                )
                
                if all_prereqs_arrived:
                    # 조건 타입에 따라 트리거 시간 계산
                    if edge.condition == 'AND':
                        # 모든 전제 조건 대기
                        trigger_time_for_event = max(
                            arrivals[prereq]
                            for prereq in from_events
                        )
                    else:  # OR
                        # 첫 번째 전제 조건에서 트리거
                        trigger_time_for_event = min(
                            arrivals[prereq]
                            for prereq in from_events
                        )
                    
                    # 현재 최선보다 빠르면 업데이트
                    if trigger_time_for_event < earliest_time[to_event]:
                        earliest_time[to_event] = trigger_time_for_event
                        heapq.heappush(pq, (trigger_time_for_event, to_event))
        
        return earliest_time
    
    def _generate_simple_puzzle(self, difficulty: str) -> CausalPuzzle:
        """폴백용 단순 선형 체인 생성"""
        num_events = 4
        events = self._generate_events(num_events)
        event_ids = sorted(events.keys())
        
        edges = []
        for i in range(len(event_ids) - 1):
            edges.append(CausalEdge(
                from_event=event_ids[i],
                to_event=event_ids[i+1],
                delay=random.randint(10, 30)
            ))
        
        trigger = event_ids[0]
        trigger_time = 0
        target_event = event_ids[-1]
        
        reach_times = self._calculate_reach_times(events, edges, trigger, trigger_time)
        answer = reach_times[target_event]
        
        return CausalPuzzle(
            events=events,
            edges=edges,
            trigger=trigger,
            trigger_time=trigger_time,
            target_event=target_event,
            answer=answer,
            difficulty=difficulty
        )
    
    def has_unique_solution(self, puzzle: CausalPuzzle) -> bool:
        """
        퍼즐의 유일 해 검증
        
        Dijkstra는 결정적이므로 주어진 그래프에 대해 항상 유일 해를 가집니다.
        그래프가 유효한지만 검증 (DAG, 연결됨).
        """
        # 타겟이 도달 가능한지 확인
        reach_times = self._calculate_reach_times(
            puzzle.events,
            puzzle.edges,
            puzzle.trigger,
            puzzle.trigger_time
        )
        
        return reach_times[puzzle.target_event] < float('inf')


def create_question(puzzle: CausalPuzzle, shuffle_edges: Optional[bool] = None) -> str:
    """퍼즐에 대한 한국어 질문 텍스트 생성."""
    
    # 사건 형식화
    event_lines = []
    for event_id in sorted(puzzle.events.keys()):
        event = puzzle.events[event_id]
        event_lines.append(f"  {event_id}: {event.name}")
        event_lines.append(f"      ({event.description})")
    
    events_description = '\n'.join(event_lines)
    
    # 인과 관계 형식화
    causal_lines = []
    if shuffle_edges is None:
        shuffle_edges = puzzle.shuffle_edges
    if shuffle_edges:
        sorted_edges = list(puzzle.edges)
        random.shuffle(sorted_edges)
    else:
        sorted_edges = sorted(puzzle.edges, key=lambda e: e.to_event)
    
    for edge in sorted_edges:
        from_events = edge.from_events if edge.from_events else [edge.from_event]
        to_name = puzzle.events[edge.to_event].name
        
        if len(from_events) == 1:
            from_name = puzzle.events[from_events[0]].name
            causal_lines.append(
                f"  {from_events[0]} ({from_name}) → "
                f"{edge.to_event} ({to_name}): {edge.delay}분"
            )
        else:
            if edge.condition == 'AND':
                prereq_str = ' 그리고 '.join(f"{e} ({puzzle.events[e].name})" 
                                          for e in from_events)
                line = f"  [{prereq_str}] → {edge.to_event} ({to_name}): {edge.delay}분"
                line += "\n      (모든 전제 조건 필요)"
            else:
                prereq_str = ' 또는 '.join(f"{e} ({puzzle.events[e].name})" 
                                         for e in from_events)
                line = f"  [{prereq_str}] → {edge.to_event} ({to_name}): {edge.delay}분"
                line += "\n      (첫 번째 전제 조건에서 트리거)"
            causal_lines.append(line)
    
    causality_description = '\n'.join(causal_lines)
    
    trigger_name = puzzle.events[puzzle.trigger].name
    target_name = puzzle.events[puzzle.target_event].name
    compare_line = ""
    answer_instruction = "답변은 정수 하나로만 제공하세요."
    if puzzle.query_type == 'time_gap' and puzzle.compare_event:
        compare_name = puzzle.events[puzzle.compare_event].name
        question_line = (
            f"사건 {puzzle.compare_event} ({compare_name})이(가) 처음 발생한 뒤, "
            f"사건 {puzzle.target_event} ({target_name})이(가) 처음 발생하기까지 "
            "몇 분이 걸립니까?"
        )
        compare_line = (
            f"- 비교 사건: {puzzle.compare_event} ({compare_name}); 두 사건의 발생 시각을 모두 계산한 뒤 "
            "대상 사건 시각에서 비교 사건 시각을 빼세요.\n"
        )
    elif puzzle.query_type == 'count_by_target_time':
        question_line = (
            f"사건 {puzzle.target_event} ({target_name})이(가) 처음 발생하는 시점까지, "
            "목록에 있는 서로 다른 사건은 총 몇 개 발생했습니까? 초기 사건과 대상 사건도 "
            "개수에 포함하세요."
        )
    elif puzzle.query_type == 'latest_event_time':
        question_line = (
            "초기 조건이 그래프 전체로 전파된 뒤, 도달 가능한 목록 내 사건 중 "
            "가장 마지막으로 처음 발생하는 사건은 몇 분에 발생합니까?"
        )
        answer_instruction = "답변은 분 단위 정수 하나로만 제공하세요."
    else:
        question_line = (
            f"사건 {puzzle.target_event} ({target_name})은(는) 몇 분에 처음 발생합니까?"
        )
        answer_instruction = "답변은 분 단위 정수 하나로만 제공하세요."
    
    question = f"""인과 관계 사건 시스템과 시간에 따른 전파를 분석하고 있습니다.

사건 목록:
{events_description}

인과 관계 (시간 지연 표시):
{causality_description}

규칙:
- 사건이 발생하면 지정된 지연 시간 후에 영향을 미칩니다
- OR 조건: 첫 번째 전제 조건이 도달하면 사건이 발생합니다
- AND 조건: 모든 전제 조건이 발생한 후에만 사건이 발생합니다
- 모든 시간은 기준점(0분)에서부터 측정됩니다

초기 조건:
- 사건 {puzzle.trigger} ({trigger_name})이(가) {puzzle.trigger_time}분에 발생합니다
{compare_line}

질문:
{question_line}

{answer_instruction}
예를 들어, 요청한 값이 45이면 답변: 45
"""
    
    return question


def generate_dataset(puzzles_per_difficulty: int = 3, verbose: bool = True) -> List[Dict]:
    """
    인과 DAG 퍼즐 전체 데이터셋 생성
    
    Args:
        puzzles_per_difficulty: 난이도별 퍼즐 수
        verbose: 생성 진행 상황 출력
    
    Returns:
        평가 준비된 퍼즐 딕셔너리 리스트
    """
    generator = CausalPuzzleGenerator()
    difficulties = ['easy', 'medium', 'hard']
    dataset = []
    
    for difficulty in difficulties:
        if verbose:
            print(f"\n=== {difficulty} 퍼즐 생성 중 ===")
        
        for i in range(puzzles_per_difficulty):
            puzzle = generator.generate_puzzle(difficulty)
            question = create_question(puzzle)
            
            puzzle_data = {
                'question': question,
                'answer': str(puzzle.answer),
                'difficulty': difficulty,
                'metadata': puzzle.to_dict()
            }
            
            dataset.append(puzzle_data)
            
            if verbose:
                print(f"  [{i+1}/{puzzles_per_difficulty}] "
                      f"{puzzle.trigger} → {puzzle.target_event}: "
                      f"{puzzle.answer}분")
    
    return dataset


SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _reach_time_trace_ko(puzzle: CausalPuzzle) -> tuple:
    """도달 시각 계산을 재현하고 SEG 추적 라인과 메타를 반환."""
    events = puzzle.events
    edges = puzzle.edges
    trigger = puzzle.trigger
    trigger_time = puzzle.trigger_time
    target = puzzle.target_event

    earliest = {eid: float('inf') for eid in events}
    earliest[trigger] = trigger_time
    prereq_arrival: Dict[int, Dict[str, int]] = {idx: {} for idx in range(len(edges))}
    resolver: Dict[str, CausalEdge] = {}

    pq = [(trigger_time, trigger)]
    processed = set()
    while pq:
        ct, ce = heapq.heappop(pq)
        if ce in processed:
            continue
        processed.add(ce)
        for edge_idx, edge in enumerate(edges):
            fes = edge.from_events if edge.from_events else [edge.from_event]
            if ce not in fes:
                continue
            at = ct + edge.delay
            pa = prereq_arrival[edge_idx]
            if ce not in pa or at < pa[ce]:
                pa[ce] = at
            if all(p in pa for p in fes):
                if edge.condition == 'AND':
                    tt = max(pa[p] for p in fes)
                else:
                    tt = min(pa[p] for p in fes)
                if tt < earliest[edge.to_event]:
                    earliest[edge.to_event] = tt
                    resolver[edge.to_event] = edge
                    heapq.heappush(pq, (tt, edge.to_event))

    def nm(eid: str) -> str:
        return events[eid].name if eid in events else eid

    reached = [eid for eid in events if earliest[eid] < float('inf')]
    reached.sort(key=lambda e: (earliest[e], e))

    lines: List[str] = []
    and_count = or_count = 0
    for seg, eid in enumerate(reached, start=1):
        star = " ★" if eid == target else ""
        if eid == trigger:
            lines.append(
                f"    [SEG {seg}] (트리거) {nm(eid)} t={earliest[eid]}분{star}")
            continue
        edge = resolver[eid]
        fes = edge.from_events if edge.from_events else [edge.from_event]
        if len(fes) > 1 and edge.condition == 'AND':
            and_count += 1
            parts = " · ".join(
                f"{nm(p)}(t={earliest[p]})" for p in fes)
            peak = max(earliest[p] for p in fes)
            lines.append(
                f"    [SEG {seg}] {nm(eid)}: AND[{parts}] "
                f"→ max={peak} + 지연 {edge.delay} = t={earliest[eid]}분{star}")
        elif len(fes) > 1 and edge.condition == 'OR':
            or_count += 1
            parts = " · ".join(
                f"{nm(p)}(t={earliest[p]})" for p in fes)
            low = min(earliest[p] for p in fes)
            lines.append(
                f"    [SEG {seg}] {nm(eid)}: OR[{parts}] "
                f"→ min={low} + 지연 {edge.delay} = t={earliest[eid]}분{star}")
        else:
            p = fes[0]
            lines.append(
                f"    [SEG {seg}] {nm(eid)}: {nm(p)}(t={earliest[p]}) "
                f"+ 지연 {edge.delay} = t={earliest[eid]}분{star}")

    path_len = 0
    cur = target
    guard = 0
    while cur != trigger and cur in resolver and guard < len(events) + 2:
        edge = resolver[cur]
        fes = edge.from_events if edge.from_events else [edge.from_event]
        if edge.condition == 'AND':
            cur = max(fes, key=lambda p: earliest[p])
        else:
            cur = min(fes, key=lambda p: earliest[p])
        path_len += 1
        guard += 1

    summary = {
        'reached': len(reached),
        'and_count': and_count,
        'or_count': or_count,
        'path_len': path_len,
    }
    return lines, summary


def _build_causal_dag_solution_ko(puzzle: CausalPuzzle) -> str:
    """SFT teacher trace: 전파 규칙과 요청된 정수 답."""
    n_ev = len(puzzle.events)
    n_ed = len(puzzle.edges)
    trace_lines, smry = _reach_time_trace_ko(puzzle)
    reach_times = CausalPuzzleGenerator()._calculate_reach_times(
        puzzle.events,
        puzzle.edges,
        puzzle.trigger,
        puzzle.trigger_time,
    )
    target_time = reach_times[puzzle.target_event]
    if puzzle.query_type == 'time_gap' and puzzle.compare_event:
        compare_time = reach_times[puzzle.compare_event]
        query_note = (
            f"  - 질의: {puzzle.compare_event}(t={compare_time})부터 "
            f"{puzzle.target_event}(t={target_time})까지의 시간 차"
        )
        answer_label = "최종 답(두 사건 사이 분 차이)"
    elif puzzle.query_type == 'count_by_target_time':
        query_note = (
            f"  - 질의: {puzzle.target_event}의 시각 t={target_time} 이하에 "
            "발생한 사건 수"
        )
        answer_label = "최종 답(사건 개수)"
    elif puzzle.query_type == 'latest_event_time':
        query_note = "  - 질의: 도달 가능한 목록 내 사건 중 가장 늦은 최초 발생 시각"
        answer_label = "최종 답(분)"
    else:
        query_note = f"  - 질의: {puzzle.target_event}의 최초 발생 시각"
        answer_label = "최종 답(분)"
    head = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {puzzle.difficulty}",
        f"  - 트리거 사건: {puzzle.trigger} (t = {puzzle.trigger_time}분)",
        f"  - 대상 사건: {puzzle.target_event}",
        f"  - 질의 유형: {puzzle.query_type}",
        query_note,
        f"  - 그래프 크기: 사건 {n_ev}개, 엣지 {n_ed}개",
        "  - 최종 요청 정수는 [STEP 3](검산)에만 명시된다. "
        "먼저 [STEP 2]의 SEG 로그를 따를 것.",
        "[STEP 1] 주어진 조건 (프롬프트와 동일한 시간 전파 및 그래프 규칙)",
        "  - 각 엣지에는 지연이 있으며, 하류 사건은 전제 사건이 발생한 뒤 "
        "발화할 수 있다.",
        "  - OR: 규칙을 만족하는 전제 도달 시각 중 **가장 이른** 시각을 사용 "
        "(문제 표기: FIRST 전제에 의해 트리거).",
        "  - AND: **모든** 전제가 발생할 때까지 기다린 뒤, 전제 완료 시각 중 "
        "**가장 늦은** 시각에 엣지 지연을 더함.",
        "[STEP 2] 풀이 전개 (도달 시각)",
        (f"  · 요약: {puzzle.trigger}(t={puzzle.trigger_time})"
         f" → {puzzle.target_event} · 도달 사건 "
         f"{smry['reached']}/{n_ev} · AND {smry['and_count']} / "
         f"OR {smry['or_count']} · 임계 경로 {smry['path_len']}홉"),
        "  · 머릿속으로: 최초 도달 시각 오름차순으로 사건을 처리; "
        "AND는 max+지연, OR는 min+지연.",
    ]
    tail = [
        "[STEP 3] 답·검산",
        f"  - {answer_label}: {puzzle.answer}",
        "  - 우선순위 큐 전파를 "
        f"{puzzle.trigger} 시각 {puzzle.trigger_time}에서 다시 실행; "
        "AND 노드에서는 max(전제)+지연, OR 노드에서는 min(전제)+지연을 확인.",
    ]
    return "\n".join(head + trace_lines + tail)


def create_dataset_files(num_questions: int):
    """
    인과 DAG 퍼즐 데이터셋 파일 생성
    
    Args:
        num_questions: 생성할 질문 수
        version: 파일명 버전 문자열
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (데이터프레임, JSON 리스트)
    """
    import pandas as pd
    
    print(f"{num_questions}개의 인과 DAG 퍼즐 생성 중...")
    
    generator = CausalPuzzleGenerator()
    
    # 난이도별 퍼즐 계산
    puzzles_per_diff = num_questions // 3
    remainder = num_questions % 3
    
    difficulties = ['easy', 'medium', 'hard']
    all_puzzles = []
    
    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        
        for j in range(count):
            puzzle = generator.generate_puzzle(difficulty, seed=i*1000+j)
            puzzle_data = {
                'id': f'causal_dag_ko_{difficulty}_{j:04d}',
                'question': create_question(puzzle),
                'answer': str(puzzle.answer),
                'solution': _build_causal_dag_solution_ko(puzzle),
                'difficulty': difficulty,
            }
            all_puzzles.append(puzzle_data)
    
    print(f"\n{len(all_puzzles)}개의 퍼즐 생성 완료")
    
    export_cols = ['id', 'question', 'answer', 'solution', 'difficulty']
    df = pd.DataFrame([{k: p[k] for k in export_cols} for p in all_puzzles])
    
    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("causal_dag_ko", all_puzzles)

    return df, all_puzzles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal DAG Puzzle Generator")
    parser.add_argument("--num", type=int, default=300, help="생성할 질문 수")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("인과 DAG 추론 퍼즐 생성기 (한국어)")
    print("=" * 70)
    
    create_dataset_files(num_questions=args.num)
    

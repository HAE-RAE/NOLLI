"""Causal DAG (Directed Acyclic Graph) Reasoning Puzzle Generator

Generates causal reasoning puzzles based on event chains and time propagation.
Tests LLM's ability to reason about cause-effect relationships over time.

Key Features:
1. DAG-based event graph (no cycles)
2. Time-delayed causal relationships
3. Shortest path reasoning (Dijkstra)
4. Unique solution guarantee (deterministic graph)
"""

import random
import heapq
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class EventType(Enum):
    """Event categories for puzzle generation"""
    TECHNICAL = "Technical"
    BUSINESS = "Business"
    ENVIRONMENTAL = "Environmental"
    OPERATIONAL = "Operational"


@dataclass
class CausalEdge:
    """Represents a causal relationship between events"""
    from_event: str
    to_event: str
    delay: int  # Time delay in minutes
    from_events: Optional[List[str]] = None  # Multiple prerequisites (for AND)
    condition: str = 'OR'  # 'OR' or 'AND'
    
    def __repr__(self):
        if self.from_events and len(self.from_events) > 1:
            cond = ' AND ' if self.condition == 'AND' else ' OR '
            return f"[{cond.join(self.from_events)}] → {self.to_event} (+{self.delay}min)"
        return f"{self.from_event} → {self.to_event} (+{self.delay}min)"


@dataclass
class Event:
    """Represents an event in the causal graph"""
    id: str
    name: str
    description: str
    event_type: EventType
    
    def __repr__(self):
        return f"{self.id}: {self.name}"


@dataclass
class CausalPuzzle:
    """Complete causal reasoning puzzle"""
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
        """Convert to dictionary for JSON serialization"""
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
    """Generate causal DAG reasoning puzzles"""
    
    def __init__(self):
        """Initialize with event templates"""
        self.event_templates = {
            EventType.TECHNICAL: [
                ('PowerOutage', 'Main power grid fails'),
                ('ServerDown', 'Application server becomes unavailable'),
                ('DatabaseCrash', 'Database service stops responding'),
                ('NetworkFailure', 'Network connectivity is lost'),
                ('DiskFull', 'Storage capacity reaches 100%'),
                ('MemoryLeak', 'Application memory usage spikes'),
                ('BackupFailed', 'Automated backup process fails'),
                ('SecurityBreach', 'Unauthorized access detected'),
                ('APITimeout', 'External API stops responding'),
                ('CacheExpired', 'Cache invalidation occurs'),
            ],
            EventType.BUSINESS: [
                ('OrderReceived', 'Customer places new order'),
                ('PaymentProcessed', 'Payment transaction completes'),
                ('InventoryLow', 'Stock level falls below threshold'),
                ('ShipmentDelayed', 'Delivery schedule is pushed back'),
                ('CustomerComplaint', 'Support ticket is created'),
                ('RefundIssued', 'Money is returned to customer'),
                ('PriceChanged', 'Product pricing is updated'),
                ('PromotionStarted', 'Marketing campaign launches'),
                ('ContractSigned', 'Legal agreement is finalized'),
                ('InvoiceSent', 'Billing document is generated'),
            ],
            EventType.ENVIRONMENTAL: [
                ('HeavyRain', 'Precipitation exceeds 50mm/hour'),
                ('RoadFlooded', 'Water level blocks traffic'),
                ('TrafficJam', 'Vehicle congestion occurs'),
                ('PowerSurge', 'Electrical grid experiences spike'),
                ('Earthquake', 'Seismic activity detected'),
                ('HeatWave', 'Temperature exceeds 35°C'),
                ('StormWarning', 'Severe weather alert issued'),
                ('Snowfall', 'Snow accumulation begins'),
                ('WindDamage', 'Strong winds cause infrastructure damage'),
                ('Drought', 'Water supply becomes limited'),
            ],
            EventType.OPERATIONAL: [
                ('MaintenanceScheduled', 'Planned system maintenance begins'),
                ('StaffShortage', 'Available workforce drops below minimum'),
                ('EquipmentFailure', 'Critical machinery stops working'),
                ('QualityIssue', 'Product defect is discovered'),
                ('SupplyChainDisruption', 'Vendor delivery is interrupted'),
                ('CapacityReached', 'Maximum throughput is exceeded'),
                ('PolicyChanged', 'New operational rules take effect'),
                ('InspectionFailed', 'Compliance check does not pass'),
                ('TrainingCompleted', 'Staff certification is achieved'),
                ('SystemUpgrade', 'Software version is updated'),
            ]
        }
    
    def generate_puzzle(self, difficulty: str, seed: Optional[int] = None) -> CausalPuzzle:
        """
        Generate a causal reasoning puzzle
        
        Args:
            difficulty: 'easy', 'medium', or 'hard'
            seed: Random seed for reproducibility
        
        Returns:
            CausalPuzzle with unique solution
        """
        if seed is not None:
            random.seed(seed)
        
        # Difficulty configuration calibrated from gemini-3-flash-preview runs.
        # The main difficulty levers are: shuffled presentation, alternative
        # causal rules, deeper targets, and query types that require broader
        # propagation than a single target path.
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
                
                # Select trigger (node with in-degree 0)
                in_degree = self._calculate_in_degree(events, edges)
                possible_triggers = [e_id for e_id, degree in in_degree.items() 
                                    if degree == 0]
                
                if not possible_triggers:
                    continue
                
                trigger = random.choice(possible_triggers)
                trigger_time = random.randint(0, 60)
                
                reach_times = self._calculate_reach_times(events, edges, trigger, trigger_time)
                
                # Select target/query (reachable event, not trigger)
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
        
        # If all attempts fail, generate simple linear chain
        return self._generate_simple_puzzle(difficulty)

    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        """Choose a key from a small weight dictionary."""
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
        """Select the question type and compute the corresponding answer."""
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
        """Generate event nodes"""
        events = {}
        
        # Distribute events across types
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
        """Generate DAG of causal relationships with AND/OR conditions"""
        edges = []
        event_ids = sorted(events.keys())
        and_prob = config.get('and_probability', 0.0)
        extra_edge_rate = config.get('edge_density', 0.0)
        
        # Create edges for each target node
        for i, to_id in enumerate(event_ids):
            if i == 0:
                continue  # Skip first node (trigger candidate)

            # Select prerequisite events (from earlier nodes only to preserve DAG)
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

                # Avoid exact duplicate prerequisite sets for the same target.
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
        """Calculate in-degree for each node"""
        in_degree = {e_id: 0 for e_id in events}
        for edge in edges:
            in_degree[edge.to_event] += 1
        return in_degree
    
    def _calculate_reach_times(self, events: Dict[str, Event],
                               edges: List[CausalEdge],
                               trigger: str,
                               trigger_time: int) -> Dict[str, int]:
        """
        Calculate earliest time each event occurs with AND/OR conditions
        
        Returns:
            Dictionary mapping event_id -> earliest occurrence time
        """
        # Track earliest time each event occurs
        earliest_time = {e_id: float('inf') for e_id in events}
        earliest_time[trigger] = trigger_time
        
        # Track prerequisite arrivals per causal rule. Multiple alternative
        # rules can point to the same event and may have different delays.
        prereq_arrival_times = {idx: {} for idx in range(len(edges))}
        
        # Priority queue: (time, event_id)
        pq = [(trigger_time, trigger)]
        processed = set()
        
        while pq:
            current_time, current_event = heapq.heappop(pq)
            
            if current_event in processed:
                continue
            processed.add(current_event)
            
            # Find all events that depend on current_event
            for edge_idx, edge in enumerate(edges):
                from_events = edge.from_events if edge.from_events else [edge.from_event]
                if current_event not in from_events:
                    continue
                
                to_event = edge.to_event
                arrival_time = current_time + edge.delay
                arrivals = prereq_arrival_times[edge_idx]
                
                # Record this prerequisite's arrival time
                if current_event not in arrivals:
                    arrivals[current_event] = arrival_time
                else:
                    # Keep earliest arrival from this prerequisite
                    arrivals[current_event] = min(
                        arrivals[current_event],
                        arrival_time
                    )
                
                # Check if all prerequisites have arrived
                all_prereqs_arrived = all(
                    prereq in arrivals
                    for prereq in from_events
                )
                
                if all_prereqs_arrived:
                    # Calculate trigger time based on condition
                    if edge.condition == 'AND':
                        # Wait for ALL prerequisites
                        trigger_time_for_event = max(
                            arrivals[prereq]
                            for prereq in from_events
                        )
                    else:  # OR
                        # Trigger on FIRST prerequisite
                        trigger_time_for_event = min(
                            arrivals[prereq]
                            for prereq in from_events
                        )
                    
                    # Update if this is earlier than current best
                    if trigger_time_for_event < earliest_time[to_event]:
                        earliest_time[to_event] = trigger_time_for_event
                        heapq.heappush(pq, (trigger_time_for_event, to_event))
        
        return earliest_time
    
    def _generate_simple_puzzle(self, difficulty: str) -> CausalPuzzle:
        """Generate simple linear chain as fallback"""
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
        Verify puzzle has unique solution
        
        Since Dijkstra is deterministic, solution is always unique for a given graph.
        Just verify the graph is valid (DAG, connected).
        """
        # Check if target is reachable
        reach_times = self._calculate_reach_times(
            puzzle.events,
            puzzle.edges,
            puzzle.trigger,
            puzzle.trigger_time
        )
        
        return reach_times[puzzle.target_event] < float('inf')


def create_question(puzzle: CausalPuzzle, shuffle_edges: Optional[bool] = None) -> str:
    """Generate English question text for the puzzle."""
    
    # Format events
    event_lines = []
    for event_id in sorted(puzzle.events.keys()):
        event = puzzle.events[event_id]
        event_lines.append(f"  {event_id}: {event.name}")
        event_lines.append(f"      ({event.description})")
    
    events_description = '\n'.join(event_lines)
    
    # Format causal relationships
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
                f"{edge.to_event} ({to_name}): {edge.delay} minutes"
            )
        else:
            if edge.condition == 'AND':
                prereq_str = ' AND '.join(f"{e} ({puzzle.events[e].name})" 
                                          for e in from_events)
                line = f"  [{prereq_str}] → {edge.to_event} ({to_name}): {edge.delay} minutes"
                line += "\n      (Requires ALL prerequisites)"
            else:
                prereq_str = ' OR '.join(f"{e} ({puzzle.events[e].name})" 
                                         for e in from_events)
                line = f"  [{prereq_str}] → {edge.to_event} ({to_name}): {edge.delay} minutes"
                line += "\n      (Triggered by FIRST prerequisite)"
            causal_lines.append(line)
    
    causality_description = '\n'.join(causal_lines)
    
    trigger_name = puzzle.events[puzzle.trigger].name
    target_name = puzzle.events[puzzle.target_event].name
    compare_line = ""
    answer_instruction = "Provide your answer as a single integer."
    if puzzle.query_type == 'time_gap' and puzzle.compare_event:
        compare_name = puzzle.events[puzzle.compare_event].name
        question_line = (
            f"How many minutes after event {puzzle.compare_event} ({compare_name}) "
            f"first occurs does event {puzzle.target_event} ({target_name}) first occur?"
        )
        compare_line = (
            f"- Compare event: {puzzle.compare_event} ({compare_name}); compute both event times, "
            "then subtract the compare event time from the target event time.\n"
        )
    elif puzzle.query_type == 'count_by_target_time':
        question_line = (
            f"By the time event {puzzle.target_event} ({target_name}) first occurs, "
            "how many distinct listed events have occurred? Include the initial event and "
            "the target event in the count."
        )
    elif puzzle.query_type == 'latest_event_time':
        question_line = (
            "At what minute does the last reachable listed event first occur after the "
            "initial condition propagates through the graph?"
        )
        answer_instruction = "Provide your answer as a single integer minute."
    else:
        question_line = (
            f"At what minute does event {puzzle.target_event} ({target_name}) first occur?"
        )
        answer_instruction = "Provide your answer as a single integer minute."
    
    question = f"""You are analyzing a system of causal events and their propagation over time.

Events:
{events_description}

Causal Relationships (showing time delays):
{causality_description}

Rules:
- When an event occurs, it triggers its effects after the specified delay
- OR condition: Event occurs when the FIRST prerequisite reaches it
- AND condition: Event occurs only when ALL prerequisites have occurred
- All times are measured in minutes from a reference point (minute 0)

Initial Condition:
- Event {puzzle.trigger} ({trigger_name}) occurs at minute {puzzle.trigger_time}
{compare_line}

Question:
{question_line}

{answer_instruction}
For example, if the requested value is 45, answer: 45
"""
    
    return question


def generate_dataset(puzzles_per_difficulty: int = 3, verbose: bool = True) -> List[Dict]:
    """
    Generate a complete dataset of causal DAG puzzles
    
    Args:
        puzzles_per_difficulty: Number of puzzles per difficulty level
        verbose: Print generation progress
    
    Returns:
        List of puzzle dictionaries ready for evaluation
    """
    generator = CausalPuzzleGenerator()
    difficulties = ['easy', 'medium', 'hard']
    dataset = []
    
    for difficulty in difficulties:
        if verbose:
            print(f"\n=== Generating {difficulty} puzzles ===")
        
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
                      f"{puzzle.answer} minutes")
    
    return dataset


SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


def _reach_time_trace_en(puzzle: CausalPuzzle) -> tuple:
    """Replay the reach-time computation and emit SEG trace lines + meta."""
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
        star = " *" if eid == target else ""
        if eid == trigger:
            lines.append(
                f"    [SEG {seg}] (trigger) {nm(eid)} t={earliest[eid]} min{star}")
            continue
        edge = resolver[eid]
        fes = edge.from_events if edge.from_events else [edge.from_event]
        if len(fes) > 1 and edge.condition == 'AND':
            and_count += 1
            parts = " | ".join(
                f"{nm(p)}(t={earliest[p]})" for p in fes)
            peak = max(earliest[p] for p in fes)
            lines.append(
                f"    [SEG {seg}] {nm(eid)}: AND[{parts}] "
                f"-> max={peak} + delay {edge.delay} = t={earliest[eid]} min{star}")
        elif len(fes) > 1 and edge.condition == 'OR':
            or_count += 1
            parts = " | ".join(
                f"{nm(p)}(t={earliest[p]})" for p in fes)
            low = min(earliest[p] for p in fes)
            lines.append(
                f"    [SEG {seg}] {nm(eid)}: OR[{parts}] "
                f"-> min={low} + delay {edge.delay} = t={earliest[eid]} min{star}")
        else:
            p = fes[0]
            lines.append(
                f"    [SEG {seg}] {nm(eid)}: {nm(p)}(t={earliest[p]}) "
                f"+ delay {edge.delay} = t={earliest[eid]} min{star}")

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


def _build_causal_dag_solution_en(puzzle: CausalPuzzle) -> str:
    """SFT teacher trace: propagation rules and requested integer answer."""
    n_ev = len(puzzle.events)
    n_ed = len(puzzle.edges)
    trace_lines, smry = _reach_time_trace_en(puzzle)
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
            f"  - Query: time gap from {puzzle.compare_event} "
            f"(t={compare_time}) to {puzzle.target_event} (t={target_time})"
        )
        answer_label = "Final answer (minutes between events)"
    elif puzzle.query_type == 'count_by_target_time':
        query_note = (
            f"  - Query: count events with occurrence time <= "
            f"{puzzle.target_event}'s time t={target_time}"
        )
        answer_label = "Final answer (event count)"
    elif puzzle.query_type == 'latest_event_time':
        query_note = "  - Query: latest first-occurrence time among reachable listed events"
        answer_label = "Final answer (minutes)"
    else:
        query_note = f"  - Query: first occurrence time of {puzzle.target_event}"
        answer_label = "Final answer (minutes)"
    head = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {puzzle.difficulty}",
        f"  - Trigger event: {puzzle.trigger} (at t = {puzzle.trigger_time} min)",
        f"  - Target event: {puzzle.target_event}",
        f"  - Query type: {puzzle.query_type}",
        query_note,
        f"  - Graph size: {n_ev} events, {n_ed} edges",
        "  - The final requested integer is only stated in [STEP 3] (verification); "
        "follow the SEG log in [STEP 2] first.",
        "[STEP 1] Given (time propagation and graph rules, as in the prompt)",
        "  - Each edge has a delay; a downstream event can fire after its "
        "prerequisites have occurred.",
        "  - OR: use the **earliest** time among prerequisite arrivals that "
        "satisfy the rule (as stated: triggered by the FIRST prerequisite).",
        "  - AND: wait until **all** prerequisites have occurred, then use the "
        "**latest** prerequisite completion time, then add the edge delay.",
        "[STEP 2] Worked solution (reach time)",
        (f"  · Summary: {puzzle.trigger}(t={puzzle.trigger_time})"
         f" -> {puzzle.target_event} · reached "
         f"{smry['reached']}/{n_ev} events · AND {smry['and_count']} / "
         f"OR {smry['or_count']} · critical path {smry['path_len']} hops"),
        "  · Mentally: process events in ascending earliest reach time; "
        "AND uses max+delay, OR uses min+delay.",
    ]
    tail = [
        "[STEP 3] Answer and verification",
        f"  - {answer_label}: {puzzle.answer}",
        "  - Re-run the priority-queue propagation from "
        f"{puzzle.trigger} at {puzzle.trigger_time}; "
        "check max(prereqs)+delay at AND nodes and min(prereqs)+delay at OR nodes.",
    ]
    return "\n".join(head + trace_lines + tail)


def create_dataset_files(num_questions: int):
    """
    Create dataset files for causal DAG puzzles
    
    Args:
        num_questions: Number of questions to generate
        version: Version string for filenames
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, json list)
    """
    import pandas as pd
    
    print(f"Generating {num_questions} causal DAG puzzles...")
    
    generator = CausalPuzzleGenerator()
    
    # Calculate puzzles per difficulty
    puzzles_per_diff = num_questions // 3
    remainder = num_questions % 3
    
    difficulties = ['easy', 'medium', 'hard']
    all_puzzles = []
    
    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        
        for j in range(count):
            puzzle = generator.generate_puzzle(difficulty, seed=i*1000+j)
            puzzle_data = {
                'id': f'causal_dag_en_{difficulty}_{j:04d}',
                'question': create_question(puzzle),
                'answer': str(puzzle.answer),
                'solution': _build_causal_dag_solution_en(puzzle),
                'difficulty': difficulty,
            }
            all_puzzles.append(puzzle_data)
    
    print(f"\nGenerated {len(all_puzzles)} puzzles")
    
    export_cols = ['id', 'question', 'answer', 'solution', 'difficulty']
    df = pd.DataFrame([{k: p[k] for k in export_cols} for p in all_puzzles])
    
    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("causal_dag_en", all_puzzles)

    return df, all_puzzles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal DAG Puzzle Generator")
    parser.add_argument("--num", type=int, default=300, help="Number of questions to generate")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Causal DAG Reasoning Puzzle Generator")
    print("=" * 70)
    
    create_dataset_files(num_questions=args.num)
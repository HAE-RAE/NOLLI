"""복면산(Cryptarithmetic) 퍼즐 생성기 - 한국어 자모(초성/중성/종성) · 실제 단어 버전

한국어 특화 메커니즘 (cipher_ko.HangulCipherSystem 의 자모 분해를 재사용):
- **피연산자와 결과가 모두 실제 한국어 단어**다 (예: 사과 + 나무 = 포도).
- 각 한글 음절을 초성/중성/종성으로 분해한다. 단어의 숫자값 = 음절 왼→오, 음절
  안에서는 초성→중성→종성 순으로 자모의 숫자를 이어붙여 만든다(받침 없는 음절은
  초성·중성 2자리).
- **초성표 / 중성표 / 종성표는 서로 독립**: 초성끼리·중성끼리·종성끼리만 서로 다른
  자모 = 서로 다른 숫자(0–9). 세 표는 별개(초성 ㄱ=3 이어도 종성 ㄱ 은 다를 수 있음).
- 각 단어의 첫 초성은 0이 될 수 없다.

추론(복면산의 본질):
- 자모↔숫자의 **일부만 공개**하고, 나머지는 **덧셈 제약(열별 받아올림)** 으로 추론.
- 공개량은 "정답(결과 단어의 숫자값)이 유일"해지는 선까지(난이도=가림 비율).

생성 방식:
- 실제 단어 풀에서 같은 자모길이 L 의 피연산자 N 개 + 결과 단어(자모길이 L 또는 L+1)를
  뽑아, 자모↔숫자 배정이 존재(덧셈 성립)하는 조합을 solver 로 찾는다. 결과 단어의
  자모 수 = 합의 자릿수 이어야 하므로, **결과가 피연산자보다 많이 길 수 없다**(예:
  사과+나무+포도=과수원 은 4+4+4 → 최대 5자리라 7자리 과수원 불가).

버전 이력:
- backups/cryptarithmetic_ko_syllable_v9.py : 음절 1:1 치환(영문식).
- (중간) 구성적 무의미음절 + 부분키 → 실제 단어로 교체.
"""

import os
import sys
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

# 자모 분해/합성은 repo 의 정통 한글 유틸을 재사용한다 (CLAUDE.md §12.2).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cipher_ko import HangulCipherSystem

MAX_SOLUTIONS = 1
_HANGUL = HangulCipherSystem()

# 토큰 타입 코드: C=초성(cho), V=중성(jung), F=종성(jong, final)
_TYPE_CHO, _TYPE_JUNG, _TYPE_JONG = "C", "V", "F"
_TYPE_KEYS = {_TYPE_CHO: "cho", _TYPE_JUNG: "jung", _TYPE_JONG: "jong"}
_TYPE_LABEL_KO = {"cho": "초성", "jung": "중성", "jong": "종성"}

# ============================================================================
# 실제 한국어 단어 풀 (자모 길이별 버킷은 런타임에 계산)
# ============================================================================

_WORD_POOL_FLAT = [
    # 2음절
    "사과", "나무", "포도", "바다", "시계", "구두", "모자", "오이", "우유", "두부",
    "가위", "나비", "다리", "머리", "허리", "가구", "의사", "가수", "배우", "화가",
    "도시", "미래", "회사", "노래", "시간", "사람", "사랑", "마음", "얼굴", "세상",
    "나라", "자연", "과일", "강산", "강물", "책상", "신발", "선물", "운동", "점심",
    "공장", "학생", "친구", "학교", "시장", "공원", "식당", "병원", "은행", "극장",
    "교실", "가족", "음식", "건물", "창문", "안경", "우산", "장갑", "양말", "동생",
    "부모", "이웃", "마을", "거리", "골목", "시골", "바위", "하늘", "구름", "바람",
    "소금", "거울", "가방", "지갑", "수박", "딸기", "마늘", "양파", "감자", "자유",
    "평화", "행복", "희망", "노력", "성공", "시작", "기차", "비누", "수건", "그릇",
    "접시", "냄비", "보석", "구슬", "동물", "식물", "과학", "역사", "음악", "미술",
    # 3음절
    "도서관", "운동장", "박물관", "미술관", "비행기", "자전거", "지하철", "무지개",
    "고양이", "강아지", "호랑이", "거북이", "소나무", "운동화", "손수건", "지우개",
    "책가방", "바닷가", "할머니", "어린이", "보름달", "민들레", "무궁화", "진달래",
    "다람쥐", "너구리", "두꺼비", "개구리", "잠자리", "지렁이", "달팽이", "병아리",
    "송아지", "얼룩말", "코뿔소", "원숭이", "공무원", "회사원", "경찰관", "소방관",
    "우체국", "정류장", "냉장고", "세탁기", "청소기", "선풍기", "음료수", "바나나",
    "토마토", "시금치", "양배추", "고구마", "도라지", "보따리", "주머니", "손가락",
    "발가락", "무릎뼈", "동그라미", "네모칸", "세모꼴", "윷놀이", "딱지치기",
]


def _jcount(word: str) -> int:
    return len(_decompose_word_to_tokens(word))


# ============================================================================
# 자모 토큰화
# ============================================================================

def _decompose_word_to_tokens(word: str, hangul: HangulCipherSystem = None) -> List[str]:
    """음절 단어를 타입 태그된 자모 토큰 리스트로 분해.

    토큰 포맷: 'C'+초성 / 'V'+중성 / 'F'+종성. 받침 없는 음절(CV)은 종성 토큰 생략.
    숫자 자릿수 순서 = 음절 왼→오, 음절 내 초성→중성→종성.
    """
    hangul = hangul or _HANGUL
    tokens: List[str] = []
    for ch in word:
        cho, jung, jong = hangul.decompose(ch)
        if cho < 0:
            continue
        tokens.append(_TYPE_CHO + hangul.CHO[cho])
        tokens.append(_TYPE_JUNG + hangul.JUNG[jung])
        if jong > 0:
            tokens.append(_TYPE_JONG + hangul.JONG[jong])
    return tokens


def _word_value(word: str, token_map: Dict[str, int]) -> int:
    """token->digit 매핑으로 단어의 숫자값 계산."""
    ds = []
    for t in _decompose_word_to_tokens(word):
        ds.append(str(token_map[t]))
    return int("".join(ds)) if ds else 0


def _perm(n: int, k: int) -> int:
    r = 1
    for i in range(k):
        r *= (n - i)
    return r


def _distinct_jamo_counts(words: List[str]) -> Tuple[int, int, int]:
    cs, vs, fs = set(), set(), set()
    for w in words:
        for t in _decompose_word_to_tokens(w):
            (cs if t[0] == _TYPE_CHO else vs if t[0] == _TYPE_JUNG else fs).add(t)
    return len(cs), len(vs), len(fs)


def search_space(words: List[str]) -> int:
    """정답 후보 공간(자모↔숫자 단사 배정 수) = P(10,kc)·P(10,kv)·P(10,kf).

    문서(module_algorithm_search_space.md) §2.1 의 cryptarithmetic 탐색공간 정의
    (L! 단사 수)와 동일한 '정답 후보 카디널리티'. 단, 3 독립표라 곱 형태.
    """
    kc, kv, kf = _distinct_jamo_counts(words)
    return _perm(10, kc) * _perm(10, kv) * _perm(10, kf)


# 탐색공간을 10!(=3,628,800) 과 같은 자릿수권의 '상수'로 고정(실제 단어 유지).
# 4-피연산자(5단어)는 필요한 distinct 초성 수 때문에 탐색공간 하한이 ~1.4×10⁷ 라,
# 전 난이도 공통 상수 밴드는 이 하한을 포함하도록 ~10⁷ 권으로 잡는다.
# (이 밴드 필터로 모든 난이도의 정답 후보 공간이 동일 자릿수권에 묶임 = 문서 Type B
#  '탐색공간 상수형' 특성에 정합. 정확히 10! 은 N=4 하한 때문에 불가 → 동일 자릿수권 근사.)
SEARCH_SPACE_BAND = (3_000_000, 30_000_000)


# 버킷: 자모 길이 -> 단어 리스트
_WORD_BY_JCOUNT: Dict[int, List[str]] = defaultdict(list)
for _w in dict.fromkeys(_WORD_POOL_FLAT):  # dedup, 순서 유지
    _WORD_BY_JCOUNT[_jcount(_w)].append(_w)


@dataclass
class PuzzleCandidate:
    operand_words: List[str]
    result: str
    answer: str
    operand_ints: List[int]
    mapping: Dict[str, Dict[str, int]] = None      # 전체(정답) 키
    revealed: Dict[str, Dict[str, int]] = None     # 공개된 부분 키
    solver_steps: int = 0

    @property
    def puzzle_str(self) -> str:
        return " + ".join(self.operand_words) + f" = {self.result}"

    @property
    def operands(self):
        return list(self.operand_words)


# ============================================================================
# Solver: 타입별(초성/중성/종성) 독립 치환 + 일부 토큰 고정(fixed)
# ============================================================================

class _NodeBudgetExceeded(Exception):
    pass


def find_solutions(
    puzzle: tuple,
    max_count: int = 4,
    _stats: Optional[Dict] = None,
    hangul: HangulCipherSystem = None,
    fixed: Optional[Dict[str, int]] = None,
    max_nodes: Optional[int] = None,
) -> List[Tuple[str, Dict[str, int]]]:
    """열별 백트래킹. 기호는 타입 태그된 자모 토큰, distinct 제약은 타입별 독립.
    fixed 로 일부 토큰을 고정(공개 키). max_nodes 초과 시 탐색 조기 중단(_stats['aborted']
    =True). 반환: (결과 단어 숫자값, token->digit) 리스트.
    """
    hangul = hangul or _HANGUL
    fixed = fixed or {}

    *operand_words, result_word = puzzle
    words = list(operand_words) + [result_word]
    word_tokens = [_decompose_word_to_tokens(w, hangul) for w in words]
    op_tokens = word_tokens[:-1]
    res_tokens = word_tokens[-1]

    all_tokens: Set[str] = set()
    for wt in word_tokens:
        all_tokens.update(wt)

    by_type: Dict[str, Set[str]] = defaultdict(set)
    for t in all_tokens:
        by_type[t[0]].add(t)
    for toks in by_type.values():
        if len(toks) > 10:
            return []

    first_tokens = set(wt[0] for wt in word_tokens if wt)

    max_len = max(len(wt) for wt in word_tokens)
    reversed_ops = [wt[::-1] for wt in op_tokens]
    wr = res_tokens[::-1]

    ordered: List[str] = []
    seen: Set[str] = set(fixed.keys())
    for col in range(max_len):
        for wt in reversed_ops + [wr]:
            if col < len(wt) and wt[col] not in seen:
                ordered.append(wt[col])
                seen.add(wt[col])
    for t in all_tokens:
        if t not in seen:
            ordered.append(t)
            seen.add(t)

    cols_op: List[Tuple[str, ...]] = []
    cols_cr: List[Optional[str]] = []
    for col in range(max_len):
        cols_op.append(tuple(rw[col] for rw in reversed_ops if col < len(rw)))
        cols_cr.append(wr[col] if col < len(wr) else None)

    solutions: List[Tuple[str, Dict[str, int]]] = []
    mapping: Dict[str, int] = dict(fixed)
    used: Dict[str, set] = {_TYPE_CHO: set(), _TYPE_JUNG: set(), _TYPE_JONG: set()}
    for _tok, _d in fixed.items():
        used[_tok[0]].add(_d)

    def _check_full(mp: Dict[str, int]) -> bool:
        carry = 0
        for col in range(max_len):
            ops = cols_op[col]
            cr = cols_cr[col]
            for ol in ops:
                if ol not in mp:
                    return True
            if cr is not None and cr not in mp:
                return True
            total = carry + sum(mp[ol] for ol in ops)
            dr = mp[cr] if cr is not None else 0
            if total % 10 != dr:
                return False
            carry = total // 10
        return carry == 0

    node_count = [0]

    def backtrack(idx: int):
        node_count[0] += 1
        if _stats is not None:
            _stats['nodes'] = node_count[0]
        if max_nodes is not None and node_count[0] > max_nodes:
            raise _NodeBudgetExceeded
        if len(solutions) >= max_count:
            return
        if idx == len(ordered):
            if _check_full(mapping):
                num_result = int("".join(str(mapping[t]) for t in res_tokens))
                solutions.append((str(num_result), dict(mapping)))
            return

        tok = ordered[idx]
        typ = tok[0]
        used_typ = used[typ]
        is_first = tok in first_tokens

        for digit in range(10):
            if digit in used_typ:
                continue
            if digit == 0 and is_first:
                continue

            mapping[tok] = digit
            used_typ.add(digit)

            valid = True
            carry = 0
            for col in range(max_len):
                ops = cols_op[col]
                cr = cols_cr[col]
                ready = True
                for ol in ops:
                    if ol not in mapping:
                        ready = False
                        break
                if ready and not (cr is None or cr in mapping):
                    ready = False
                if ready:
                    total = carry + sum(mapping[ol] for ol in ops)
                    dr = mapping[cr] if cr is not None else 0
                    if total % 10 != dr:
                        valid = False
                        break
                    carry = total // 10
                else:
                    break

            if valid:
                backtrack(idx + 1)

            del mapping[tok]
            used_typ.discard(digit)

    try:
        backtrack(0)
    except _NodeBudgetExceeded:
        if _stats is not None:
            _stats['aborted'] = True
    return solutions


# ============================================================================
# 산술 헬퍼
# ============================================================================

def count_carries(*nums: int) -> int:
    carries = 0
    carry = 0
    str_nums = [str(n)[::-1] for n in nums]
    max_len = max(len(s) for s in str_nums)
    for i in range(max_len):
        total = carry
        for s in str_nums:
            total += int(s[i]) if i < len(s) else 0
        carry = total // 10
        if carry > 0:
            carries += 1
    return carries


# ============================================================================
# 부분 키 선택 (유일해 유지 최소 공개)
# ============================================================================

def _select_revealed(puzzle: tuple, full_tokens: Dict[str, int], hide_ratio: float):
    """전체(정답) 키에서 일부를 가려도 정답이 유일하도록 공개 토큰 집합을 고른다."""
    tokens = list(full_tokens.keys())
    total = len(tokens)
    target_hide = max(1, round(hide_ratio * total))
    random.shuffle(tokens)

    revealed = dict(full_tokens)
    hidden = 0
    for tok in tokens:
        if hidden >= target_hide:
            break
        trial = {t: d for t, d in revealed.items() if t != tok}
        sols = find_solutions(puzzle, max_count=MAX_SOLUTIONS + 1, fixed=trial)
        if len(sols) == 1:
            del revealed[tok]
            hidden += 1

    if hidden == 0:
        return None
    return revealed, hidden


def _tokens_to_tables(tokens: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    out = {"cho": {}, "jung": {}, "jong": {}}
    for tok, d in tokens.items():
        out[_TYPE_KEYS[tok[0]]][tok[1:]] = d
    return out


def _tables_to_tokens(tables: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for typ, key in _TYPE_KEYS.items():
        for jamo, d in (tables or {}).get(key, {}).items():
            out[typ + jamo] = d
    return out


# ============================================================================
# 난이도 설정 & 생성 (실제 단어)
# ============================================================================
# 난이도 = 피연산자 수(주 레버) + 단어 길이(자모 수) + 가림 비율 + 종성 요구(3중 치환).
# 실제 추론 난이도를 난이도별로 벌리고, 종성표(받침)를 살려 초성/중성/종성 3중 치환을
# 유지한다. 목표 정확도(gemini-3-flash, docs §2.1): easy~75% · medium~50% · hard~20-25%.
#  - op_jcounts: 피연산자 단어의 자모 수 후보(클수록 자릿수↑·자모↑ → 난이도↑)
#  - min_jong:   서로 다른 종성(받침) 최소 수 → 3중 치환 보장(받침 단어 강제)
#  - hide_ratio: 가림 비율(추론량). 클수록 어려움
DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    "easy":   {"num_operands": 4, "op_jcounts": (6,),    "min_jong": 2,
               "hide_ratio": 0.45, "min_carries": 2, "max_attempts": 35000},
    "medium": {"num_operands": 4, "op_jcounts": (7,),    "min_jong": 2,
               "hide_ratio": 0.53, "min_carries": 2, "max_attempts": 50000},
    "hard":   {"num_operands": 4, "op_jcounts": (7, 8),  "min_jong": 3,
               "hide_ratio": 0.92, "min_carries": 3, "max_attempts": 70000},
}


def generate_puzzle_by_difficulty(
    difficulty: str,
    used_patterns: Set[str] = None,
    **overrides,
) -> Optional[PuzzleCandidate]:
    if used_patterns is None:
        used_patterns = set()

    config = dict(DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["easy"]))
    config.update(overrides)

    N = config["num_operands"]
    hide_ratio = config["hide_ratio"]
    min_carries = config.get("min_carries", 0)
    min_jong = config.get("min_jong", 0)
    op_jcounts = config.get("op_jcounts", (5, 6))
    max_attempts = config.get("max_attempts", 5000)

    # 피연산자 후보 길이: 지정된 자모 수 버킷 중 단어가 N개 이상 있는 것
    op_lens = [L for L in op_jcounts if len(_WORD_BY_JCOUNT.get(L, [])) >= N]
    if not op_lens:
        return None

    for _ in range(max_attempts):
        L = random.choice(op_lens)
        ops = random.sample(_WORD_BY_JCOUNT[L], N)
        # 결과 단어: 자모길이 L 또는 L+1 (합의 자릿수 = L 또는 L+1)
        res_cands = [w for Lr in (L, L + 1)
                     for w in _WORD_BY_JCOUNT.get(Lr, []) if w not in ops]
        if not res_cands:
            continue
        random.shuffle(res_cands)

        for r in res_cands[:8]:
            # 3중 치환 보장: 서로 다른 종성(받침) 최소 수 충족
            if min_jong > 0:
                _, _, kf = _distinct_jamo_counts(ops + [r])
                if kf < min_jong:
                    continue
            puzzle = tuple(ops + [r])
            # 노드 예산으로 해 없는/탐색비싼 조합을 빠르게 reject (>=1해 찾으면 진행).
            sols = find_solutions(puzzle, max_count=2, max_nodes=30000)
            if not sols:
                continue

            full_tokens = sols[0][1]
            answer = sols[0][0]
            operand_ints = [_word_value(w, full_tokens) for w in ops]
            if count_carries(*operand_ints) < min_carries:
                continue

            pattern = "+".join(ops) + "=" + r
            if pattern in used_patterns:
                continue

            sel = _select_revealed(puzzle, full_tokens, hide_ratio)
            if sel is None:
                continue
            revealed_tokens, _hidden = sel

            stats = {'nodes': 0}
            s2 = find_solutions(puzzle, max_count=MAX_SOLUTIONS + 1,
                                fixed=revealed_tokens, _stats=stats)
            if len(s2) != 1 or s2[0][0] != answer:
                continue

            used_patterns.add(pattern)
            return PuzzleCandidate(
                operand_words=ops,
                result=r,
                answer=answer,
                operand_ints=operand_ints,
                mapping=_tokens_to_tables(full_tokens),
                revealed=_tokens_to_tables(revealed_tokens),
                solver_steps=stats['nodes'],
            )

    return None


# ============================================================================
# 추론 연쇄 (teacher trace)
# ============================================================================

def _deduction_steps(
    puzzle: tuple,
    revealed_tokens: Dict[str, int],
    full_tokens: Dict[str, int],
    operand_ints: List[int],
    answer: str,
) -> List[str]:
    """공개 키에서 출발해 미지 자모를 하나씩 '강제'로 확정하는 추론 연쇄."""
    *operand_words, result_word = puzzle
    words = list(operand_words) + [result_word]
    word_tokens = [_decompose_word_to_tokens(w) for w in words]
    op_tokens = word_tokens[:-1]
    res_tokens = word_tokens[-1]
    max_len = max(len(wt) for wt in word_tokens)
    rev_ops = [wt[::-1] for wt in op_tokens]
    rev_res = res_tokens[::-1]
    first_tokens = set(wt[0] for wt in word_tokens if wt)

    op_strs = [str(n) for n in operand_ints]
    L = max(len(s) for s in op_strs + [answer])
    pop = [s.rjust(L) for s in op_strs]
    carries = [0] * (L + 1)
    for pos in range(L):
        ci = L - 1 - pos
        s = sum(int(x[ci]) for x in pop if x[ci] != ' ') + carries[pos]
        carries[pos + 1] = s // 10

    known = dict(revealed_tokens)
    used = {_TYPE_CHO: set(), _TYPE_JUNG: set(), _TYPE_JONG: set()}
    for t, d in known.items():
        used[t[0]].add(d)
    remaining = [t for t in full_tokens if t not in known]

    def _forced(t: str) -> Optional[int]:
        typ = t[0]
        cands = []
        for d in range(10):
            if d in used[typ]:
                continue
            if d == 0 and t in first_tokens:
                continue
            if find_solutions(puzzle, max_count=1, fixed={**known, t: d}):
                cands.append(d)
                if len(cands) > 1:
                    break
        return cands[0] if len(cands) == 1 else None

    def _single_unknown_col(t: str):
        for pos in range(max_len):
            col_ops = [ro[pos] for ro in rev_ops if pos < len(ro)]
            col_res = rev_res[pos] if pos < len(rev_res) else None
            toks = col_ops + ([col_res] if col_res else [])
            if t in toks and all(o in known for o in toks if o != t):
                return pos, col_ops, col_res
        return None

    steps: List[str] = []
    step_no = 0
    while remaining:
        picked = None
        for t in remaining:
            d = _forced(t)
            if d is not None:
                picked = (t, d)
                break
        if picked is None:
            labels = ", ".join(
                f"{t[1:]}({_TYPE_LABEL_KO[_TYPE_KEYS[t[0]]]})" for t in remaining)
            steps.append(f"  · [추론] 남은 자모 {labels} 는 여러 열 제약을 동시에 "
                         f"만족하는 유일한 배정으로 확정된다.")
            for t in remaining:
                known[t] = full_tokens[t]
            break

        t, d = picked
        step_no += 1
        jamo = t[1:]
        tlabel = _TYPE_LABEL_KO[_TYPE_KEYS[t[0]]]
        col = _single_unknown_col(t)
        if col:
            pos, col_ops, col_res = col
            cin = carries[pos]
            cout = carries[pos + 1]
            parts = [f"[{jamo}=?]" if o == t else f"{full_tokens[o]}" for o in col_ops]
            terms = " + ".join(parts)
            if cin:
                terms += f" + 받아올림 {cin}"
            colsum = sum(full_tokens[o] for o in col_ops) + cin
            if col_res == t:
                steps.append(f"  · [추론 {step_no}] 자리 {pos + 1}(우→좌, {tlabel}열): "
                             f"{terms} = {colsum} → 끝자리가 {d} 이므로 결과 {tlabel} {jamo}={d}.")
            else:
                rd = full_tokens[col_res] if col_res else 0
                steps.append(f"  · [추론 {step_no}] 자리 {pos + 1}(우→좌, {tlabel}열): "
                             f"{terms} = 결과 {rd}(받아올림 {cout}) 가 되려면 {tlabel} {jamo}={d}.")
        else:
            steps.append(f"  · [추론 {step_no}] {tlabel} {jamo}: 공개값과 기확정값, "
                         f"모든 열 제약을 함께 보면 가능한 값은 {d} 뿐 → {jamo}={d}.")
        known[t] = d
        used[t[0]].add(d)
        remaining.remove(t)

    return steps


# ============================================================================
# 문제 텍스트
# ============================================================================

def _format_table(table: Dict[str, int]) -> str:
    if not table:
        return "(공개 없음)"
    return ", ".join(f"{j}={d}" for j, d in sorted(table.items(), key=lambda kv: kv[0]))


def _annotated_table(full: Dict[str, int], revealed: Dict[str, int]) -> str:
    parts = []
    for j, d in sorted(full.items(), key=lambda kv: kv[0]):
        tag = "공개" if j in revealed else "추론"
        parts.append(f"{j}={d}[{tag}]")
    return ", ".join(parts) if parts else "(없음)"


def _decompose_example(word: str, mapping: Dict[str, Dict[str, int]]) -> str:
    ch = word[0]
    cho, jung, jong = _HANGUL.decompose(ch)
    if cho < 0:
        return ""
    cj, vj = _HANGUL.CHO[cho], _HANGUL.JUNG[jung]
    if jong > 0:
        fj = _HANGUL.JONG[jong]
        return f"(예: 음절 '{ch}' = 초성 {cj} + 중성 {vj} + 종성 {fj} → 세 자리; 받침 없으면 두 자리)"
    return f"(예: 음절 '{ch}' = 초성 {cj} + 중성 {vj} → 두 자리; 받침 있으면 세 자리)"


def create_question(candidate: PuzzleCandidate) -> str:
    operand_words = candidate.operands
    revealed = candidate.revealed or {"cho": {}, "jung": {}, "jong": {}}

    cho_t = _format_table(revealed.get("cho", {}))
    jung_t = _format_table(revealed.get("jung", {}))
    jong_t = _format_table(revealed.get("jong", {}))

    op_lines = f"  {operand_words[0]}\n"
    for w in operand_words[1:]:
        op_lines += f"+ {w}\n"
    max_word_len = max(len(w) for w in operand_words + [candidate.result])
    separator = '-' * (max_word_len + 2)

    question = (
        "한글 자모(초성·중성·종성) 기반 복면산 퍼즐을 풀어주세요. "
        "각 한글 음절을 초성/중성/종성으로 분해합니다. 단어의 숫자값은 음절을 "
        "왼쪽→오른쪽, 각 음절 안에서는 초성→중성→종성 순으로 각 자모에 대응하는 숫자를 "
        "이어붙여 만듭니다(받침이 없는 음절은 초성·중성 두 자리).\n"
        "- 초성끼리: 서로 다른 초성은 서로 다른 숫자(0–9), 같은 초성은 같은 숫자.\n"
        "- 중성끼리·종성끼리도 각각 마찬가지(서로 다른 자모 = 서로 다른 숫자).\n"
        "- 초성표·중성표·종성표는 서로 독립입니다(초성 ㄱ=1이어도 종성 ㄱ은 다를 수 있음).\n"
        "- 각 단어의 첫 초성은 0이 될 수 없습니다.\n"
        "아래 **일부 자모의 값만 공개**되어 있습니다. 나머지 자모의 값은 덧셈이 "
        "성립하도록 추론해야 합니다.\n"
        f"{_decompose_example(operand_words[0], candidate.mapping)}\n\n"
        f"[공개된 초성] {cho_t}\n"
        f"[공개된 중성] {jung_t}\n"
        f"[공개된 종성] {jong_t}\n\n"
        f"아래 덧셈이 성립하도록 공개되지 않은 자모↔숫자를 추론하고, "
        f"{candidate.result}이(가) 나타내는 숫자 값을 구하세요.\n\n"
        f"{op_lines}"
        f"{separator}\n"
        f"= {candidate.result}"
    )
    return question


# ============================================================================
# Guided-distillation style solution (teacher trace)
# ============================================================================

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_cryptarithmetic_solution_ko(
    candidate: PuzzleCandidate,
    difficulty: str,
    carries: int,
) -> str:
    operand_words = candidate.operands
    result_word = candidate.result
    full = candidate.mapping or {"cho": {}, "jung": {}, "jong": {}}
    revealed = candidate.revealed or {"cho": {}, "jung": {}, "jong": {}}
    answer = candidate.answer
    operand_ints = candidate.operand_ints or []

    op_lines_words = " + ".join(operand_words) + f" = {result_word}"
    op_lines_digits = " + ".join(str(n) for n in operand_ints) + f" = {answer}"

    n_hidden = sum(len(full[t]) - len(revealed.get(t, {})) for t in ("cho", "jung", "jong"))
    n_total = sum(len(full[t]) for t in ("cho", "jung", "jong"))

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {difficulty}",
        f"  - 퍼즐: {op_lines_words}",
        f"  - 자모 수: 총 {n_total} (공개 {n_total - n_hidden} · 추론 {n_hidden})",
        f"  - 받아올림(carry) 수: {carries}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 각 음절을 초성/중성/종성으로 분해(받침 없는 음절은 초성·중성 2자리).",
        "  - 단어 숫자값 = 음절 왼→오, 음절 내 초성→중성→종성 순으로 자릿수 연결.",
        "  - 초성끼리/중성끼리/종성끼리 서로 다른 자모 = 서로 다른 숫자. 세 표는 독립.",
        "  - 각 단어의 첫 초성 ≠ 0.",
        f"  - 공개된 초성: {_format_table(revealed.get('cho', {}))}",
        f"  - 공개된 중성: {_format_table(revealed.get('jung', {}))}",
        f"  - 공개된 종성: {_format_table(revealed.get('jong', {}))}",
        f"  - 구하려는 값: {result_word}의 숫자값.",
        "[STEP 2] 풀이 전개 (공개 키에서 출발해 열별 덧셈 제약으로 미지 자모를 차례로 확정)",
    ]
    lines.extend(_deduction_steps(
        tuple(operand_words + [result_word]),
        _tables_to_tokens(revealed),
        _tables_to_tokens(full),
        operand_ints, answer,
    ))
    lines.extend([
        "  · 확정된 전체 키([공개]=주어짐, [추론]=위에서 확정):",
        f"    - 초성표: {_annotated_table(full.get('cho', {}), revealed.get('cho', {}))}",
        f"    - 중성표: {_annotated_table(full.get('jung', {}), revealed.get('jung', {}))}",
        f"    - 종성표: {_annotated_table(full.get('jong', {}), revealed.get('jong', {}))}",
        f"  · 매핑 검증(숫자로 치환): {op_lines_digits}",
    ])

    # 열별 받아올림 전개 (우→좌)
    operand_digit_strs = [str(n) for n in operand_ints]
    max_len = max(len(s) for s in operand_digit_strs + [answer])
    padded_ops = [s.rjust(max_len) for s in operand_digit_strs]
    padded_res = answer.rjust(max_len)

    carry = 0
    for pos in range(max_len):
        col_idx = max_len - 1 - pos
        col_digits = [int(s[col_idx]) for s in padded_ops if s[col_idx] != ' ']
        s = sum(col_digits) + carry
        new_carry = s // 10
        res_d = padded_res[col_idx]
        terms = " + ".join(str(d) for d in col_digits) if col_digits else "0"
        if carry:
            terms += f" + 받아올림 {carry}"
        verdict = f" → 결과 자리 {res_d} (받아올림 {new_carry})" if res_d != ' ' else f" → 받아올림 {new_carry}"
        lines.append(f"    [SEG {pos + 1}] 자리 {pos + 1}(우→좌): {terms} = {s}{verdict}")
        carry = new_carry

    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답: {result_word} = {answer}",
        "  - 각 자리 합과 받아올림이 일치하는지 [SEG] 전개로 재확인.",
        "  - 모든 단어의 첫 초성 숫자가 0이 아닌지 확인.",
        "  - 같은 타입(초성/중성/종성) 안에서 서로 다른 자모가 서로 다른 숫자인지 확인.",
    ])
    return "\n".join(lines)


# ============================================================================
# 데이터셋 생성
# ============================================================================

def create_dataset_files(num_questions: int):
    import pandas as pd

    print(f"{num_questions}개의 자모 복면산(실제 단어·부분키 추론) 퍼즐을 생성합니다...")

    difficulties = ["easy", "medium", "hard"]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []
    used_patterns = set()

    for i, difficulty in enumerate(difficulties):
        target_count = puzzles_per_diff + (1 if i < remainder else 0)
        if target_count == 0:
            continue

        print(f"\n=== {difficulty} 퍼즐 생성 중 ({target_count}개 필요) ===")
        generated = 0
        attempts = 0
        max_total_attempts = max(5000, target_count * 400)

        while generated < target_count and attempts < max_total_attempts:
            attempts += 1
            candidate = generate_puzzle_by_difficulty(
                difficulty, used_patterns=used_patterns,
            )
            if candidate:
                carries = count_carries(*candidate.operand_ints)
                puzzle_data = {
                    "id": f"cryptarithmetic_ko_{difficulty}_{generated:04d}",
                    "question": create_question(candidate),
                    "answer": candidate.answer,
                    "solution": _build_cryptarithmetic_solution_ko(
                        candidate, difficulty, carries
                    ),
                    "difficulty": difficulty,
                }
                all_puzzles.append(puzzle_data)
                generated += 1
                print(f"  [{generated}/{target_count}] {candidate.puzzle_str} -> "
                      f"{candidate.answer} (carries={carries}, steps={candidate.solver_steps})")

        if generated < target_count:
            print(f"  경고: {difficulty} 퍼즐을 {target_count}개 중 {generated}개만 생성했습니다")

    print(f"\n총 {len(all_puzzles)}개의 퍼즐을 생성했습니다")

    df = pd.DataFrame(all_puzzles)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("cryptarithmetic_ko", all_puzzles)

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="복면산(Cryptarithmetic) 퍼즐 생성기 - 한국어 자모 실제단어 부분키 추론")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")
    args = parser.parse_args()

    create_dataset_files(num_questions=args.num)

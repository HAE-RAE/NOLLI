"""한글 자모 합성 퍼즐 생성기 (KO, 2축: 한국어 표기·구조 의존)

생년월일 같은 외부 지식이 아니라 **한글의 음절 블록 구조**(초성·중성·종성 합성,
겹받침)에 난이도가 의존하는 task. 영어로 옮기면 단순 시저 암호로 붕괴하므로
번역 불가 → 진짜 2축. 규칙은 프롬프트에 자족적으로 주어지므로 문화 지식 불요.

과제: 각 글자를 (초성, 중성, 종성)으로 분해 → 초성만 정해진 칸수 순환 이동 →
      다시 한 글자로 합쳐 결과 문자열을 만든다.

난이도(프로토타입 측정, gemini-3-flash-preview reasoning=medium):
    n=3 + 겹받침 → 25% (hard 밴드).  난이도 노브 = 음절 수 · 겹받침 유무.
    easy   -> 2음절, 단받침만(겹받침 없음)
    medium -> 3음절, 단받침만
    hard   -> 3음절, 겹받침 포함

답은 변환된 한글 문자열 1개로 수렴. 완전 결정론(유니코드 합성, 어휘 예외 0).

실행:  python jamo_ko.py --num 300 --seed 0
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"           # 19
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"        # 21
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹ") + \
    ["ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ"] + list("ㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")  # 28
assert len(JONG) == 28

# 겹받침 종성 인덱스(분해/재조합이 까다로운 부분)
DOUBLE_JONG = {3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 18}
SINGLE_JONG = [i for i in range(28) if i not in DOUBLE_JONG]  # 0(없음) 포함

SHIFT = 1  # 초성 이동 칸수

# 난이도 측정(gemini reasoning=medium): n2-none=easy 후보, n2-single=45%(medium),
# n3-mixed≈hard. 모델이 자모 조작에 약해 전 구간이 아래로 쏠리므로 받침 제거로 easy 확보.
TIER_CFG = {
    "easy":   {"n": 2, "jong": "light"},
    "medium": {"n": 2, "jong": "single"},
    "hard":   {"n": 3, "jong": "mixed"},
}


def compose(c: int, j: int, t: int) -> str:
    return chr(0xAC00 + (c * 21 + j) * 28 + t)


def decompose(ch: str):
    n = ord(ch) - 0xAC00
    return n // 588, (n // 28) % 21, n % 28


def _rand_syllable(rng, jong_mode: str):
    c = rng.randint(0, 18)
    j = rng.randint(0, 20)
    if jong_mode == "none":
        t = 0
    elif jong_mode == "light":          # 일부 음절만 단받침 (받침 절벽 완화)
        t = rng.choice(SINGLE_JONG) if rng.random() < 0.4 else 0
    elif jong_mode == "single":
        t = rng.choice(SINGLE_JONG)
    elif jong_mode == "mixed":
        t = rng.randint(0, 27) if rng.random() < 0.5 else rng.choice(SINGLE_JONG)
    else:  # any
        t = rng.randint(0, 27)
    return c, j, t


def _cho_list_str() -> str:
    return " ".join(f"{i}:{ch}" for i, ch in enumerate(CHO))


def generate_one(difficulty: str, rng: random.Random):
    cfg = TIER_CFG[difficulty]
    syls = [_rand_syllable(rng, cfg["jong"]) for _ in range(cfg["n"])]
    word = "".join(compose(*s) for s in syls)
    out = "".join(compose((c + SHIFT) % 19, j, t) for c, j, t in syls)

    steps = [f"[STEP 1] 초성 순서: {_cho_list_str()}"]
    for idx, (c, j, t) in enumerate(syls, 1):
        c2 = (c + SHIFT) % 19
        jong_str = JONG[t] if JONG[t] else "없음"
        steps.append(
            f"[STEP 2.{idx}] '{compose(c, j, t)}' = 초성 {CHO[c]}(idx {c}) + 중성 {JUNG[j]} + 종성 {jong_str}"
            f" → 초성 {SHIFT}칸 이동 {CHO[c]}→{CHO[c2]}(idx {c2}) → '{compose(c2, j, t)}'"
        )
    steps.append(f"[STEP 3] 합치면 = '{out}'")

    q = (
        "한글 자모 변환 문제입니다.\n"
        f"초성 순서(0부터): {_cho_list_str()}\n"
        f"규칙: 각 글자를 초성·중성·종성으로 분해한 뒤, 초성만 위 순서에서 "
        f"{SHIFT}칸 뒤로 순환 이동(맨 끝이면 처음으로)시키고 중성·종성은 그대로 두어 "
        f"다시 한 글자로 합치세요.\n"
        f"글자: '{word}'\n"
        f"변환된 문자열을 구하세요. 마지막 줄에 'Answer: 결과' 형식으로 답하세요."
    )
    return q, out, steps


def build_solution_trace(steps: List[str], answer: str, difficulty: str) -> str:
    cfg = TIER_CFG[difficulty]
    solution = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 한글 자모 변환: {cfg['n']}음절, 종성={cfg['jong']}; 초성 {SHIFT}칸 순환 이동",
        "[STEP 1] 주어진 조건",
    ]
    for s in steps:
        if s.startswith("[STEP 1]"):
            solution.append("  - " + s[len("[STEP 1] "):])
    solution.append("[STEP 2] 풀이 전개")
    for s in steps:
        if s.startswith("[STEP 2"):
            solution.append("  " + s)
    solution.append("[STEP 3] 답·검산")
    for s in steps:
        if s.startswith("[STEP 3]"):
            solution.append("  - " + s[len("[STEP 3] "):])
    return "\n".join(solution)


def create_dataset_files(num_questions: int, seed: int = None):
    import pandas as pd

    difficulties = ["easy", "medium", "hard"]
    per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)
    rng = random.Random(seed)

    all_records: List[Dict] = []
    for di, difficulty in enumerate(difficulties):
        count = per_diff + (1 if di < remainder else 0)
        if count == 0:
            continue
        print(f"\n=== {difficulty} 퍼즐 생성 ({count}개 필요) ===")
        seen, records, attempts = set(), [], 0
        while len(records) < count and attempts < count * 400:
            attempts += 1
            q, a, steps = generate_one(difficulty, rng)
            if q in seen:
                continue
            seen.add(q)
            idx = len(records)
            records.append({
                "id": f"jamo_ko_{difficulty}_{idx:04d}",
                "question": q,
                "answer": a,
                "solution": build_solution_trace(steps, a, difficulty),
                "difficulty": difficulty,
            })
        print(f"  생성: {len(records)}개")
        all_records.extend(records)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("jamo_ko", all_records)
    print(f"\n총 {len(all_records)}개 퍼즐 생성")
    df = pd.DataFrame(all_records)
    return df, all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한글 자모 합성 퍼즐 생성기 (KO)")
    parser.add_argument("--num", type=int, default=300, help="총 퍼즐 수 (3등분)")
    parser.add_argument("--seed", type=int, default=None, help="재현용 랜덤 시드")
    args = parser.parse_args()
    print("=" * 60)
    print("한글 자모 합성 퍼즐 생성기 (KO, 2축)")
    print("=" * 60)
    create_dataset_files(num_questions=args.num, seed=args.seed)

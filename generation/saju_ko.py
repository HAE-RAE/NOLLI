"""사주(四柱) 만세력 퍼즐 생성기 (KO)

kinship에 이은 **한국 특화** task. 생년월일시 → 사주 4기둥(연·월·일·시주)의 60갑자.
난이도는 "모델이 추론으로 복원할 수 없는 만세력 지식"에서 나온다(프로토타입 측정: 일주
20% / 시주 25% — gemini-3-flash-preview thinking medium 기준 15-35% 밴드 자연 안착).

기둥별 계산(전부 결정론적, ground truth 보장):
    연주  연(年) 간지, 입춘(立春) 경계로 해가 바뀜              [easy]
    월주  절기(節) 기준 월지 + 월두법(五虎遁)으로 월간          [medium]
    일주  연속 60갑자 일진 = 간지((JDN+49)%60)                  [hard]
    시주  시지(2시간 블록) + 시두법(五鼠遁)으로 시간            [hard]

절기는 PyEphem으로 태양 황경(당일분점, epoch=date) 15° 배수 교차 시각을 계산(KASI 2024
12절기와 전부 일치 검증). 일주는 KoreanLunarCalendar로 교차검증.

의존성: ephem, korean_lunar_calendar
실행:  python saju_ko.py --num 300 --seed 0
"""

import argparse
import datetime
import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import ephem

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)

GAN = "갑을병정무기경신임계"
JI = "자축인묘진사오미신유술해"
ZODIAC = ["쥐", "소", "호랑이", "토끼", "용", "뱀", "말", "양", "원숭이", "닭", "개", "돼지"]
KST = datetime.timedelta(hours=9)

# 12 節 (월 시작 절기): (태양황경, 한글, 근사 월일, 월지 index[자=0])
JIE = [
    (315, "입춘", (2, 4), 2), (345, "경칩", (3, 5), 3), (15, "청명", (4, 5), 4),
    (45, "입하", (5, 5), 5), (75, "망종", (6, 6), 6), (105, "소서", (7, 7), 7),
    (135, "입추", (8, 7), 8), (165, "백로", (9, 7), 9), (195, "한로", (10, 8), 10),
    (225, "입동", (11, 7), 11), (255, "대설", (12, 7), 0), (285, "소한", (1, 6), 1),
]


# ---------------------------------------------------------------------------
# 절기 엔진 (태양 황경)
# ---------------------------------------------------------------------------
def _sun_lon(dt_utc: datetime.datetime) -> float:
    return math.degrees(ephem.Ecliptic(ephem.Sun(dt_utc), epoch=dt_utc).lon) % 360.0


def _signed(lon: float, target: float) -> float:
    return ((lon - target + 180) % 360) - 180


@lru_cache(maxsize=4096)
def jeolgi_instant_utc(year: int, target: int, approx_m: int, approx_d: int) -> float:
    """주어진 해의 특정 절기(태양황경 target) 통과 시각(ephem.Date float, UTC)."""
    lo = ephem.Date(datetime.datetime(year, approx_m, approx_d) - datetime.timedelta(days=8))
    hi = ephem.Date(datetime.datetime(year, approx_m, approx_d) + datetime.timedelta(days=8))
    for _ in range(60):
        mid = ephem.Date((lo + hi) / 2)
        if _signed(_sun_lon(mid.datetime()), target) < 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def ipchun_kst(year: int) -> datetime.datetime:
    inst = jeolgi_instant_utc(year, 315, 2, 4)
    return ephem.Date(inst).datetime() + KST


# ---------------------------------------------------------------------------
# 기둥 계산
# ---------------------------------------------------------------------------
def _jdn(y: int, m: int, d: int) -> int:
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    return d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045


def ilju(y: int, m: int, d: int) -> str:
    i = (_jdn(y, m, d) + 49) % 60
    return GAN[i % 10] + JI[i % 12]


def year_pillar(dt_kst: datetime.datetime) -> Tuple[str, int]:
    """연주 + 연간 index. 입춘 이전 출생은 전년."""
    saju_year = dt_kst.year if dt_kst >= ipchun_kst(dt_kst.year) else dt_kst.year - 1
    yg = (saju_year - 4) % 10
    return GAN[yg] + JI[(saju_year - 4) % 12], yg


def _month_branch_index(dt_kst: datetime.datetime) -> int:
    """절기 기준 월지 index (자=0). 출생 시각의 태양 황경으로 결정."""
    L = _sun_lon((dt_kst - KST))
    k = int(((L - 315) % 360) // 30)  # 0 = 인月(입춘~)
    return (2 + k) % 12


def month_pillar(dt_kst: datetime.datetime) -> str:
    yg = year_pillar(dt_kst)[1]
    zi = _month_branch_index(dt_kst)
    start = (yg % 5) * 2 + 2          # 월두법: 인月 천간 시작
    order = (zi - 2) % 12             # 인月부터의 순서
    return GAN[(start + order) % 10] + JI[zi]


def _hour_branch_index(hour: int) -> int:
    return ((hour + 1) // 2) % 12     # 자(0) = 23~01시


def hour_pillar(dt_kst: datetime.datetime, hour: int) -> str:
    dg = GAN.index(ilju(dt_kst.year, dt_kst.month, dt_kst.day)[0])  # 일간(천간)
    zi = _hour_branch_index(hour)
    start = (dg % 5) * 2              # 시두법: 자시 천간 시작
    return GAN[(start + zi) % 10] + JI[zi]


# ---------------------------------------------------------------------------
# tier별 레시피 (단일 간지 답)
# ---------------------------------------------------------------------------
def _rand_dt(rng, with_hour=False):
    y = rng.randint(1950, 2015)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    h = rng.randint(0, 23) if with_hour else 12
    return datetime.datetime(y, m, d, h), h


def e_year_pillar(rng):
    dt, _ = _rand_dt(rng)
    a, _ = year_pillar(dt)
    ip = ipchun_kst(dt.year)
    note = "입춘 이전 → 전년 간지" if dt < ip else "입춘 이후 → 당년 간지"
    steps = [
        f"[STEP 1] {dt.year}년 입춘 = {ip.month}월 {ip.day}일 ({note})",
        f"[STEP 2] 연주(年柱) = {a}",
    ]
    q = f"양력 {dt.year}년 {dt.month}월 {dt.day}일에 태어난 사람의 사주 연주(年柱) 간지는? (입춘 기준, 예: 갑자)"
    return q, a, steps, "연주(年柱) 계산"


def m_month_pillar(rng):
    dt, h = _rand_dt(rng, with_hour=True)
    a = month_pillar(dt)
    yg = year_pillar(dt)[1]
    zi = _month_branch_index(dt)
    steps = [
        f"[STEP 1] 연간 = {GAN[yg]}; 절기 기준 월지 = {JI[zi]}월",
        f"[STEP 2] 월두법(五虎遁) 적용 → 월주(月柱) = {a}",
    ]
    q = (f"양력 {dt.year}년 {dt.month}월 {dt.day}일 {h}시에 태어난 사람의 사주 월주(月柱) 간지는? "
         f"(절기 기준 월지 + 월두법, 예: 갑자)")
    return q, a, steps, "월주(月柱) 계산"


def m_hour_pillar_given_day(rng):
    """일주를 제공하고 시두법(五鼠遁)만 적용 → 시주. 일진 지식 불필요(medium ~60%)."""
    gi = rng.randint(0, 59)               # 유효한 60갑자 일주
    dg = gi % 10
    ilju_str = GAN[dg] + JI[gi % 12]
    h = rng.randint(0, 23)
    zi = _hour_branch_index(h)
    start = (dg % 5) * 2
    a = GAN[(start + zi) % 10] + JI[zi]
    steps = [
        f"[STEP 1] 주어진 일주 '{ilju_str}'의 일간 = {GAN[dg]}; 시지 = {JI[zi]}시 ({h}시)",
        f"[STEP 2] 시두법(五鼠遁) 적용 → 시주(時柱) = {a}",
    ]
    q = (f"사주에서 일주(日柱)가 '{ilju_str}'인 사람이 {h}시에 태어났다. "
         f"시두법(五鼠遁)을 적용한 시주(時柱) 간지는? (예: 갑자)")
    return q, a, steps, "시주(時柱) 계산 (일주 제공)"


def h_day_pillar(rng):
    dt, _ = _rand_dt(rng)
    a = ilju(dt.year, dt.month, dt.day)
    jdn = _jdn(dt.year, dt.month, dt.day)
    i = (jdn + 49) % 60
    steps = [
        f"[STEP 1] 양력 {dt.year}년 {dt.month}월 {dt.day}일이 주어짐 (일주는 날짜로 정해지는 연속 60갑자 일진).",
        f"[STEP 2] 율리우스일수 JDN={jdn}; 간지 인덱스 = (JDN+49) mod 60 = {i} "
        f"→ 천간 {GAN[i % 10]} · 지지 {JI[i % 12]} → 일주(日柱) = {a}",
    ]
    q = f"양력 {dt.year}년 {dt.month}월 {dt.day}일의 사주 일주(日柱) 간지는? (60갑자 일진, 예: 갑자)"
    return q, a, steps, "일주(日柱) 계산"


def h_hour_pillar(rng):
    dt, h = _rand_dt(rng, with_hour=True)
    a = hour_pillar(dt, h)
    dg = ilju(dt.year, dt.month, dt.day)
    steps = [
        f"[STEP 1] 일간 = {dg[0]}; 시지 = {JI[_hour_branch_index(h)]}시 ({h}시)",
        f"[STEP 2] 시두법(五鼠遁) 적용 → 시주(時柱) = {a}",
    ]
    q = (f"양력 {dt.year}년 {dt.month}월 {dt.day}일 {h}시에 태어난 사람의 사주 시주(時柱) 간지는? "
         f"(일간 + 시두법, 예: 갑자)")
    return q, a, steps, "시주(時柱) 계산"


RECIPES = {
    # Latest calibration:
    # - year pillar alone scored 97%, and hour-with-given-day scored 87%.
    # - day/hour raw pillar tasks scored 24%, so mix them into easy/medium
    #   to create intermediate target bands without changing the evaluator.
    "easy": [
        m_hour_pillar_given_day,
        m_hour_pillar_given_day,
        m_hour_pillar_given_day,
        m_hour_pillar_given_day,
        h_day_pillar,
    ],
    "medium": [
        m_hour_pillar_given_day,
        m_hour_pillar_given_day,
        h_day_pillar,
        h_hour_pillar,
        h_day_pillar,
    ],
    "hard": [
        h_day_pillar,
        h_hour_pillar,
        h_day_pillar,
        h_hour_pillar,
        h_day_pillar,
        h_hour_pillar,
        h_day_pillar,
        h_hour_pillar,
        m_hour_pillar_given_day,
    ],
}


def generate_one(difficulty: str, rng: random.Random):
    return rng.choice(RECIPES[difficulty])(rng)


def build_solution_trace(steps: List[str], answer: str, q_type: str) -> str:
    solution = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 사주 기둥 계산 유형: {q_type}",
        "[STEP 1] 주어진 조건",
    ]
    for s in steps:
        if s.startswith("[STEP 1]"):
            solution.append("  - " + s[len("[STEP 1] "):])
    step2_lines = [s for s in steps if s.startswith("[STEP 2]")]
    if step2_lines:
        solution.append("[STEP 2] 풀이 전개")
        for s in step2_lines:
            solution.append("  - " + s[len("[STEP 2] "):])
    solution.append(f"[STEP 3] 답·검산\n  - 정답: {answer}")
    return "\n".join(solution)


# ---------------------------------------------------------------------------
# 데이터셋 조립
# ---------------------------------------------------------------------------
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
            try:
                q, a, steps, q_type = generate_one(difficulty, rng)
            except Exception:
                continue
            if q in seen:
                continue
            seen.add(q)
            idx = len(records)
            records.append({
                "id": f"saju_ko_{difficulty}_{idx:04d}",
                "question": q,
                "answer": a,
                "solution": build_solution_trace(steps, a, q_type),
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

    write_release_jsonl("saju_ko", all_records)
    print(f"\n총 {len(all_records)}개 퍼즐 생성")
    df = pd.DataFrame(all_records)
    return df, all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="사주 만세력 퍼즐 생성기 (KO)")
    parser.add_argument("--num", type=int, default=300, help="총 퍼즐 수 (3등분)")
    parser.add_argument("--seed", type=int, default=None, help="재현용 랜덤 시드")
    args = parser.parse_args()
    print("=" * 60)
    print("사주(四柱) 만세력 퍼즐 생성기 (KO)")
    print("=" * 60)
    create_dataset_files(num_questions=args.num, seed=args.seed)

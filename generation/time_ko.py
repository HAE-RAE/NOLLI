"""한국어 날짜·만세력 추론 퍼즐 생성기 (time_ko).

문제: 양력 공휴일 기준으로 상대일(오늘/내일/어제 …) + N일 후 날짜를 구한 뒤,
그 날짜에 대해 *출력 유형*에 맞춰 답한다.
  - date    : 그 날의 양력 날짜 ('YYYY.M.D')
  - weekday : 그 날의 요일 ('X요일')
  - ganji   : 그 날의 일진(日辰) 60갑자  (만세력 — saju_ko.ilju 재사용)

난이도(easy/medium/hard) = "from-scratch 일진(60갑자) 문제의 비율"(saju 식 recipe-mix).
일진(만세력 JDN→간지)은 gemini-3-flash 취약 축이고 날짜/요일은 강점이라, 둘을 비율로 섞으면
정답률을 연속 조절할 수 있다. 정식 evaluator(system prompt 포함) N=100 측정:
일진 비율 0.40 / 0.60 / 0.91 → 정답률 ≈ 75 / 50 / 24.

달력 변환은 lunarcalendar, 일진은 saju_ko.ilju 사용.
"""
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lunarcalendar import Lunar, Converter


# --------------------------------------------------------------------------- #
# 데이터 (공휴일 / 상대일 표현)
# --------------------------------------------------------------------------- #
# calendar: 공휴일 날짜(dates 의 "M.D")가 어느 달력 기준인지.
HOLIDAYS: List[Dict] = [
    {"name": "새해 첫날", "dates": ["1.1"], "calendar": "양력"},
    {"name": "설날 연휴", "dates": ["1.1"], "calendar": "음력"},
    {"name": "3·1절", "dates": ["3.1"], "calendar": "양력"},
    {"name": "부처님오신날", "dates": ["4.8"], "calendar": "음력"},
    {"name": "어린이날", "dates": ["5.5"], "calendar": "양력"},
    {"name": "현충일", "dates": ["6.6"], "calendar": "양력"},
    {"name": "광복절", "dates": ["8.15"], "calendar": "양력"},
    {"name": "추석 연휴", "dates": ["8.15"], "calendar": "음력"},
    {"name": "개천절", "dates": ["10.3"], "calendar": "양력"},
    {"name": "한글날", "dates": ["10.9"], "calendar": "양력"},
    {"name": "성탄절", "dates": ["12.25"], "calendar": "양력"},
    {"name": "제헌절", "dates": ["7.17"], "calendar": "양력"},
    {"name": "식목일", "dates": ["4.5"], "calendar": "양력"},
]

# 다일(연휴) 공휴일: 첫날/둘째날/마지막날 disambiguation 을 위해 별도 dates.
MULTIDAY_HOLIDAYS: List[Dict] = [
    {"name": "설날 연휴", "dates": ["12.30", "1.1", "1.2"], "calendar": "음력"},
    {"name": "추석 연휴", "dates": ["8.14", "8.15", "8.16"], "calendar": "음력"},
    {"name": "신정 연휴", "dates": ["1.1", "1.2", "1.3"], "calendar": "양력"},
]

# 상대일 표현: 오프셋(일) -> 표현 후보. 음수=과거.
OFFSET_EXPRESSIONS: Dict[int, List[str]] = {
    0: ["오늘", "금일"],
    1: ["내일", "익일", "명일", "다음날", "이튿날"],
    2: ["모레", "내일모레", "낼모레"],
    3: ["글피", "삼명일"],
    4: ["그글피"],
    -1: ["어제", "작일"],
    -2: ["그저께", "엊그제"],
    -3: ["그끄저께"],
}

_WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


# --------------------------------------------------------------------------- #
# 헬퍼 (조사 / 달력 변환 / 일진)
# --------------------------------------------------------------------------- #
def _has_no_coda(word: str) -> bool:
    """word 의 마지막 음절에 받침이 없으면 True."""
    if not word:
        return False
    ch = word[-1]
    if not ("가" <= ch <= "힣"):
        return False
    return (ord(ch) - 0xAC00) % 28 == 0


def _lunar_to_solar(y: int, m: int, d: int, is_leap: bool = False) -> Tuple[int, int, int]:
    o = Converter.Lunar2Solar(Lunar(y, m, d, isleap=is_leap))
    return o.year, o.month, o.day


def _add_days_solar(y: int, m: int, d: int, delta: int) -> Tuple[int, int, int]:
    d2 = date(y, m, d) + timedelta(days=delta)
    return d2.year, d2.month, d2.day


def _fmt(y: int, m: int, d: int) -> str:
    return f"{y}.{m}.{d}"


def _ilju(y: int, m: int, d: int) -> str:
    """saju_ko.ilju 재사용(§12.2): 일진 = 간지((JDN+49)%60).
    패키지 import(generation.time_ko)·직접 실행(generation/ on path) 모두 지원."""
    try:
        from generation.saju_ko import ilju
    except ModuleNotFoundError:
        from saju_ko import ilju
    return ilju(y, m, d)


# --------------------------------------------------------------------------- #
# 핵심 생성: 날짜 산술(공휴일+상대일+N)은 공통, 최종 *출력 유형*만 다르게
# --------------------------------------------------------------------------- #
def _build_problem(
    rng: random.Random, *,
    output: str,
    n_range: Tuple[int, int],
    offset_keys: List[int],
    holiday_calendar: Optional[str] = "양력",
    allow_multiday: bool = False,
) -> Dict:
    """한 문제 생성. (question, answer, solution) 반환.

    output: "date"(양력 날짜) | "weekday"(요일) | "ganji"(일진 60갑자).
    """
    year = rng.randint(1990, 2025)
    pool = [h for h in (HOLIDAYS + (MULTIDAY_HOLIDAYS if allow_multiday else []))
            if (not holiday_calendar or h["calendar"] == holiday_calendar)]
    holiday = rng.choice(pool)
    dates = holiday["dates"]
    if len(dates) > 1:
        idx = rng.randrange(len(dates))
        label = {0: "첫날", len(dates) - 1: "마지막날"}.get(idx, "둘째날")
        disp = f"{holiday['name']} {label}"
        hm, hd = map(int, dates[idx].split("."))
    else:
        disp = holiday["name"]
        hm, hd = map(int, dates[0].split("."))

    # 공휴일 -> 양력(solar)로 통일
    if holiday["calendar"] == "양력":
        hs_y, hs_m, hs_d = year, hm, hd
        conv_line = f"  · 양력 공휴일 그대로 {hs_y}.{hs_m}.{hs_d}"
    else:
        hs_y, hs_m, hs_d = _lunar_to_solar(year, hm, hd)
        conv_line = f"  · 음력 공휴일 {year}.{hm}.{hd} → 양력 {hs_y}.{hs_m}.{hs_d}"

    # 상대일 오프셋 + N일 가산
    offset = rng.choice(offset_keys)
    expr = rng.choice(OFFSET_EXPRESSIONS[offset])
    bs_y, bs_m, bs_d = _add_days_solar(hs_y, hs_m, hs_d, offset)
    n = rng.randint(*n_range)
    fy, fm, fd = _add_days_solar(bs_y, bs_m, bs_d, n)

    # 출력 유형별 답
    if output == "date":
        answer = _fmt(fy, fm, fd)
        ask = "그 날의 양력 날짜는 무엇인가? 답은 'YYYY.M.D' 형식으로 써라."
        final = f"  · 답(양력 날짜) = {answer}"
    elif output == "weekday":
        answer = _WEEKDAY_KO[date(fy, fm, fd).weekday()] + "요일"
        ask = "그 날은 무슨 요일인가? (예: 월요일)"
        final = f"  · {fy}.{fm}.{fd} 의 요일 = (JDN mod 7) → {answer}"
    elif output == "ganji":
        answer = _ilju(fy, fm, fd)
        ask = "그 날의 일진(日辰), 즉 그 날에 해당하는 60갑자는 무엇인가? (예: 갑자)"
        final = f"  · 일진 = 간지((JDN({fy}.{fm}.{fd})+49) mod 60) = {answer}"
    else:
        raise ValueError(f"unknown output: {output}")

    particle = "가" if _has_no_coda(expr) else "이"
    verb = "이였어" if offset < 0 else "이야"
    question = (
        f"{year}년 {disp}에 \"{expr}{particle} 내 생일{verb}\" 라는 말을 들었다. "
        f"그 생일로부터 {n}일 후, {ask}"
    )
    solution = "\n".join([
        "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산",
        "[STEP 0] 문제 메타",
        f"  - 날짜 산술 후 '{output}'(을)를 구하는 달력 추론 퍼즐. 답은 [STEP 3]에만 있다.",
        "[STEP 1] 주어진 조건",
        f"  - 연도={year}, 공휴일='{disp}' ({holiday['calendar']} {hm}.{hd})",
        f"  - 상대일='{expr}'(오프셋 {offset:+d}일), 가산 N={n}, 출력유형={output}",
        "[STEP 2] 풀이 전개",
        conv_line,
        f"  · 상대일 {offset:+d}일 → 생일(양력) {bs_y}.{bs_m}.{bs_d}",
        f"  · +{n}일 → {fy}.{fm}.{fd} (양력)",
        final,
        "[STEP 3] 답·검산",
        f"  - 최종 답: {answer}",
    ])
    return {"question": question, "answer": answer, "solution": solution}


# --------------------------------------------------------------------------- #
# 확정 난이도 설정 + 데이터셋 생성
# --------------------------------------------------------------------------- #
# 난이도 = from-scratch 일진(60갑자) 문제의 혼합 비율(saju 식 recipe-mix).
# 정식 evaluator 경로(system prompt 포함) N=100 보정:
#   medium 0.60→51 ✓, hard 0.91→24 ✓. easy: 0.29→87, 0.40→63 → 0.34 채택(≈75 예상).
_COMMON = dict(n_range=(2, 30), offset_keys=[0, 1, 2], holiday_calendar="양력",
               allow_multiday=False)
DIFFICULTY_CONFIGS = {
    "easy":   {"mix": {"ganji": 0.34, "date": 0.66}},
    "medium": {"mix": {"ganji": 0.60, "weekday": 0.40}},
    "hard":   {"mix": {"ganji": 0.91, "date": 0.09}},
}
_SEED_BASE = {"easy": 1_000_000, "medium": 2_000_000, "hard": 3_000_000}


def generate_problem(difficulty: str, problem_id: int = 0, seed: Optional[int] = None) -> Dict:
    """확정 DIFFICULTY_CONFIGS 로 한 문제 생성(표준 스키마: id/question/answer/solution/difficulty).

    각 샘플마다 출력유형을 혼합비대로 추첨(출력유형별 난이도가 bimodal 이라 비율로 정답률 조절).
    """
    rng = random.Random(_SEED_BASE.get(difficulty, 0) + problem_id if seed is None else seed)
    mix = DIFFICULTY_CONFIGS[difficulty]["mix"]
    r, cum, output = rng.random(), 0.0, list(mix)[-1]
    for otype, p in mix.items():
        cum += p
        if r < cum:
            output = otype
            break
    rec = _build_problem(rng, output=output, **_COMMON)
    return {
        "id": f"time_ko_{difficulty}_{problem_id:04d}",
        "question": rec["question"],
        "answer": rec["answer"],
        "solution": rec["solution"],
        "difficulty": difficulty,
    }


def create_dataset_files(num_questions: int, difficulties: Optional[List[str]] = None) -> None:
    """data/time_ko_{tier}.jsonl 작성.

    num_questions = 난이도당 문항 수가 아니라 *총합*(난이도 수로 분할; saju/yacht 관습).
    """
    import json
    tiers = difficulties or ["easy", "medium", "hard"]
    per = num_questions // len(tiers)
    remainder = num_questions % len(tiers)
    combined: List[Dict] = []
    for di, tier in enumerate(tiers):
        count = per + (1 if di < remainder else 0)
        recs = [generate_problem(tier, i) for i in range(count)]
        combined += recs

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("time_ko", combined)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="time_ko 날짜·만세력 추론 퍼즐 생성기")
    ap.add_argument("--num", type=int, default=300, help="총 문항 수(난이도로 3분할)")
    ap.add_argument("--difficulty", nargs="+", choices=["easy", "medium", "hard"], default=None)
    args = ap.parse_args()
    create_dataset_files(num_questions=args.num, difficulties=args.difficulty)

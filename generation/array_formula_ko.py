"""
Array Formula Puzzle Generator (v5 - Quartile Calibration)
Excel 배열 수식 기반 논리 퍼즐 생성기

Problem Types:
1. lookup_query: INDEX-MATCH, VLOOKUP 스타일 데이터 조회
2. conditional_aggregation: SUMIF, COUNTIF 스타일 조건부 집계
3. array_computation: SUMPRODUCT 스타일 배열 연산
4. multi_condition: SUMIFS, MAXIFS 스타일 다중 조건 문제

Difficulty Levels (v12 - Re-calibrated to gemini-3-flash-preview):
- easy: 45% easy + 55% medium template mix → target ~75%
- medium: 50% medium + 50% hard template mix with larger medium tables → target ~50%
- hard: hard templates with distractor columns, 80-85 products, 280-360 orders → target ~25%
"""

import json
import random
import hashlib
import csv
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from pathlib import Path


class ProblemType(Enum):
    LOOKUP_QUERY = "lookup_query"
    CONDITIONAL_AGGREGATION = "conditional_aggregation"
    ARRAY_COMPUTATION = "array_computation"
    MULTI_CONDITION = "multi_condition"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ArrayFormulaConfig:
    """문제 생성 구성"""
    difficulty: str = "medium"
    problem_type: Optional[str] = None
    seed: Optional[int] = None

    min_rows: int = 22
    max_rows: int = 30

    num_categories: int = 6
    num_regions: int = 5

    def __post_init__(self):
        if self.difficulty == "easy":
            self.min_rows, self.max_rows = 24, 32
            self.num_categories = 6
            self.num_regions = 6
        elif self.difficulty == "medium":
            self.min_rows, self.max_rows = 55, 65
            self.num_categories = 8
            self.num_regions = 8
        elif self.difficulty == "hard":
            self.min_rows, self.max_rows = 80, 85
            self.num_categories = 8
            self.num_regions = 8


# ============================================================
# 데이터 생성 유틸리티
# ============================================================

PRODUCT_NAMES = [
    "사과", "배", "포도", "딸기", "바나나", "오렌지", "수박", "멜론", "복숭아", "자두",
    "우유", "치즈", "요구르트", "버터", "아이스크림", "두부", "계란", "햄", "소시지", "베이컨",
    "빵", "쌀", "라면", "파스타", "시리얼", "쿠키", "초콜릿", "사탕", "젤리", "껌",
    "콜라", "사이다", "주스", "커피", "녹차", "생수", "맥주", "소주", "와인", "막걸리",
    "키위", "망고", "체리", "레몬", "라임", "코코넛", "아몬드", "호두", "땅콩", "캐슈너트",
    "참치", "연어", "새우", "게", "오징어", "미역", "양파", "마늘", "고추", "토마토",
    "아보카도", "블루베리", "라즈베리", "석류", "무화과",
    "고등어", "굴", "조개", "멸치", "랍스터",
    "상추", "시금치", "당근", "오이", "버섯",
    "만두", "어묵", "육포", "팝콘", "감자칩",
    "브랜디", "위스키", "밀크셰이크", "스무디", "콤부차",
]

CATEGORIES = ["과일", "유제품", "육류", "곡물", "음료", "채소", "수산물", "가공식품"]
REGIONS = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"]
QUARTERS = ["1분기", "2분기", "3분기", "4분기"]
MEMBERSHIPS = ["골드", "실버", "브론즈", "없음"]
# Answers that read as "no answer" and are ungradeable if returned as the final answer.
# "없음" is a valid membership tier (a non-member) and may appear in tables, but a
# membership-lookup question must never RESOLVE to it.
AMBIGUOUS_ANSWERS = {"없음", ""}

CUSTOMER_NAMES = [
    "김민수", "이영희", "박철수", "최지영", "정동현",
    "강수진", "조현우", "윤미나", "장준호", "임서연",
    "한지훈", "서유나", "손태우", "권수연", "신우진",
    "오하은", "백선우", "홍예린", "유재민", "문다인"
]


def generate_product_table(
    num_rows: int,
    num_categories: int,
    seed: int,
    difficulty: str = "easy"
) -> List[Dict[str, Any]]:
    """상품 테이블 생성"""
    rng = random.Random(seed)

    categories = rng.sample(CATEGORIES, min(num_categories, len(CATEGORIES)))
    products = rng.sample(PRODUCT_NAMES, num_rows)

    # 카테고리 불균등 분포 보장 (avg-of-avg 트랩에 중요)
    table = []
    for i, product in enumerate(products):
        row = {
            "id": i + 1,
            "상품명": product,
            "카테고리": rng.choice(categories),
            "가격": rng.randint(5, 50) * 100,
            "재고": rng.randint(10, 200),
            "할인율": rng.choice([0, 5, 10, 15, 20]),
        }
        if difficulty in ("medium", "hard"):
            row.update({
                "공급사": f"S-{rng.randint(1, 12):02d}",
                "창고": rng.choice(["북부", "남부", "동부", "서부"]),
            })
        if difficulty == "hard":
            row.update({
                "세율": rng.choice([0, 3, 5, 8, 10]),
                "평점": rng.randint(1, 5),
            })
        table.append(row)

    return table


def generate_sales_table(
    product_table: List[Dict],
    num_orders: int,
    num_regions: int,
    seed: int,
    customer_table: Optional[List[Dict]] = None,
    difficulty: str = "easy"
) -> List[Dict[str, Any]]:
    """주문 테이블 생성, customer_id 선택적 포함"""
    rng = random.Random(seed + 1000)

    regions = rng.sample(REGIONS, min(num_regions, len(REGIONS)))
    products = [p["상품명"] for p in product_table]

    table = []
    for i in range(num_orders):
        row = {
            "주문번호": f"ORD-{i+1:03d}",
            "상품명": rng.choice(products),
            "지역": rng.choice(regions),
            "수량": rng.randint(1, 50),
            "분기": rng.choice(QUARTERS),
        }
        if customer_table is not None:
            row["고객번호"] = rng.choice(customer_table)["고객번호"]
        if difficulty in ("medium", "hard"):
            row.update({
                "채널": rng.choice(["온라인", "매장", "제휴", "전화"]),
                "우선순위": rng.choice(["낮음", "보통", "높음"]),
            })
        if difficulty == "hard":
            row.update({
                "배송비": rng.randint(0, 20) * 100,
                "프로모션율": rng.choice([0, 5, 10, 15]),
            })
        table.append(row)

    return table


def generate_customer_table(
    num_customers: int,
    num_regions: int,
    seed: int,
    difficulty: str = "easy"
) -> List[Dict[str, Any]]:
    """고객 테이블 생성"""
    rng = random.Random(seed + 3000)

    regions = rng.sample(REGIONS, min(num_regions, len(REGIONS)))
    names = rng.sample(CUSTOMER_NAMES, min(num_customers, len(CUSTOMER_NAMES)))

    table = []
    for i, name in enumerate(names):
        row = {
            "고객번호": f"CUST-{i+1:03d}",
            "이름": name,
            "등급": rng.choice(MEMBERSHIPS),
            "가입연도": rng.randint(2018, 2024),
            "지역": rng.choice(regions),
        }
        if difficulty in ("medium", "hard"):
            row["세그먼트"] = rng.choice(["개인", "기업", "교육", "공공"])
        if difficulty == "hard":
            row.update({
                "연령대": rng.choice(["20대", "30대", "40대", "50대"]),
                "포인트": rng.randint(0, 5000),
            })
        table.append(row)

    return table


# ============================================================
# 헬퍼 유틸리티
# ============================================================

def _group_sum(items, key_fn, val_fn):
    """항목을 그룹화하고 값을 합산. dict {key: sum} 반환."""
    groups = {}
    for item in items:
        k = key_fn(item)
        groups[k] = groups.get(k, 0) + val_fn(item)
    return groups


def _rank_groups(group_dict, reverse=True):
    """(key, value) 리스트를 값 기준으로 정렬. reverse=True면 내림차순."""
    return sorted(group_dict.items(), key=lambda x: x[1], reverse=reverse)


def _group_count(items, key_fn):
    """그룹별 개수 세기. dict {key: count} 반환."""
    groups = {}
    for item in items:
        k = key_fn(item)
        groups[k] = groups.get(k, 0) + 1
    return groups


def _group_avg(items, key_fn, val_fn):
    """그룹별 평균. dict {key: avg} 반환."""
    sums = {}
    counts = {}
    for item in items:
        k = key_fn(item)
        sums[k] = sums.get(k, 0) + val_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def _median(values):
    """중앙값 계산. 정수로 버림."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return int(s[n // 2])
    return int((s[n // 2 - 1] + s[n // 2]) / 2)


def _group_distinct_count(items, key_fn, val_fn):
    """그룹별 고유값 수 세기. dict {key: distinct_count} 반환."""
    groups = {}
    for item in items:
        k = key_fn(item)
        if k not in groups:
            groups[k] = set()
        groups[k].add(val_fn(item))
    return {k: len(v) for k, v in groups.items()}


def _std_dev(values):
    """모집단 표준편차."""
    if len(values) == 0:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _weighted_choice(rng, templates):
    """가중치가 있는 템플릿에서 선택. 각 템플릿은 (question, answer, weight, solution)."""
    weights = [t[2] for t in templates]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0
    for t in templates:
        cumulative += t[2]
        if r <= cumulative:
            return t[0], t[1], t[3]
    return templates[-1][0], templates[-1][1], templates[-1][3]


def _calibrate_template_weights(difficulty: str, templates):
    """난이도가 올라갈수록 다단계 숫자형 템플릿이 더 자주 뽑히도록 보정."""
    if difficulty == "easy":
        weight_multipliers = {1: 0.75, 2: 1.80}
        text_multiplier = 0.90
        numeric_multiplier = 1.05
    elif difficulty == "medium":
        weight_multipliers = {1: 0.40, 2: 3.60}
        text_multiplier = 0.45
        numeric_multiplier = 1.25
    elif difficulty == "hard":
        weight_multipliers = {1: 0.20, 2: 5.00}
        text_multiplier = 0.30
        numeric_multiplier = 1.40
    else:
        weight_multipliers = {}
        text_multiplier = 1.0
        numeric_multiplier = 1.0

    return [
        (
            question,
            answer,
            weight
            * weight_multipliers.get(weight, 1.0)
            * (numeric_multiplier if isinstance(answer, (int, float)) else text_multiplier),
            solution,
        )
        for question, answer, weight, solution in templates
    ]


def _make_tables_dict(product_table, sales_table, customer_table):
    """반환값용 표준 3테이블 딕셔너리 생성."""
    def columns(base_columns, table):
        extra_columns = []
        if table:
            extra_columns = [col for col in table[0].keys() if col not in base_columns]
        return base_columns + extra_columns

    return {
        "상품": {
            "columns": columns(["id", "상품명", "카테고리", "가격", "재고", "할인율"], product_table),
            "data": product_table
        },
        "주문": {
            "columns": columns(["주문번호", "상품명", "지역", "수량", "분기", "고객번호"], sales_table),
            "data": sales_table
        },
        "고객": {
            "columns": columns(["고객번호", "이름", "등급", "가입연도", "지역"], customer_table),
            "data": customer_table
        }
    }


# ============================================================
# 생성기 공통 데이터 생성
# ============================================================

_ORDER_COUNTS = {"easy": (35, 50), "medium": (120, 170), "hard": (220, 300)}
_CUSTOMER_COUNTS = {"easy": (10, 14), "medium": (18, 20), "hard": (20, 20)}


def _generate_all_tables(config, rng):
    """3개 테이블 + 헬퍼 맵 생성."""
    num_rows = rng.randint(config.min_rows, config.max_rows)
    seed = rng.randint(1, 10000)
    product_table = generate_product_table(num_rows, config.num_categories, seed, config.difficulty)

    min_o, max_o = _ORDER_COUNTS[config.difficulty]
    min_c, max_c = _CUSTOMER_COUNTS[config.difficulty]
    num_orders = rng.randint(min_o, max_o)
    num_customers = rng.randint(min_c, max_c)

    customer_table = generate_customer_table(num_customers, config.num_regions, seed, config.difficulty)
    sales_table = generate_sales_table(
        product_table, num_orders, config.num_regions, seed, customer_table, config.difficulty
    )

    product_map = {p["상품명"]: p for p in product_table}
    customer_map = {c["고객번호"]: c for c in customer_table}

    return product_table, sales_table, customer_table, product_map, customer_map


# ============================================================
# 문제 유형별 생성기
# ============================================================

def generate_lookup_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """LOOKUP 문제 생성 (v3)"""
    pt, st, ct, pm, cm = _generate_all_tables(config, rng)

    categories = list(set(p["카테고리"] for p in pt))
    regions_in_orders = list(set(s["지역"] for s in st))

    if config.difficulty == "easy":
        # T1: 할인가 적용 주문 가치
        target_order = rng.choice(st)
        t1_price = pm[target_order["상품명"]]["가격"]
        t1_disc = pm[target_order["상품명"]]["할인율"]
        t1_disc_price = int(t1_price * (100 - t1_disc) / 100)
        t1_disc_value = t1_disc_price * target_order["수량"]

        # T2: 총 수량 rank-2 → 카테고리
        product_qty = _group_sum(st, lambda s: s["상품명"], lambda s: s["수량"])
        qty_ranked = _rank_groups(product_qty)
        rank2_name = qty_ranked[1][0] if len(qty_ranked) >= 2 else qty_ranked[0][0]
        rank2_cat = pm[rank2_name]["카테고리"]

        # T3: 카테고리 주문 건수
        tgt_cat = rng.choice(categories)
        cat_prods = set(p["상품명"] for p in pt if p["카테고리"] == tgt_cat)
        cat_order_count = len([s for s in st if s["상품명"] in cat_prods])

        # T4: 카테고리 평균 주문 수량 (버림)
        cat_orders = [s for s in st if s["상품명"] in cat_prods]
        cat_avg_qty = int(sum(s["수량"] for s in cat_orders) / len(cat_orders)) if cat_orders else 0

        # T5: 주문 → 고객 → 지역 체인
        chain_order = rng.choice(st)
        chain_cust = cm[chain_order["고객번호"]]

        # T6 (new-hard): 특정 카테고리의 총 할인 매출
        cat_disc_rev = sum(
            int(pm[s["상품명"]]["가격"] * (100 - pm[s["상품명"]]["할인율"]) / 100) * s["수량"]
            for s in cat_orders
        )

        # T7 (new-hard): 주문수량 상위 3개 카테고리 중 최대 재고 상품 → 할인율
        cat_qty = _group_sum(st, lambda s: pm[s["상품명"]]["카테고리"], lambda s: s["수량"])
        cat_qty_ranked = _rank_groups(cat_qty)
        top3_cats = set(c for c, _ in cat_qty_ranked[:3])
        top3_cat_prods = [p for p in pt if p["카테고리"] in top3_cats]
        if top3_cat_prods:
            max_stock_prod = max(top3_cat_prods, key=lambda p: p["재고"])
            t7_disc = max_stock_prod["할인율"]
        else:
            max_stock_prod = pt[0]
            t7_disc = pt[0]["할인율"]

        # T8 (new-hard): 주문 건수 최다 지역 → 총 할인 매출
        reg_count = _group_count(st, lambda s: s["지역"])
        reg_count_ranked = _rank_groups(reg_count)
        top_count_reg = reg_count_ranked[0][0] if reg_count_ranked else regions_in_orders[0]
        top_reg_orders = [s for s in st if s["지역"] == top_count_reg]
        top_reg_disc_rev = sum(
            int(pm[s["상품명"]]["가격"] * (100 - pm[s["상품명"]]["할인율"]) / 100) * s["수량"]
            for s in top_reg_orders
        )

        question_templates = [
            (f"'{target_order['주문번호']}' 주문의 할인 적용 가치는 얼마입니까? (할인가 = 가격×(100-할인율)/100 소수점 버림, 가치 = 할인가 × 수량)",
             t1_disc_value, 1,
             f"1단계: '{target_order['주문번호']}' → 상품명 = '{target_order['상품명']}'\n"
             f"2단계: 가격 = {t1_price}, 할인율 = {t1_disc}%\n"
             f"3단계: 할인가 = {t1_price}×(100-{t1_disc})/100 = {t1_disc_price}\n"
             f"4단계: 가치 = {t1_disc_price} × {target_order['수량']} = {t1_disc_value}\n"
             f"최종 답: {t1_disc_value}"),

            (f"총 주문수량 2위 상품의 카테고리는 무엇입니까?",
             rank2_cat, 1,
             f"1단계: 상품별 수량 합산\n"
             f"2단계: {', '.join(f'{k}({v})' for k,v in qty_ranked[:5])}\n"
             f"3단계: 2위 = '{rank2_name}'\n"
             f"4단계: 카테고리 = '{rank2_cat}'\n최종 답: {rank2_cat}"),

            (f"'{tgt_cat}' 카테고리 상품의 주문 건수는 몇 건입니까?",
             cat_order_count, 1,
             f"1단계: '{tgt_cat}' 상품: {', '.join(list(cat_prods)[:5])}\n"
             f"2단계: 주문 테이블에서 필터링\n"
             f"3단계: 건수 = {cat_order_count}\n최종 답: {cat_order_count}"),

            (f"'{tgt_cat}' 카테고리 상품 주문의 평균 수량은 얼마입니까? (소수점 버림)",
             cat_avg_qty, 1,
             f"1단계: '{tgt_cat}' 상품: {', '.join(list(cat_prods)[:5])}\n"
             f"2단계: 주문 필터링: {len(cat_orders)}건\n"
             f"3단계: 평균 수량 = {cat_avg_qty}\n최종 답: {cat_avg_qty}"),

            (f"'{chain_order['주문번호']}'를 주문한 고객의 지역은 어디입니까?",
             chain_cust["지역"], 1,
             f"1단계: '{chain_order['주문번호']}' → 고객번호 = '{chain_order['고객번호']}'\n"
             f"2단계: 고객 테이블에서 조회\n"
             f"3단계: 지역 = '{chain_cust['지역']}'\n"
             f"최종 답: {chain_cust['지역']}"),

            (f"'{tgt_cat}' 카테고리 상품의 총 할인 매출액은 얼마입니까? (각 주문: 할인가 = 가격×(100-할인율)/100 소수점 버림, 그 다음 수량을 곱해 모두 합산)",
             cat_disc_rev, 2,
             f"1단계: '{tgt_cat}' 상품: {', '.join(list(cat_prods)[:5])}\n"
             f"2단계: 주문 필터링: {len(cat_orders)}건\n"
             f"3단계: 각 주문마다 할인가(버림) × 수량\n"
             f"4단계: 합계 = {cat_disc_rev}\n최종 답: {cat_disc_rev}"),

            (f"총 주문수량 상위 3개 카테고리 안에서 재고가 가장 많은 상품의 할인율은 몇 %입니까?",
             t7_disc, 2,
             f"1단계: 카테고리별 주문수량: {', '.join(f'{k}({v})' for k,v in cat_qty_ranked[:5])}\n"
             f"2단계: 상위 3개 카테고리: {', '.join(top3_cats)}\n"
             f"3단계: 해당 카테고리 상품 수: {len(top3_cat_prods)}개\n"
             f"4단계: 최대 재고 상품 = '{max_stock_prod['상품명']}' (재고={max_stock_prod['재고']})\n"
             f"5단계: 할인율 = {t7_disc}%\n최종 답: {t7_disc}"),

            (f"주문 건수가 가장 많은 지역은 어디입니까? 그 지역의 총 할인 매출액은 얼마입니까? (할인가 = 가격×(100-할인율)/100, 항목별 버림 후 × 수량)",
             top_reg_disc_rev, 2,
             f"1단계: 지역별 주문 건수: {', '.join(f'{k}({v})' for k,v in reg_count_ranked[:5])}\n"
             f"2단계: 최다 지역 = '{top_count_reg}'\n"
             f"3단계: '{top_count_reg}' 주문: {len(top_reg_orders)}건\n"
             f"4단계: 할인 매출 = {top_reg_disc_rev}\n최종 답: {top_reg_disc_rev}"),
        ]

    elif config.difficulty == "medium":
        # T1: 매출 rank-2 상품의 카테고리
        prod_revenue = _group_sum(st, lambda s: s["상품명"],
                                  lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        rev_ranked = _rank_groups(prod_revenue)
        rank2_name = rev_ranked[1][0] if len(rev_ranked) >= 2 else rev_ranked[0][0]
        rank2_cat = pm[rank2_name]["카테고리"]

        # T2: 특정 지역 최다 소비 고객 (3테이블 조인)
        tgt_region = rng.choice(regions_in_orders)
        reg_orders = [s for s in st if s["지역"] == tgt_region]
        reg_cust_spend = _group_sum(reg_orders, lambda s: s["고객번호"],
                                     lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        reg_spend_ranked = _rank_groups(reg_cust_spend)
        reg_top_cid = reg_spend_ranked[0][0] if reg_spend_ranked else ct[0]["고객번호"]
        reg_top_cust_name = cm[reg_top_cid]["이름"]

        # T3: 특정 등급 고객 최다 수량 주문 상품 → 할인율
        tgt_grade = rng.choice(["골드", "실버", "브론즈"])
        grade_cids = set(c["고객번호"] for c in ct if c["등급"] == tgt_grade)
        grade_orders = [s for s in st if s["고객번호"] in grade_cids]
        grade_prod_qty = _group_sum(grade_orders, lambda s: s["상품명"], lambda s: s["수량"])
        grade_qty_ranked = _rank_groups(grade_prod_qty)
        grade_top_prod = grade_qty_ranked[0][0] if grade_qty_ranked else pt[0]["상품명"]
        grade_top_disc = pm[grade_top_prod]["할인율"]

        # T4: 분기 최다 매출 상품 → 해당 고객 중 최초 가입연도
        tgt_q = rng.choice(QUARTERS)
        q_orders = [s for s in st if s["분기"] == tgt_q]
        q_prod_rev = _group_sum(q_orders, lambda s: s["상품명"],
                                lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        q_rev_ranked = _rank_groups(q_prod_rev)
        q_top_prod = q_rev_ranked[0][0] if q_rev_ranked else pt[0]["상품명"]
        q_top_orders = [s for s in q_orders if s["상품명"] == q_top_prod]
        q_top_cids = set(s["고객번호"] for s in q_top_orders)
        earliest_year = min((cm[cid]["가입연도"] for cid in q_top_cids), default=2020)

        # T5 (new): 주문된 고유 상품 수 2위 카테고리 → 총 매출
        cat_distinct = _group_distinct_count(st, lambda s: pm[s["상품명"]]["카테고리"], lambda s: s["상품명"])
        cat_dist_ranked = _rank_groups(cat_distinct)
        t5_cat = cat_dist_ranked[1][0] if len(cat_dist_ranked) >= 2 else cat_dist_ranked[0][0]
        t5_cat_prods = set(p["상품명"] for p in pt if p["카테고리"] == t5_cat)
        t5_cat_orders = [s for s in st if s["상품명"] in t5_cat_prods]
        t5_cat_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in t5_cat_orders)

        # T6 (new): 평균 수량 초과 주문의 매출
        all_qtys = [s["수량"] for s in st]
        avg_qty = sum(all_qtys) / len(all_qtys) if all_qtys else 0
        above_avg_orders = [s for s in st if s["수량"] > avg_qty]
        above_avg_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in above_avg_orders)

        question_templates = [
            (f"매출 2위 상품의 카테고리는 무엇입니까? (매출 = 가격 × 수량)",
             rank2_cat, 1,
             f"1단계: 주문별 매출 계산\n"
             f"2단계: 상품별 매출 합산\n"
             f"3단계: {', '.join(f'{k}({v})' for k,v in rev_ranked[:5])}\n"
             f"4단계: 2위 = '{rank2_name}' → 카테고리 = '{rank2_cat}'\n최종 답: {rank2_cat}"),

            (f"'{tgt_region}'에서 가장 많이 소비한 고객의 이름은 무엇입니까? (소비 = 가격 × 수량)",
             reg_top_cust_name, 1,
             f"1단계: '{tgt_region}' 주문 필터링: {len(reg_orders)}건\n"
             f"2단계: 고객별 소비 합산\n"
             f"3단계: {', '.join(f'{k}({v})' for k,v in reg_spend_ranked[:5])}\n"
             f"4단계: 1위 = '{reg_top_cid}' → '{reg_top_cust_name}'\n최종 답: {reg_top_cust_name}"),

            (f"'{tgt_grade}' 등급 고객이 가장 많이 주문한 상품(수량 기준)의 할인율은 몇 %입니까?",
             grade_top_disc, 1,
             f"1단계: '{tgt_grade}' 고객: {len(grade_cids)}명\n"
             f"2단계: 해당 주문 필터링: {len(grade_orders)}건\n"
             f"3단계: 상품별 수량: {', '.join(f'{k}({v})' for k,v in grade_qty_ranked[:5])}\n"
             f"4단계: 1위 = '{grade_top_prod}' → 할인율 = {grade_top_disc}%\n최종 답: {grade_top_disc}"),

            (f"{tgt_q} 매출 1위 상품을 주문한 고객 중 가장 오래된 가입연도는 몇 년입니까?",
             earliest_year, 1,
             f"1단계: {tgt_q} 주문 필터링: {len(q_orders)}건\n"
             f"2단계: 상품별 매출: {', '.join(f'{k}({v})' for k,v in q_rev_ranked[:3])}\n"
             f"3단계: 1위 = '{q_top_prod}'\n"
             f"4단계: 해당 주문 고객: {', '.join(q_top_cids)}\n"
             f"5단계: 가장 오래된 가입연도 = {earliest_year}\n최종 답: {earliest_year}"),

            (f"주문된 고유 상품 수가 두 번째로 많은 카테고리는 무엇입니까? 그 카테고리의 총 매출액은 얼마입니까? (매출 = 가격 × 수량)",
             t5_cat_rev, 2,
             f"1단계: 카테고리별 주문된 고유 상품 수: {', '.join(f'{k}({v})' for k,v in cat_dist_ranked[:5])}\n"
             f"2단계: 2위 = '{t5_cat}'\n"
             f"3단계: '{t5_cat}' 상품: {', '.join(list(t5_cat_prods)[:5])}\n"
             f"4단계: 매출 = {t5_cat_rev}\n최종 답: {t5_cat_rev}"),

            (f"평균 주문 수량({int(avg_qty)})을 초과하는 주문의 총 매출액은 얼마입니까? (매출 = 가격 × 수량)",
             above_avg_rev, 2,
             f"1단계: 평균 수량 = {int(avg_qty)}\n"
             f"2단계: 수량 > 평균인 주문: {len(above_avg_orders)}건\n"
             f"3단계: 매출 합계 = {above_avg_rev}\n최종 답: {above_avg_rev}"),
        ]

    else:  # hard
        # T1 (Pattern A): 평균 초과 소비 고객 → 고유 카테고리 수 rank-2 → 지역
        cust_spend = _group_sum(st, lambda s: s["고객번호"],
                                lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        avg_cust_spend = sum(cust_spend.values()) / len(cust_spend) if cust_spend else 0
        above_avg_cids = {cid for cid, v in cust_spend.items() if v > avg_cust_spend}
        above_avg_distinct = _group_distinct_count(
            [s for s in st if s["고객번호"] in above_avg_cids],
            lambda s: s["고객번호"],
            lambda s: pm[s["상품명"]]["카테고리"]
        )
        above_avg_dist_ranked = _rank_groups(above_avg_distinct)
        t1_cid = above_avg_dist_ranked[1][0] if len(above_avg_dist_ranked) >= 2 else above_avg_dist_ranked[0][0]
        t1_region = cm[t1_cid]["지역"]

        # T2: 3건 이상 주문 고객 중 평균 주문 가치 rank-2 → 등급
        cust_order_counts = _group_count(st, lambda s: s["고객번호"])
        cust_3plus = {cid for cid, cnt in cust_order_counts.items() if cnt >= 3}
        if len(cust_3plus) < 2:
            cust_3plus = {cid for cid, cnt in cust_order_counts.items() if cnt >= 2}
        if len(cust_3plus) < 2:
            cust_3plus = set(cid for cid, _ in sorted(cust_order_counts.items(), key=lambda x: x[1], reverse=True)[:2])
        cust_avg_val = {}
        for cid in cust_3plus:
            cid_orders = [s for s in st if s["고객번호"] == cid]
            total_val = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in cid_orders)
            cust_avg_val[cid] = total_val / len(cid_orders) if cid_orders else 0
        avg_val_ranked = _rank_groups(cust_avg_val)
        t2_cid = avg_val_ranked[1][0] if len(avg_val_ranked) >= 2 else avg_val_ranked[0][0]
        t2_grade = cm[t2_cid]["등급"]

        # T3 (new): 매출 하위 3개 카테고리 → 평균 수량 최대 상품 → 최다 주문자 → 등급
        cat_rev = _group_sum(st, lambda s: pm[s["상품명"]]["카테고리"],
                             lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        cat_rev_ranked = _rank_groups(cat_rev)
        bottom3_cats = set(c for c, _ in cat_rev_ranked[-3:]) if len(cat_rev_ranked) >= 3 else set(c for c, _ in cat_rev_ranked)
        bottom3_prod_avg_qty = {}
        for p_name in set(p["상품명"] for p in pt if p["카테고리"] in bottom3_cats):
            p_orders = [s for s in st if s["상품명"] == p_name]
            if p_orders:
                bottom3_prod_avg_qty[p_name] = sum(s["수량"] for s in p_orders) / len(p_orders)
        if bottom3_prod_avg_qty:
            t3_prod = max(bottom3_prod_avg_qty, key=bottom3_prod_avg_qty.get)
        else:
            t3_prod = pt[0]["상품명"]
        t3_orders = [s for s in st if s["상품명"] == t3_prod]
        t3_cust_qty = _group_sum(t3_orders, lambda s: s["고객번호"], lambda s: s["수량"])
        t3_cust_ranked = _rank_groups(t3_cust_qty)
        t3_top_cid = t3_cust_ranked[0][0] if t3_cust_ranked else ct[0]["고객번호"]
        t3_membership = cm[t3_top_cid]["등급"]

        # T4 (new): join<2021 → 주문당 평균 소비 1위 → 최다 주문 카테고리 → 해당 카테고리 평균 재고
        old_cids = set(c["고객번호"] for c in ct if c["가입연도"] < 2021)
        old_orders = [s for s in st if s["고객번호"] in old_cids]
        old_cust_avg = {}
        old_cust_counts = _group_count(old_orders, lambda s: s["고객번호"])
        old_cust_rev = _group_sum(old_orders, lambda s: s["고객번호"],
                                  lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        for cid in old_cust_rev:
            old_cust_avg[cid] = old_cust_rev[cid] / old_cust_counts.get(cid, 1)
        old_avg_ranked = _rank_groups(old_cust_avg)
        t4_cid = old_avg_ranked[0][0] if old_avg_ranked else ct[0]["고객번호"]
        t4_orders = [s for s in st if s["고객번호"] == t4_cid]
        t4_cat_qty = _group_sum(t4_orders, lambda s: pm[s["상품명"]]["카테고리"], lambda s: s["수량"])
        t4_cat_ranked = _rank_groups(t4_cat_qty)
        t4_top_cat = t4_cat_ranked[0][0] if t4_cat_ranked else categories[0]
        t4_cat_prods = [p for p in pt if p["카테고리"] == t4_top_cat]
        t4_avg_stock = int(sum(p["재고"] for p in t4_cat_prods) / len(t4_cat_prods)) if t4_cat_prods else 0

        # T5: 2021 이후 가입 고객 → 최다 소비 → 최다 주문 카테고리
        recent_cids = set(c["고객번호"] for c in ct if c["가입연도"] >= 2021)
        recent_orders = [s for s in st if s["고객번호"] in recent_cids]
        recent_spend = _group_sum(recent_orders, lambda s: s["고객번호"],
                                   lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        recent_ranked = _rank_groups(recent_spend)
        t5_cid = recent_ranked[0][0] if recent_ranked else ct[0]["고객번호"]
        t5_orders = [s for s in st if s["고객번호"] == t5_cid]
        t5_cat_qty = _group_sum(t5_orders, lambda s: pm[s["상품명"]]["카테고리"], lambda s: s["수량"])
        t5_cat_ranked = _rank_groups(t5_cat_qty)
        t5_top_cat = t5_cat_ranked[0][0] if t5_cat_ranked else categories[0]

        question_templates = [
            (f"평균 소비액 초과 고객 중 고유 카테고리 수 기준 2위 고객의 지역은 어디입니까? (소비 = 가격×수량)",
             t1_region, 1,
             f"1단계: 고객별 소비액 계산\n"
             f"2단계: 평균 = {int(avg_cust_spend)}\n"
             f"3단계: 평균 초과 고객: {len(above_avg_cids)}명\n"
             f"4단계: 평균 초과 고객별 고유 카테고리 수: {', '.join(f'{k}({v})' for k,v in above_avg_dist_ranked[:5])}\n"
             f"5단계: 2위 = '{t1_cid}' → 지역 = '{t1_region}'\n최종 답: {t1_region}"),

            (f"3건 이상 주문한 고객 중 평균 주문 가치(가격×수량/주문수) 2위 고객의 등급은 무엇입니까?",
             t2_grade, 1,
             f"1단계: 고객별 주문 수 집계\n"
             f"2단계: 3건 이상: {', '.join(cust_3plus)}\n"
             f"3-4단계: 고객별 평균 주문 가치\n"
             f"5단계: {', '.join(f'{k}({int(v)})' for k,v in avg_val_ranked[:5])}\n"
             f"6단계: 2위 = '{t2_cid}' → 등급 = '{t2_grade}'\n최종 답: {t2_grade}"),

            (f"총 매출 하위 3개 카테고리 중 평균 주문 수량이 가장 높은 상품을 찾으세요. 그 상품을 수량 기준으로 가장 많이 주문한 고객의 등급은 무엇입니까?",
             t3_membership, 1,
             f"1단계: 카테고리별 매출: {', '.join(f'{k}({v})' for k,v in cat_rev_ranked)}\n"
             f"2단계: 하위 3개: {', '.join(bottom3_cats)}\n"
             f"3단계: 하위 3개 카테고리 상품별 평균 주문 수량 계산\n"
             f"4단계: 평균 수량 최대 상품 = '{t3_prod}'\n"
             f"5단계: 최다 주문 고객 = '{t3_top_cid}' → 등급 = '{t3_membership}'\n최종 답: {t3_membership}"),

            (f"2021년 이전 가입 고객 중 주문당 평균 매출이 가장 높은 고객이 가장 많이 주문한 카테고리의 평균 재고는 얼마입니까? (소수점 버림)",
             t4_avg_stock, 1,
             f"1단계: 2021 이전 가입 고객: {len(old_cids)}명\n"
             f"2단계: 주문당 평균 매출: {', '.join(f'{k}({int(v)})' for k,v in old_avg_ranked[:5])}\n"
             f"3단계: 1위 = '{t4_cid}'\n"
             f"4단계: 최다 주문 카테고리 = '{t4_top_cat}'\n"
             f"5단계: '{t4_top_cat}' 상품: {len(t4_cat_prods)}개, 평균 재고 = {t4_avg_stock}\n최종 답: {t4_avg_stock}"),

            (f"2021년 이후 가입 고객 중 소비액 1위 고객이 수량 기준으로 가장 많이 주문한 카테고리는 무엇입니까? (카테고리명으로 답하세요)",
             t5_top_cat, 1,
             f"1단계: 2021 이후 가입 고객: {len(recent_cids)}명\n"
             f"2단계: 소비액: {', '.join(f'{k}({int(v)})' for k,v in recent_ranked[:5])}\n"
             f"3단계: 1위 = '{t5_cid}'\n"
             f"4단계: '{t5_cid}' 카테고리별 수량: {', '.join(f'{k}({v})' for k,v in t5_cat_ranked)}\n"
             f"5단계: 최다 카테고리 = '{t5_top_cat}'\n최종 답: {t5_top_cat}"),
        ]

    question, answer, solution = _weighted_choice(
        rng, _calibrate_template_weights(config.difficulty, question_templates)
    )

    return {
        "type": ProblemType.LOOKUP_QUERY.value,
        "difficulty": config.difficulty,
        "tables": _make_tables_dict(pt, st, ct),
        "question": question,
        "answer": answer,
        "answer_type": "number" if isinstance(answer, (int, float)) else "text",
        "solution": solution,
    }


def generate_conditional_aggregation_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """조건부 집계 문제 생성 (v3)"""
    pt, st, ct, pm, cm = _generate_all_tables(config, rng)

    categories = list(set(p["카테고리"] for p in pt))
    regions_in_orders = list(set(s["지역"] for s in st))
    tgt_cat = rng.choice(categories)
    cat_products = [p for p in pt if p["카테고리"] == tgt_cat]
    cat_prod_names = set(p["상품명"] for p in cat_products)

    if config.difficulty == "easy":
        # T1: 카테고리 매출
        cat_orders = [s for s in st if s["상품명"] in cat_prod_names]
        cat_revenue = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in cat_orders)

        # T2: 할인 재고 가치
        disc_inv = sum(int(p["가격"] * (100 - p["할인율"]) / 100) * p["재고"] for p in cat_products)

        # T3: 카테고리 평균 가격
        avg_price = int(sum(p["가격"] for p in cat_products) / len(cat_products)) if cat_products else 0

        # T4: 분기 총 수량
        tgt_q = rng.choice(QUARTERS)
        q_orders = [s for s in st if s["분기"] == tgt_q]
        q_total_qty = sum(s["수량"] for s in q_orders)

        # T5: 할인율 >= 10% 주문 건수
        disc_prods = set(p["상품명"] for p in pt if p["할인율"] >= 10)
        disc_order_count = len([s for s in st if s["상품명"] in disc_prods])

        # T6 (new): 수량이 중앙값을 초과하는 주문의 매출
        all_qtys = sorted(s["수량"] for s in st)
        med_qty = _median(all_qtys)
        above_med_orders = [s for s in st if s["수량"] > med_qty]
        above_med_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in above_med_orders)

        # T7 (new): 평균 가격 초과 상품 → 할인 재고 가치
        all_avg_price = int(sum(p["가격"] for p in pt) / len(pt)) if pt else 0
        above_avg_prods = [p for p in pt if p["가격"] > all_avg_price]
        above_avg_disc_inv = sum(int(p["가격"] * (100 - p["할인율"]) / 100) * p["재고"] for p in above_avg_prods)

        # T8 (new): 평균 할인율이 가장 높은 카테고리 → 주문 건수
        cat_avg_disc = _group_avg(pt, lambda p: p["카테고리"], lambda p: p["할인율"])
        cat_disc_ranked = _rank_groups(cat_avg_disc)
        top_disc_cat = cat_disc_ranked[0][0] if cat_disc_ranked else categories[0]
        top_disc_cat_prods = set(p["상품명"] for p in pt if p["카테고리"] == top_disc_cat)
        top_disc_cat_orders = len([s for s in st if s["상품명"] in top_disc_cat_prods])

        question_templates = [
            (f"'{tgt_cat}' 카테고리 상품의 총 매출액은 얼마입니까? (매출 = 가격 × 수량, 상품 테이블에서 가격 조회)",
             cat_revenue, 1,
             f"1단계: '{tgt_cat}' 상품: {', '.join(list(cat_prod_names)[:5])}\n"
             f"2단계: 주문 필터링: {len(cat_orders)}건\n"
             f"3단계: 가격×수량 합계 = {cat_revenue}\n최종 답: {cat_revenue}"),

            (f"'{tgt_cat}' 카테고리의 총 할인 재고 가치는 얼마입니까? (할인가 = 가격 × (100-할인율)/100 소수점 버림, 할인가 × 재고의 합)",
             disc_inv, 1,
             f"1단계: '{tgt_cat}' 상품 {len(cat_products)}개\n"
             f"2단계: 각 상품의 할인가 = 가격*(100-할인율)/100 (버림)\n"
             f"3단계: 할인가 × 재고\n"
             f"4단계: 합계 = {disc_inv}\n최종 답: {disc_inv}"),

            (f"'{tgt_cat}' 카테고리의 평균 가격은 얼마입니까? (소수점 버림)",
             avg_price, 1,
             f"1단계: '{tgt_cat}' 상품: {len(cat_products)}개\n"
             f"2단계: 가격 합계 / 개수 = {avg_price}\n최종 답: {avg_price}"),

            (f"{tgt_q}에 주문된 전체 수량의 합계는 얼마입니까?",
             q_total_qty, 1,
             f"1단계: {tgt_q} 주문 필터링: {len(q_orders)}건\n"
             f"2단계: 수량 합계 = {q_total_qty}\n최종 답: {q_total_qty}"),

            (f"할인율 10% 이상인 상품의 총 주문 건수는 몇 건입니까?",
             disc_order_count, 1,
             f"1단계: 할인율 ≥ 10% 상품: {len(disc_prods)}개\n"
             f"2단계: 해당 주문 필터링\n"
             f"3단계: 건수 = {disc_order_count}\n최종 답: {disc_order_count}"),

            (f"주문 수량이 중앙값({med_qty})을 초과하는 주문의 총 매출액은 얼마입니까? (매출 = 가격 × 수량)",
             above_med_rev, 2,
             f"1단계: 주문 수량 중앙값 = {med_qty}\n"
             f"2단계: 수량 > {med_qty} 주문: {len(above_med_orders)}건\n"
             f"3단계: 매출 합계 = {above_med_rev}\n최종 답: {above_med_rev}"),

            (f"전체 평균 가격({all_avg_price})보다 가격이 높은 상품들의 총 할인 재고 가치는 얼마입니까? (할인가 = 가격×(100-할인율)/100 소수점 버림, 그 다음 재고를 곱해 합산)",
             above_avg_disc_inv, 2,
             f"1단계: 평균 가격 = {all_avg_price}\n"
             f"2단계: 평균 초과 상품: {len(above_avg_prods)}개\n"
             f"3단계: 각 상품마다 할인가(버림) × 재고\n"
             f"4단계: 합계 = {above_avg_disc_inv}\n최종 답: {above_avg_disc_inv}"),

            (f"평균 할인율이 가장 높은 카테고리는 무엇입니까? 그 카테고리 상품의 주문 건수는 몇 건입니까?",
             top_disc_cat_orders, 2,
             f"1단계: 카테고리별 평균 할인율: {', '.join(f'{k}({v:.1f})' for k,v in cat_disc_ranked[:5])}\n"
             f"2단계: 최고 = '{top_disc_cat}'\n"
             f"3단계: '{top_disc_cat}' 상품: {len(top_disc_cat_prods)}개\n"
             f"4단계: 주문 건수 = {top_disc_cat_orders}\n최종 답: {top_disc_cat_orders}"),
        ]

    elif config.difficulty == "medium":
        # T1: 카테고리 매출 전체 대비 %
        cat_rev = _group_sum(st, lambda s: pm[s["상품명"]]["카테고리"],
                             lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        total_rev = sum(cat_rev.values())
        cat_rev_ranked = _rank_groups(cat_rev)
        tgt_cat_rev = cat_rev.get(tgt_cat, 0)
        tgt_cat_pct = int(tgt_cat_rev * 100 / total_rev) if total_rev > 0 else 0

        # T2: 평균 소비 초과 고객 수
        cust_spend = _group_sum(st, lambda s: s["고객번호"],
                                lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        avg_spend = sum(cust_spend.values()) / len(cust_spend) if cust_spend else 0
        custs_above = sum(1 for v in cust_spend.values() if v > avg_spend)

        # T3: 등급별 매출 %
        tgt_grade = rng.choice(["골드", "실버", "브론즈"])
        grade_cids = set(c["고객번호"] for c in ct if c["등급"] == tgt_grade)
        grade_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in st if s["고객번호"] in grade_cids)
        grade_pct = int(grade_rev * 100 / total_rev) if total_rev > 0 else 0

        # T4: 재고 중앙값 초과 상품 매출 %
        stocks = [p["재고"] for p in pt]
        median_stock = _median(stocks)
        high_stock_prods = set(p["상품명"] for p in pt if p["재고"] > median_stock)
        high_stock_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in st if s["상품명"] in high_stock_prods)
        high_stock_pct = int(high_stock_rev * 100 / total_rev) if total_rev > 0 else 0

        # T5 (new): 주문 최다 분기 → 골드 고객 매출 %
        q_counts = _group_count(st, lambda s: s["분기"])
        q_count_ranked = _rank_groups(q_counts)
        top_q = q_count_ranked[0][0] if q_count_ranked else "1분기"
        top_q_orders = [s for s in st if s["분기"] == top_q]
        top_q_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in top_q_orders)
        gold_cids = set(c["고객번호"] for c in ct if c["등급"] == "골드")
        top_q_gold_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in top_q_orders if s["고객번호"] in gold_cids)
        top_q_gold_pct = int(top_q_gold_rev * 100 / top_q_rev) if top_q_rev > 0 else 0

        # T6 (new): 수량 가중 평균 가격 vs 단순 평균 가격 차이
        total_qty = sum(s["수량"] for s in st)
        qty_weighted_price = int(sum(pm[s["상품명"]]["가격"] * s["수량"] for s in st) / total_qty) if total_qty > 0 else 0
        simple_avg_price = int(sum(p["가격"] for p in pt) / len(pt)) if pt else 0
        price_diff = abs(qty_weighted_price - simple_avg_price)

        question_templates = [
            (f"'{tgt_cat}' 카테고리의 매출이 전체 매출의 몇 %를 차지합니까? (소수점 버림, 매출 = 가격 × 수량)",
             tgt_cat_pct, 1,
             f"1단계: 카테고리별 매출: {', '.join(f'{k}({v})' for k,v in cat_rev_ranked)}\n"
             f"2단계: 전체 매출 = {total_rev}\n"
             f"3단계: '{tgt_cat}' 매출 = {tgt_cat_rev}\n"
             f"4단계: 비율 = {tgt_cat_pct}%\n최종 답: {tgt_cat_pct}"),

            (f"고객별 평균 소비액보다 많이 소비한 고객은 몇 명입니까? (소비 = 가격 × 수량의 합)",
             custs_above, 1,
             f"1단계: 고객별 소비액 계산 ({len(cust_spend)}명)\n"
             f"2단계: 평균 = {int(avg_spend)}\n"
             f"3단계: 평균 초과 수: {custs_above}\n최종 답: {custs_above}"),

            (f"'{tgt_grade}' 등급 고객의 매출이 전체 매출의 몇 %입니까? (소수점 버림)",
             grade_pct, 1,
             f"1단계: '{tgt_grade}' 고객: {len(grade_cids)}명\n"
             f"2단계: '{tgt_grade}' 매출 = {grade_rev}\n"
             f"3단계: 전체 매출 = {total_rev}\n"
             f"4단계: 비율 = {grade_pct}%\n최종 답: {grade_pct}"),

            (f"재고 중앙값({median_stock})보다 재고가 많은 상품의 매출이 전체의 몇 %입니까? (소수점 버림)",
             high_stock_pct, 1,
             f"1단계: 재고 중앙값 = {median_stock}\n"
             f"2단계: 재고 > {median_stock} 상품: {len(high_stock_prods)}개\n"
             f"3단계: 해당 매출 = {high_stock_rev}\n"
             f"4단계: 전체 = {total_rev}, 비율 = {high_stock_pct}%\n최종 답: {high_stock_pct}"),

            (f"주문 건수가 가장 많은 분기({top_q})에서, 해당 분기 매출 중 골드 등급 고객 매출은 몇 %입니까? (소수점 버림)",
             top_q_gold_pct, 2,
             f"1단계: 분기별 주문 건수: {', '.join(f'{k}({v})' for k,v in q_count_ranked)}\n"
             f"2단계: 최다 주문 분기 = {top_q}\n"
             f"3단계: {top_q} 매출 = {top_q_rev}, {top_q} 골드 매출 = {top_q_gold_rev}\n"
             f"4단계: 비율 = {top_q_gold_pct}%\n최종 답: {top_q_gold_pct}"),

            (f"수량 가중 평균 가격(sum(가격×수량)/sum(수량))과 상품 단순 평균 가격의 차이는 얼마입니까? (절대값, 소수점 버림)",
             price_diff, 2,
             f"1단계: 수량 가중 평균 가격 = {qty_weighted_price}\n"
             f"2단계: 단순 평균 가격 = {simple_avg_price}\n"
             f"3단계: 차이 = {price_diff}\n최종 답: {price_diff}"),
        ]

    else:  # hard
        # T1: avg-of-avg 트랩 (카테고리별 평균 가격의 평균)
        cat_avg = {}
        for cat in categories:
            prods = [p for p in pt if p["카테고리"] == cat]
            if prods:
                cat_avg[cat] = sum(p["가격"] for p in prods) / len(prods)
        avg_of_avgs = int(sum(cat_avg.values()) / len(cat_avg)) if cat_avg else 0

        # T2 (Pattern C): 골드 전용 avg-of-avg vs 전체 avg-of-avg 차이
        gold_cids = set(c["고객번호"] for c in ct if c["등급"] == "골드")
        gold_orders = [s for s in st if s["고객번호"] in gold_cids]
        gold_cat_avg = {}
        for cat in categories:
            cat_prods_set = set(p["상품명"] for p in pt if p["카테고리"] == cat)
            cat_gold = [s for s in gold_orders if s["상품명"] in cat_prods_set]
            if cat_gold:
                gold_cat_avg[cat] = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in cat_gold) / len(cat_gold)
        gold_avg_of_avg = int(sum(gold_cat_avg.values()) / len(gold_cat_avg)) if gold_cat_avg else 0
        all_cat_avg_rev = {}
        for cat in categories:
            cat_prods_set = set(p["상품명"] for p in pt if p["카테고리"] == cat)
            cat_all = [s for s in st if s["상품명"] in cat_prods_set]
            if cat_all:
                all_cat_avg_rev[cat] = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in cat_all) / len(cat_all)
        overall_avg_of_avg = int(sum(all_cat_avg_rev.values()) / len(all_cat_avg_rev)) if all_cat_avg_rev else 0
        avg_of_avg_diff = abs(gold_avg_of_avg - overall_avg_of_avg)

        # T3: 분기별 할인 매출 비율 → 최대 분기 vs 전체 비율
        total_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in st)
        total_disc_rev = int(sum(
            pm[s["상품명"]]["가격"] * (100 - pm[s["상품명"]]["할인율"]) / 100 * s["수량"]
            for s in st
        ))
        overall_disc_pct = int(total_disc_rev * 100 / total_rev) if total_rev > 0 else 0
        q_disc_pcts = {}
        for q in QUARTERS:
            q_orders = [s for s in st if s["분기"] == q]
            q_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in q_orders)
            q_disc = int(sum(pm[s["상품명"]]["가격"] * (100 - pm[s["상품명"]]["할인율"]) / 100 * s["수량"] for s in q_orders))
            q_disc_pcts[q] = int(q_disc * 100 / q_rev) if q_rev > 0 else 0
        q_disc_ranked = _rank_groups(q_disc_pcts)
        max_q_disc_pct = q_disc_ranked[0][1] if q_disc_ranked else 0
        disc_pct_diff = abs(max_q_disc_pct - overall_disc_pct)

        # T4 (Pattern D): 할인율을 5%p 줄인 카운터팩추얼
        counterfactual_rev = int(sum(
            pm[s["상품명"]]["가격"] * (100 - max(0, pm[s["상품명"]]["할인율"] - 5)) / 100 * s["수량"]
            for s in st
        ))
        actual_disc_rev = total_disc_rev
        counterfactual_diff = abs(counterfactual_rev - actual_disc_rev)

        # T5: 분기 avg-of-avg vs 전체 평균 차이 (유지)
        q_rev_map = _group_sum(st, lambda s: s["분기"],
                               lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        q_counts = _group_count(st, lambda s: s["분기"])
        q_avgs = {q: q_rev_map.get(q, 0) / q_counts.get(q, 1) for q in q_rev_map}
        avg_of_q_avgs = int(sum(q_avgs.values()) / len(q_avgs)) if q_avgs else 0
        overall_avg = int(total_rev / len(st)) if st else 0
        diff_avgs = abs(avg_of_q_avgs - overall_avg)

        # T6 (new): 카테고리 매출 변동계수
        cat_rev = _group_sum(st, lambda s: pm[s["상품명"]]["카테고리"],
                             lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        cat_rev_vals = list(cat_rev.values())
        cat_rev_mean = sum(cat_rev_vals) / len(cat_rev_vals) if cat_rev_vals else 1
        cat_rev_sd = _std_dev(cat_rev_vals)
        cv = int(cat_rev_sd * 100 / cat_rev_mean) if cat_rev_mean > 0 else 0

        question_templates = [
            (f"카테고리별 평균 가격의 전체 평균은 얼마입니까? (먼저 카테고리별 평균을 구한 후, 그 평균들의 평균을 구하세요. 소수점 버림)",
             avg_of_avgs, 1,
             f"1단계: 카테고리별 상품 그룹화\n"
             f"2단계: 카테고리별 평균: {', '.join(f'{k}({v:.0f})' for k,v in cat_avg.items())}\n"
             f"3단계: 평균의 평균 = {avg_of_avgs}\n최종 답: {avg_of_avgs}"),

            (f"골드 고객만의 카테고리별 평균 주문 매출의 평균과 전체 고객의 카테고리별 평균 주문 매출의 평균을 구하세요. 그 절대 차이는 얼마입니까? (카테고리별 평균 = 해당 카테고리 총매출 / 주문 수, 이후 평균, 소수점 버림)",
             avg_of_avg_diff, 1,
             f"1단계: 골드 전용 카테고리별 평균 주문 매출: {', '.join(f'{k}({int(v)})' for k,v in gold_cat_avg.items())}\n"
             f"2단계: 골드 평균의 평균 = {gold_avg_of_avg}\n"
             f"3단계: 전체 카테고리별 평균 주문 매출: {', '.join(f'{k}({int(v)})' for k,v in all_cat_avg_rev.items())}\n"
             f"4단계: 전체 평균의 평균 = {overall_avg_of_avg}\n"
             f"5단계: 차이 = {avg_of_avg_diff}\n최종 답: {avg_of_avg_diff}"),

            (f"각 분기별로 정가 매출 대비 할인 매출 비율을 계산하세요(할인가 = 가격×(100-할인율)/100, 항목별 버림). 이 비율이 가장 높은 분기와 전체 할인 매출 비율의 차이는 얼마입니까? (절대값)",
             disc_pct_diff, 1,
             f"1단계: 분기별 할인 매출 비율: {', '.join(f'{k}({v})' for k,v in q_disc_ranked)}\n"
             f"2단계: 최대 = {max_q_disc_pct}%\n"
             f"3단계: 전체 할인 비율 = {overall_disc_pct}%\n"
             f"4단계: 차이 = {disc_pct_diff}\n최종 답: {disc_pct_diff}"),

            (f"모든 상품의 할인율이 5%p 낮아진다면(최소 0%), 실제 할인 매출과 가정 할인 매출의 차이는 얼마입니까? (절대값, 할인가 = 가격×(100-할인율)/100 항목별 버림)",
             counterfactual_diff, 1,
             f"1단계: 실제 할인 매출 = {actual_disc_rev}\n"
             f"2단계: 가정(할인율-5%p, 최소 0) 매출 = {counterfactual_rev}\n"
             f"3단계: 차이 = {counterfactual_diff}\n최종 답: {counterfactual_diff}"),

            (f"분기별 주문당 평균 매출의 평균과 전체 주문당 평균 매출의 차이는 얼마입니까? (절대값, 소수점 버림)",
             diff_avgs, 1,
             f"1단계: 분기별 매출/주문수: {', '.join(f'{k}({int(v)})' for k,v in q_avgs.items())}\n"
             f"2단계: 분기별 평균의 평균 = {avg_of_q_avgs}\n"
             f"3단계: 전체 주문당 평균 = {overall_avg}\n"
             f"4단계: 차이 = {diff_avgs}\n최종 답: {diff_avgs}"),

            (f"카테고리별 매출의 변동계수는 얼마입니까? (CV = 표준편차 / 평균 × 100, 정수로 버림. 모집단 표준편차 사용)",
             cv, 1,
             f"1단계: 카테고리별 매출: {', '.join(f'{k}({v})' for k,v in _rank_groups(cat_rev))}\n"
             f"2단계: 평균 = {int(cat_rev_mean)}\n"
             f"3단계: 표준편차 = {int(cat_rev_sd)}\n"
             f"4단계: CV = {cv}\n최종 답: {cv}"),
        ]

    question, answer, solution = _weighted_choice(
        rng, _calibrate_template_weights(config.difficulty, question_templates)
    )

    return {
        "type": ProblemType.CONDITIONAL_AGGREGATION.value,
        "difficulty": config.difficulty,
        "tables": _make_tables_dict(pt, st, ct),
        "question": question,
        "answer": answer,
        "answer_type": "number" if isinstance(answer, (int, float)) else "text",
        "solution": solution,
    }


def generate_array_computation_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """배열 연산 문제 생성 (v3)"""
    pt, st, ct, pm, cm = _generate_all_tables(config, rng)

    categories = list(set(p["카테고리"] for p in pt))
    regions_in_orders = list(set(s["지역"] for s in st))

    if config.difficulty == "easy":
        product_prices = {p["상품명"]: p["가격"] for p in pt}
        product_discounts = {p["상품명"]: p["할인율"] for p in pt}

        # T1: 전체 매출
        total_sales = sum(product_prices[s["상품명"]] * s["수량"] for s in st)

        # T2: 전체 할인 매출
        total_disc_sales = int(sum(
            product_prices[s["상품명"]] * (100 - product_discounts[s["상품명"]]) / 100 * s["수량"]
            for s in st
        ))

        # T3: 카테고리 매출
        tgt_cat = rng.choice(categories)
        cat_prods = set(p["상품명"] for p in pt if p["카테고리"] == tgt_cat)
        cat_orders = [s for s in st if s["상품명"] in cat_prods]
        cat_sales = sum(product_prices[s["상품명"]] * s["수량"] for s in cat_orders)

        # T4: 고수량 조건 매출
        qty_thresh = rng.choice([15, 20, 25, 30])
        high_qty_orders = [s for s in st if s["수량"] > qty_thresh]
        high_qty_sales = sum(product_prices[s["상품명"]] * s["수량"] for s in high_qty_orders)

        # NEW T5: 지역 매출 교차 조회
        tgt_region = rng.choice(list(set(s["지역"] for s in st)))
        reg_orders = [s for s in st if s["지역"] == tgt_region]
        reg_revenue = sum(product_prices[s["상품명"]] * s["수량"] for s in reg_orders)

        # NEW T6: 총 매출 vs 할인 매출 차
        rev_diff = total_sales - total_disc_sales

        # NEW T7: 재고 중앙값 초과 상품 매출
        stocks = sorted(p["재고"] for p in pt)
        median_stock = stocks[len(stocks) // 2]
        above_med_prods = set(p["상품명"] for p in pt if p["재고"] > median_stock)
        above_med_orders = [s for s in st if s["상품명"] in above_med_prods]
        above_med_rev = sum(product_prices[s["상품명"]] * s["수량"] for s in above_med_orders)

        question_templates = [
            (f"전체 주문의 총 매출액은 얼마입니까? (상품 테이블에서 가격 조회, 매출 = 가격 × 수량)",
             total_sales, 1,
             f"1단계: 각 주문의 가격 조회\n"
             f"2단계: 가격 × 수량 계산\n"
             f"3단계: 합계 = {total_sales}\n최종 답: {total_sales}"),

            (f"전체 주문의 할인 적용 총 매출액은 얼마입니까? (할인가 = 가격×(100-할인율)/100, 소수점 버림 후 수량 곱하기의 합)",
             total_disc_sales, 1,
             f"1단계: 각 주문의 가격, 할인율 조회\n"
             f"2단계: 할인가 × 수량\n"
             f"3단계: 합계 = {total_disc_sales}\n최종 답: {total_disc_sales}"),

            (f"'{tgt_cat}' 카테고리 상품의 총 매출액은 얼마입니까?",
             cat_sales, 1,
             f"1단계: '{tgt_cat}' 상품: {', '.join(list(cat_prods)[:5])}\n"
             f"2단계: 주문 필터링: {len(cat_orders)}건\n"
             f"3단계: 합계 = {cat_sales}\n최종 답: {cat_sales}"),

            (f"수량이 {qty_thresh}을 초과하는 주문의 총 매출액은 얼마입니까?",
             high_qty_sales, 1,
             f"1단계: 수량 > {qty_thresh} 주문 필터링: {len(high_qty_orders)}건\n"
             f"2단계: 가격 조회, 매출 계산\n"
             f"3단계: 합계 = {high_qty_sales}\n최종 답: {high_qty_sales}"),

            # NEW: 지역 매출 교차 조회 (weight=2)
            (f"'{tgt_region}' 지역 주문의 총 매출액은 얼마입니까? (상품 테이블에서 가격 조회)",
             reg_revenue, 2,
             f"1단계: '{tgt_region}' 주문 필터링: {len(reg_orders)}건\n"
             f"2단계: 가격 × 수량 합산\n"
             f"3단계: 합계 = {reg_revenue}\n최종 답: {reg_revenue}"),

            # NEW: 총 매출 vs 할인 매출 차 (weight=2)
            (f"총 매출액과 할인 적용 총 매출액의 차이는 얼마입니까? (총매출 - 할인매출, 할인매출은 소수점 버림 후 합산)",
             rev_diff, 2,
             f"1단계: 총 매출 = {total_sales}\n"
             f"2단계: 할인 매출 = {total_disc_sales}\n"
             f"3단계: 차이 = {rev_diff}\n최종 답: {rev_diff}"),

            # NEW: 재고 중앙값 초과 상품 매출 (weight=2)
            (f"재고가 중앙값({median_stock}) 초과인 상품의 총 매출액은 얼마입니까? (중앙값 = 정렬 후 가운데 값)",
             above_med_rev, 2,
             f"1단계: 재고 정렬, 중앙값 = {median_stock}\n"
             f"2단계: 재고 > {median_stock} 상품: {len(above_med_prods)}개\n"
             f"3단계: 해당 주문: {len(above_med_orders)}건\n"
             f"4단계: 매출 합계 = {above_med_rev}\n최종 답: {above_med_rev}"),
        ]

    elif config.difficulty == "medium":
        product_prices = {p["상품명"]: p["가격"] for p in pt}
        product_discounts = {p["상품명"]: p["할인율"] for p in pt}

        total_rev = sum(product_prices[s["상품명"]] * s["수량"] for s in st)

        # T1 (REPLACED): 평균 수량 초과 주문의 할인 매출
        avg_qty = total_rev / len(st) if st else 0
        order_qtys = [s["수량"] for s in st]
        avg_order_qty = sum(order_qtys) / len(order_qtys) if order_qtys else 0
        above_avg_qty_orders = [s for s in st if s["수량"] > avg_order_qty]
        above_avg_disc_rev = int(sum(
            product_prices[s["상품명"]] * (100 - product_discounts[s["상품명"]]) / 100 * s["수량"]
            for s in above_avg_qty_orders
        ))

        # T2: 최다 매출 지역 매출
        reg_rev = _group_sum(st, lambda s: s["지역"],
                             lambda s: product_prices[s["상품명"]] * s["수량"])
        reg_ranked = _rank_groups(reg_rev)
        top_reg = reg_ranked[0][0] if reg_ranked else ""
        top_reg_rev = reg_ranked[0][1] if reg_ranked else 0

        # T3: Gold-Silver 차
        gold_cids = set(c["고객번호"] for c in ct if c["등급"] == "골드")
        gold_orders = [s for s in st if s["고객번호"] in gold_cids]
        gold_sales = sum(product_prices[s["상품명"]] * s["수량"] for s in gold_orders)
        silver_cids = set(c["고객번호"] for c in ct if c["등급"] == "실버")
        silver_orders = [s for s in st if s["고객번호"] in silver_cids]
        silver_sales = sum(product_prices[s["상품명"]] * s["수량"] for s in silver_orders)
        gold_silver_diff = gold_sales - silver_sales

        # T4: 분기 할인 매출
        tgt_q = rng.choice(QUARTERS)
        q_orders = [s for s in st if s["분기"] == tgt_q]
        q_disc_rev = int(sum(
            product_prices[s["상품명"]] * (100 - product_discounts[s["상품명"]]) / 100 * s["수량"]
            for s in q_orders
        ))

        # T5 (REPLACED): 카테고리별 매출 범위 (최대 - 최소)
        cat_rev = _group_sum(st, lambda s: pm[s["상품명"]]["카테고리"],
                             lambda s: product_prices[s["상품명"]] * s["수량"])
        cat_ranked = _rank_groups(cat_rev)
        cat_rev_range = cat_ranked[0][1] - cat_ranked[-1][1] if len(cat_ranked) >= 2 else 0

        # T6 (NEW): 분기별 할인 매출 최대-최소 차이
        q_disc_revs = {}
        for q in QUARTERS:
            qo = [s for s in st if s["분기"] == q]
            q_disc_revs[q] = int(sum(
                product_prices[s["상품명"]] * (100 - product_discounts[s["상품명"]]) / 100 * s["수량"]
                for s in qo
            ))
        q_disc_ranked = _rank_groups(q_disc_revs)
        q_disc_diff = q_disc_ranked[0][1] - q_disc_ranked[-1][1] if len(q_disc_ranked) >= 2 else 0

        question_templates = [
            # REPLACED T1: 평균 수량 초과 할인 매출
            (f"평균 주문 수량({int(avg_order_qty)})을 초과하는 주문의 할인 적용 매출 합계는 얼마입니까? (할인가 = 가격×(100-할인율)/100, 소수점 버림)",
             above_avg_disc_rev, 2,
             f"1단계: 평균 수량 = {int(avg_order_qty)}\n"
             f"2단계: 수량 > 평균 주문: {len(above_avg_qty_orders)}건\n"
             f"3단계: 할인 매출 합계\n"
             f"4단계: 합계 = {above_avg_disc_rev}\n최종 답: {above_avg_disc_rev}"),

            (f"매출이 가장 높은 지역의 매출액은 얼마입니까? (매출 = 가격 × 수량)",
             top_reg_rev, 1,
             f"1단계: 주문별 매출 계산\n"
             f"2단계: 지역별 합산: {', '.join(f'{k}({v})' for k,v in reg_ranked)}\n"
             f"3단계: 1위 = '{top_reg}' ({top_reg_rev})\n최종 답: {top_reg_rev}"),

            (f"골드 등급 고객과 실버 등급 고객의 매출 차이는 얼마입니까? (골드 - 실버)",
             gold_silver_diff, 1,
             f"1단계: 골드 매출 = {gold_sales}\n"
             f"2단계: 실버 매출 = {silver_sales}\n"
             f"3단계: 차이 = {gold_silver_diff}\n최종 답: {gold_silver_diff}"),

            (f"{tgt_q} 주문의 할인 적용 총 매출액은 얼마입니까? (할인가 = 가격×(100-할인율)/100, 소수점 버림)",
             q_disc_rev, 1,
             f"1단계: {tgt_q} 주문: {len(q_orders)}건\n"
             f"2단계: 가격, 할인율 조회\n"
             f"3단계: 합계 = {q_disc_rev}\n최종 답: {q_disc_rev}"),

            # REPLACED T5: 카테고리별 매출 범위
            (f"카테고리별 매출에서 최대와 최소의 차이는 얼마입니까? (매출 = 가격 × 수량)",
             cat_rev_range, 2,
             f"1단계: 카테고리별 매출: {', '.join(f'{k}({v})' for k,v in cat_ranked)}\n"
             f"2단계: 최대 = {cat_ranked[0][1] if cat_ranked else 0}, 최소 = {cat_ranked[-1][1] if cat_ranked else 0}\n"
             f"3단계: 범위 = {cat_rev_range}\n최종 답: {cat_rev_range}"),

            # NEW T6: 분기별 할인 매출 최대-최소
            (f"분기별 할인 적용 매출액 중 최대와 최소의 차이는 얼마입니까? (할인가 = 가격×(100-할인율)/100, 분기별 합계 소수점 버림)",
             q_disc_diff, 2,
             f"1단계: 분기별 할인 매출 합산\n"
             f"2단계: {', '.join(f'{k}({v})' for k,v in q_disc_ranked)}\n"
             f"3단계: 최대 - 최소 = {q_disc_diff}\n최종 답: {q_disc_diff}"),
        ]

    else:  # hard
        product_prices = {p["상품명"]: p["가격"] for p in pt}
        product_discounts = {p["상품명"]: p["할인율"] for p in pt}
        total_rev = sum(product_prices[s["상품명"]] * s["수량"] for s in st)

        # T1 (KEEP): 매출 가중 할인율
        weighted_disc_sum = sum(
            product_prices[s["상품명"]] * s["수량"] * product_discounts[s["상품명"]]
            for s in st
        )
        rev_weighted_disc = int(weighted_disc_sum / total_rev) if total_rev > 0 else 0

        # T2 (REPLACED): 등급×카테고리 피벗 최대/최소 비율
        mem_cat_rev = {}
        for s in st:
            mem = cm[s["고객번호"]]["등급"]
            cat = pm[s["상품명"]]["카테고리"]
            key = (mem, cat)
            rev = product_prices[s["상품명"]] * s["수량"]
            mem_cat_rev[key] = mem_cat_rev.get(key, 0) + rev
        if mem_cat_rev:
            mc_max = max(mem_cat_rev.values())
            mc_min = min(v for v in mem_cat_rev.values() if v > 0) if any(v > 0 for v in mem_cat_rev.values()) else 1
            mc_ratio = int(mc_max * 100 / mc_min) if mc_min > 0 else 0
        else:
            mc_max, mc_min, mc_ratio = 0, 1, 0

        # T3 (REPLACED): 지역별 할인 매출 비교 (전체 vs 2021+ 가입 고객만)
        post2021_cids = set(c["고객번호"] for c in ct if c["가입연도"] >= 2021)
        reg_disc_all = {}
        reg_disc_post = {}
        for s in st:
            reg = s["지역"]
            disc_rev = product_prices[s["상품명"]] * (100 - product_discounts[s["상품명"]]) / 100 * s["수량"]
            reg_disc_all[reg] = reg_disc_all.get(reg, 0) + disc_rev
            if s["고객번호"] in post2021_cids:
                reg_disc_post[reg] = reg_disc_post.get(reg, 0) + disc_rev
        reg_diff_vals = {}
        for reg in reg_disc_all:
            reg_diff_vals[reg] = int(reg_disc_all[reg]) - int(reg_disc_post.get(reg, 0))
        max_reg_diff = max(reg_diff_vals.values()) if reg_diff_vals else 0

        # T4 (REPLACED): 중앙값 기준 매출 필터 → 전체 대비 %
        order_revs = sorted(product_prices[s["상품명"]] * s["수량"] for s in st)
        median_rev = order_revs[len(order_revs) // 2] if order_revs else 0
        above_med_orders = [s for s in st if product_prices[s["상품명"]] * s["수량"] > median_rev]
        above_med_total = sum(product_prices[s["상품명"]] * s["수량"] for s in above_med_orders)
        above_med_pct = int(above_med_total * 100 / total_rev) if total_rev > 0 else 0

        # T5 (REPLACED): 카테고리별 할인/정가 매출 비율 범위
        cat_full_rev = _group_sum(st, lambda s: pm[s["상품명"]]["카테고리"],
                                  lambda s: product_prices[s["상품명"]] * s["수량"])
        cat_disc_rev = _group_sum(st, lambda s: pm[s["상품명"]]["카테고리"],
                                  lambda s: product_prices[s["상품명"]] * (100 - product_discounts[s["상품명"]]) / 100 * s["수량"])
        cat_ratios = {}
        for cat in cat_full_rev:
            if cat_full_rev[cat] > 0:
                cat_ratios[cat] = int(cat_disc_rev.get(cat, 0) * 100 / cat_full_rev[cat])
        ratio_range = max(cat_ratios.values()) - min(cat_ratios.values()) if len(cat_ratios) >= 2 else 0

        # T6 (KEEP): 할인 vs 무할인 가격 차
        disc_prods = [p for p in pt if p["할인율"] > 0]
        nodisc_prods = [p for p in pt if p["할인율"] == 0]
        if disc_prods:
            disc_weighted_price = int(sum(
                p["가격"] * (100 - p["할인율"]) / 100 for p in disc_prods
            ) / len(disc_prods))
        else:
            disc_weighted_price = 0
        nodisc_avg_price = int(sum(p["가격"] for p in nodisc_prods) / len(nodisc_prods)) if nodisc_prods else 0
        price_gap = abs(disc_weighted_price - nodisc_avg_price)

        question_templates = [
            # T1 (KEEP): 매출 가중 할인율
            (f"매출 가중 평균 할인율은 얼마입니까? (= sum(매출×할인율)/sum(매출), 정수로 버림)",
             rev_weighted_disc, 1,
             f"1단계: 각 주문의 매출 × 할인율\n"
             f"2단계: 가중 합계 = {weighted_disc_sum}\n"
             f"3단계: 총 매출 = {total_rev}\n"
             f"4단계: 평균 = {rev_weighted_disc}\n최종 답: {rev_weighted_disc}"),

            # T2 (REPLACED): 등급×카테고리 피벗 비율
            (f"등급×카테고리 매출 표에서 최대 셀 값은 최소 양수 셀 값의 몇 배입니까? (최대×100/최소, 소수점 버림)",
             mc_ratio, 2,
             f"1단계: 각 (등급, 카테고리) 쌍의 매출 계산\n"
             f"2단계: 최대 = {mc_max}, 최소(양수) = {mc_min}\n"
             f"3단계: 비율 = {mc_ratio}\n최종 답: {mc_ratio}"),

            # T3 (REPLACED): 지역별 할인 매출, 전체 vs 2021 이후 가입 고객
            (f"각 지역의 전체 고객 할인 매출과 2021년 이후 가입 고객 할인 매출의 차이 중 최대값은 얼마입니까? (할인가 = 가격×(100-할인율)/100, 지역별 소수점 버림)",
             max_reg_diff, 2,
             f"1단계: 2021년 이후 가입 고객: {len(post2021_cids)}명\n"
             f"2단계: 지역별 전체 vs 2021+ 할인 매출 계산\n"
             f"3단계: 차이: {', '.join(f'{k}({v})' for k,v in sorted(reg_diff_vals.items(), key=lambda x: -x[1]))}\n"
             f"4단계: 최대 차이 = {max_reg_diff}\n최종 답: {max_reg_diff}"),

            # T4 (REPLACED): 중앙값 필터 매출 비율
            (f"주문별 매출(가격×수량)이 중앙값을 초과하는 주문의 매출이 전체의 몇 %입니까? (중앙값 = 정렬 후 가운데 값, 소수점 버림)",
             above_med_pct, 2,
             f"1단계: 주문별 매출 계산, 정렬\n"
             f"2단계: 중앙값 = {median_rev}\n"
             f"3단계: 초과 주문: {len(above_med_orders)}건, 매출 = {above_med_total}\n"
             f"4단계: 비율 = {above_med_pct}%\n최종 답: {above_med_pct}"),

            # T5 (REPLACED): 카테고리별 할인/정가 비율 범위
            (f"카테고리별 할인 보존율(= 할인매출×100/정가매출, 카테고리별 소수점 버림)의 범위(최대-최소)는 얼마입니까?",
             ratio_range, 2,
             f"1단계: 카테고리별 정가 매출, 할인 매출 계산\n"
             f"2단계: 비율: {', '.join(f'{k}({v})' for k,v in sorted(cat_ratios.items(), key=lambda x: -x[1]))}\n"
             f"3단계: 범위 = {ratio_range}\n최종 답: {ratio_range}"),

            # T6 (KEEP): 할인 vs 무할인 가격 차
            (f"할인 상품(할인율>0)의 평균 할인가와 무할인 상품(할인율=0)의 평균 가격 차이는 얼마입니까? (절대값, 소수점 버림)",
             price_gap, 1,
             f"1단계: 할인 상품 {len(disc_prods)}개의 평균 할인가 = {disc_weighted_price}\n"
             f"2단계: 무할인 상품 {len(nodisc_prods)}개의 평균 가격 = {nodisc_avg_price}\n"
             f"3단계: 차이 = {price_gap}\n최종 답: {price_gap}"),
        ]

    question, answer, solution = _weighted_choice(
        rng, _calibrate_template_weights(config.difficulty, question_templates)
    )

    return {
        "type": ProblemType.ARRAY_COMPUTATION.value,
        "difficulty": config.difficulty,
        "tables": _make_tables_dict(pt, st, ct),
        "question": question,
        "answer": answer,
        "answer_type": "number" if isinstance(answer, (int, float)) else "text",
        "solution": solution,
    }


def generate_multi_condition_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """다중 조건 문제 생성 (v3)"""
    pt, st, ct, pm, cm = _generate_all_tables(config, rng)

    categories = list(set(p["카테고리"] for p in pt))
    regions_in_orders = list(set(s["지역"] for s in st))
    tgt_cat = rng.choice(categories)

    if config.difficulty == "easy":
        tgt_region = rng.choice(regions_in_orders)
        product_cats = {p["상품명"]: p["카테고리"] for p in pt}

        # T1: 지역+카테고리 건수
        filtered_rc = [s for s in st
                       if s["지역"] == tgt_region
                       and product_cats.get(s["상품명"]) == tgt_cat]
        rc_count = len(filtered_rc)

        # T2: 지역+분기 수량
        tgt_quarter = rng.choice(QUARTERS)
        filtered_rq = [s for s in st
                       if s["지역"] == tgt_region and s["분기"] == tgt_quarter]
        rq_qty = sum(s["수량"] for s in filtered_rq)

        # T3: 지역 매출
        reg_orders = [s for s in st if s["지역"] == tgt_region]
        reg_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in reg_orders)

        # T4: Gold 총 수량
        gold_cids = set(c["고객번호"] for c in ct if c["등급"] == "골드")
        gold_orders = [s for s in st if s["고객번호"] in gold_cids]
        gold_qty = sum(s["수량"] for s in gold_orders)

        # T5: 지역+카테고리 매출
        rc_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in filtered_rc)

        # NEW T6: 카테고리+분기 주문 건수
        cat_prods_set = set(p["상품명"] for p in pt if p["카테고리"] == tgt_cat)
        cat_q_orders = [s for s in st if s["상품명"] in cat_prods_set and s["분기"] == tgt_quarter]
        cat_q_count = len(cat_q_orders)

        # NEW T7: 지역 할인 매출
        reg_disc_rev = int(sum(
            pm[s["상품명"]]["가격"] * (100 - pm[s["상품명"]]["할인율"]) / 100 * s["수량"]
            for s in reg_orders
        ))

        # NEW T8: 지역 내 평균 가격 초과 상품 건수
        avg_price = sum(p["가격"] for p in pt) / len(pt) if pt else 0
        high_price_prods = set(p["상품명"] for p in pt if p["가격"] > avg_price)
        high_price_reg_orders = [s for s in st if s["상품명"] in high_price_prods and s["지역"] == tgt_region]
        high_price_reg_count = len(high_price_reg_orders)

        question_templates = [
            (f"'{tgt_region}'에서 '{tgt_cat}' 상품 주문은 몇 건입니까?",
             rc_count, 1,
             f"1단계: 상품 테이블에서 '{tgt_cat}' 상품 식별\n"
             f"2단계: 지역='{tgt_region}' AND 카테고리='{tgt_cat}' 필터링\n"
             f"3단계: 건수 = {rc_count}\n최종 답: {rc_count}"),

            (f"'{tgt_region}'에서 {tgt_quarter} 총 주문 수량은 얼마입니까?",
             rq_qty, 1,
             f"1단계: 지역='{tgt_region}' AND 분기='{tgt_quarter}' 필터링: {len(filtered_rq)}건\n"
             f"2단계: 수량 합계 = {rq_qty}\n최종 답: {rq_qty}"),

            (f"'{tgt_region}' 전체 주문의 총 매출액은 얼마입니까? (상품 테이블에서 가격 조회)",
             reg_rev, 1,
             f"1단계: '{tgt_region}' 주문 필터링: {len(reg_orders)}건\n"
             f"2단계: 가격×수량 합계 = {reg_rev}\n최종 답: {reg_rev}"),

            (f"골드 등급 고객의 총 주문 수량은 얼마입니까?",
             gold_qty, 1,
             f"1단계: 골드 고객: {len(gold_cids)}명\n"
             f"2단계: 골드 주문: {len(gold_orders)}건\n"
             f"3단계: 수량 합계 = {gold_qty}\n최종 답: {gold_qty}"),

            (f"'{tgt_region}'에서 '{tgt_cat}' 상품의 총 매출액은 얼마입니까? (상품 테이블에서 가격 조회)",
             rc_rev, 1,
             f"1단계: '{tgt_region}'의 '{tgt_cat}' 주문: {len(filtered_rc)}건\n"
             f"2단계: 가격×수량 합계 = {rc_rev}\n최종 답: {rc_rev}"),

            # NEW: 카테고리+분기 주문 건수 (weight=2)
            (f"'{tgt_cat}' 카테고리 상품의 {tgt_quarter} 주문은 몇 건입니까?",
             cat_q_count, 2,
             f"1단계: '{tgt_cat}' 상품 식별\n"
             f"2단계: 분기='{tgt_quarter}' AND 카테고리='{tgt_cat}' 필터링\n"
             f"3단계: 건수 = {cat_q_count}\n최종 답: {cat_q_count}"),

            # NEW: 지역 할인 매출 (weight=2)
            (f"'{tgt_region}' 주문의 할인 적용 총 매출액은 얼마입니까? (할인가 = 가격×(100-할인율)/100, 소수점 버림)",
             reg_disc_rev, 2,
             f"1단계: '{tgt_region}' 주문 필터링: {len(reg_orders)}건\n"
             f"2단계: 할인가 × 수량 합산\n"
             f"3단계: 합계 = {reg_disc_rev}\n최종 답: {reg_disc_rev}"),

            # NEW: 지역 내 평균 가격 초과 상품 (weight=2)
            (f"'{tgt_region}'에서 평균 가격({int(avg_price)}) 초과 상품의 주문 건수는 몇 건입니까?",
             high_price_reg_count, 2,
             f"1단계: 평균 가격 = {int(avg_price)}\n"
             f"2단계: 가격 > 평균 상품: {len(high_price_prods)}개\n"
             f"3단계: '{tgt_region}'에서 해당 주문: {high_price_reg_count}건\n최종 답: {high_price_reg_count}"),
        ]

    elif config.difficulty == "medium":
        # T1: 카테고리 지역 1위
        cat_prods = set(p["상품명"] for p in pt if p["카테고리"] == tgt_cat)
        cat_orders = [s for s in st if s["상품명"] in cat_prods]
        reg_qty = _group_sum(cat_orders, lambda s: s["지역"], lambda s: s["수량"])
        reg_ranked = _rank_groups(reg_qty)
        top_region = reg_ranked[0][0] if reg_ranked else ""

        # T2: 지역 매출 2위
        region_rev = _group_sum(st, lambda s: s["지역"],
                                lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        rev_ranked = _rank_groups(region_rev)
        second_rev = rev_ranked[1][1] if len(rev_ranked) >= 2 else 0

        # T3: 주문 최다 분기
        q_counts = _group_count(st, lambda s: s["분기"])
        q_count_ranked = _rank_groups(q_counts)
        top_q = q_count_ranked[0][0] if q_count_ranked else "1분기"

        # T4 (REPLACED): Gold 고유 상품 수 최다 카테고리
        gold_cids = set(c["고객번호"] for c in ct if c["등급"] == "골드")
        gold_orders = [s for s in st if s["고객번호"] in gold_cids]
        gold_cat_prods = {}
        for s in gold_orders:
            cat = pm[s["상품명"]]["카테고리"]
            if cat not in gold_cat_prods:
                gold_cat_prods[cat] = set()
            gold_cat_prods[cat].add(s["상품명"])
        gold_cat_distinct = {k: len(v) for k, v in gold_cat_prods.items()}
        gold_cat_ranked = _rank_groups(gold_cat_distinct)
        top_gold_cat = gold_cat_ranked[0][0] if gold_cat_ranked else ""

        # T5 (REPLACED): 주문 최다 카테고리 → 해당 카테고리 할인 매출
        cat_order_counts = _group_count(st, lambda s: pm[s["상품명"]]["카테고리"])
        cat_count_ranked = _rank_groups(cat_order_counts)
        top_count_cat = cat_count_ranked[0][0] if cat_count_ranked else ""
        top_count_cat_prods = set(p["상품명"] for p in pt if p["카테고리"] == top_count_cat)
        top_count_cat_orders = [s for s in st if s["상품명"] in top_count_cat_prods]
        top_count_cat_disc_rev = int(sum(
            pm[s["상품명"]]["가격"] * (100 - pm[s["상품명"]]["할인율"]) / 100 * s["수량"]
            for s in top_count_cat_orders
        ))

        # T6 (NEW): 지역별 Gold vs 비Gold 주문 건수 차 (전체 대비)
        tgt_region = rng.choice(regions_in_orders)
        reg_total_orders = len([s for s in st if s["지역"] == tgt_region])
        reg_gold_orders = len([s for s in st if s["지역"] == tgt_region and s["고객번호"] in gold_cids])
        reg_non_gold = reg_total_orders - reg_gold_orders

        question_templates = [
            (f"'{tgt_cat}' 카테고리에서 수량 기준 주문이 가장 많은 지역은 어디입니까? (지역명으로 답하세요)",
             top_region, 1,
             f"1단계: '{tgt_cat}' 상품 식별\n"
             f"2단계: 주문 필터링: {len(cat_orders)}건\n"
             f"3단계: 지역별: {', '.join(f'{k}({v})' for k,v in reg_ranked)}\n"
             f"4단계: 1위 = '{top_region}'\n최종 답: {top_region}"),

            (f"전체 지역 중 2번째로 매출이 높은 지역의 매출액은 얼마입니까?",
             second_rev, 1,
             f"1-2단계: 지역별 매출: {', '.join(f'{k}({v})' for k,v in rev_ranked)}\n"
             f"3단계: 2위 = {second_rev}\n최종 답: {second_rev}"),

            (f"주문 건수가 가장 많은 분기는 무엇입니까? (분기명으로 답하세요, 예: 1분기)",
             top_q, 1,
             f"1단계: 분기별 건수: {', '.join(f'{k}({v})' for k,v in q_count_ranked)}\n"
             f"2단계: 1위 = '{top_q}'\n최종 답: {top_q}"),

            # REPLACED T4: Gold 카테고리별 고유 상품 수
            (f"골드 등급 고객의 주문에서 고유 상품 수가 가장 많은 카테고리는 무엇입니까? (카테고리명으로 답하세요)",
             top_gold_cat, 2,
             f"1단계: 골드 고객: {len(gold_cids)}명\n"
             f"2단계: 골드 주문: {len(gold_orders)}건\n"
             f"3단계: 카테고리별 고유 상품 수: {', '.join(f'{k}({v})' for k,v in gold_cat_ranked)}\n"
             f"4단계: 1위 = '{top_gold_cat}'\n최종 답: {top_gold_cat}"),

            # REPLACED T5: 주문 최다 카테고리 → 할인 매출
            (f"주문 건수가 가장 많은 카테고리의 할인 적용 매출액은 얼마입니까? (할인가 = 가격×(100-할인율)/100, 소수점 버림)",
             top_count_cat_disc_rev, 2,
             f"1단계: 카테고리별 주문 건수: {', '.join(f'{k}({v})' for k,v in cat_count_ranked)}\n"
             f"2단계: 1위 = '{top_count_cat}'\n"
             f"3단계: '{top_count_cat}' 할인 매출 = {top_count_cat_disc_rev}\n최종 답: {top_count_cat_disc_rev}"),

            # NEW T6: 지역 Gold vs 비Gold 주문 건수 차
            (f"'{tgt_region}'에서 비골드 주문 건수와 골드 주문 건수의 차이는 얼마입니까? (비골드 - 골드)",
             reg_non_gold - reg_gold_orders, 2,
             f"1단계: '{tgt_region}' 전체 주문: {reg_total_orders}건\n"
             f"2단계: 골드 주문: {reg_gold_orders}건\n"
             f"3단계: 비골드 = {reg_non_gold}건\n"
             f"4단계: 차이 = {reg_non_gold - reg_gold_orders}\n최종 답: {reg_non_gold - reg_gold_orders}"),
        ]

    else:  # hard
        # T1 (KEEP): 소비 상위 3명 제외 + 지역
        tgt_region = rng.choice(regions_in_orders)
        cust_spend = _group_sum(st, lambda s: s["고객번호"],
                                lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        spend_ranked = _rank_groups(cust_spend)
        top3_cids = set(cid for cid, _ in spend_ranked[:3])
        excl_top3_reg = [s for s in st if s["고객번호"] not in top3_cids and s["지역"] == tgt_region]
        excl_top3_reg_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in excl_top3_reg)

        # T2 (REPLACED): 중앙값 초과 vs 이하 소비 → disc≥15% 매출 % 차
        all_spends = sorted(cust_spend.values())
        median_spend = all_spends[len(all_spends) // 2] if all_spends else 0
        above_med_cids = set(cid for cid, sp in cust_spend.items() if sp > median_spend)
        below_med_cids = set(cid for cid, sp in cust_spend.items() if sp <= median_spend)
        high_disc_prods_15 = set(p["상품명"] for p in pt if p["할인율"] >= 15)
        above_med_disc15_rev = sum(pm[s["상품명"]]["가격"] * s["수량"]
                                   for s in st if s["고객번호"] in above_med_cids and s["상품명"] in high_disc_prods_15)
        above_med_total = sum(pm[s["상품명"]]["가격"] * s["수량"]
                              for s in st if s["고객번호"] in above_med_cids)
        below_med_disc15_rev = sum(pm[s["상품명"]]["가격"] * s["수량"]
                                   for s in st if s["고객번호"] in below_med_cids and s["상품명"] in high_disc_prods_15)
        below_med_total = sum(pm[s["상품명"]]["가격"] * s["수량"]
                              for s in st if s["고객번호"] in below_med_cids)
        above_pct = int(above_med_disc15_rev * 100 / above_med_total) if above_med_total > 0 else 0
        below_pct = int(below_med_disc15_rev * 100 / below_med_total) if below_med_total > 0 else 0
        med_pct_diff = abs(above_pct - below_pct)

        # T3 (KEEP): Gold 카테고리 %
        gold_cids = set(c["고객번호"] for c in ct if c["등급"] == "골드")
        gold_orders = [s for s in st if s["고객번호"] in gold_cids]
        gold_total = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in gold_orders)
        cat_prods = set(p["상품명"] for p in pt if p["카테고리"] == tgt_cat)
        gold_cat_orders = [s for s in gold_orders if s["상품명"] in cat_prods]
        gold_cat_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in gold_cat_orders)
        gold_cat_pct = int(gold_cat_rev * 100 / gold_total) if gold_total > 0 else 0

        # T4 (REPLACED): 매출 1위 지역 → 상위 2명 제외 → 평균 주문 가치
        region_rev = _group_sum(st, lambda s: s["지역"],
                                lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        reg_rev_ranked = _rank_groups(region_rev)
        top_rev_region = reg_rev_ranked[0][0] if reg_rev_ranked else tgt_region
        top2_cids = set(cid for cid, _ in spend_ranked[:2])
        top_reg_excl_orders = [s for s in st if s["지역"] == top_rev_region and s["고객번호"] not in top2_cids]
        top_reg_excl_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in top_reg_excl_orders)
        top_reg_excl_avg = int(top_reg_excl_rev / len(top_reg_excl_orders)) if top_reg_excl_orders else 0

        # T5 (REPLACED): 주문 건수 3위 지역 → 최고 소비 고객 → 해당 지역 소비 %
        reg_counts = _group_count(st, lambda s: s["지역"])
        reg_count_ranked = _rank_groups(reg_counts)
        third_region = reg_count_ranked[2][0] if len(reg_count_ranked) >= 3 else (reg_count_ranked[-1][0] if reg_count_ranked else tgt_region)
        third_reg_orders = [s for s in st if s["지역"] == third_region]
        third_reg_cust_spend = _group_sum(third_reg_orders, lambda s: s["고객번호"],
                                          lambda s: pm[s["상품명"]]["가격"] * s["수량"])
        third_reg_spend_ranked = _rank_groups(third_reg_cust_spend)
        top_cust_in_third = third_reg_spend_ranked[0][0] if third_reg_spend_ranked else ""
        top_cust_total_spend = cust_spend.get(top_cust_in_third, 0)
        top_cust_third_spend = third_reg_cust_spend.get(top_cust_in_third, 0)
        top_cust_third_pct = int(top_cust_third_spend * 100 / top_cust_total_spend) if top_cust_total_spend > 0 else 0

        # T6 (REPLACED): Gold+Silver vs Bronze+None 주문당 평균 매출 → ratio×100
        gs_cids = set(c["고객번호"] for c in ct if c["등급"] in ("골드", "실버"))
        bn_cids = set(c["고객번호"] for c in ct if c["등급"] in ("브론즈", "없음"))
        gs_orders = [s for s in st if s["고객번호"] in gs_cids]
        bn_orders = [s for s in st if s["고객번호"] in bn_cids]
        gs_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in gs_orders)
        bn_rev = sum(pm[s["상품명"]]["가격"] * s["수량"] for s in bn_orders)
        gs_avg = gs_rev / len(gs_orders) if gs_orders else 0
        bn_avg = bn_rev / len(bn_orders) if bn_orders else 1
        gs_bn_ratio = int(gs_avg * 100 / bn_avg) if bn_avg > 0 else 0

        question_templates = [
            # T1 (KEEP): 소비 상위 3명 제외 + 지역
            (f"소비액 상위 3명의 고객을 제외하고 '{tgt_region}'에서의 매출액은 얼마입니까?",
             excl_top3_reg_rev, 1,
             f"1단계: 고객별 소비액: {', '.join(f'{k}({v})' for k,v in spend_ranked[:5])}\n"
             f"2단계: 상위 3명 = {', '.join(top3_cids)}\n"
             f"3단계: 제외 후 '{tgt_region}' 주문: {len(excl_top3_reg)}건\n"
             f"4단계: 매출 = {excl_top3_reg_rev}\n최종 답: {excl_top3_reg_rev}"),

            # T2 (REPLACED): 중앙값 초과 vs 이하 disc≥15% 매출 %
            (f"고객을 소비액 중앙값 기준으로 나눌 때, 할인율 15% 이상 상품 매출 비율(%)의 차이는 얼마입니까? (각 그룹의 %를 소수점 버림 후 절대값)",
             med_pct_diff, 2,
             f"1단계: 소비액 정렬, 중앙값 = {median_spend}\n"
             f"2단계: 중앙값 초과: 할인≥15% 비율 = {above_pct}%\n"
             f"3단계: 중앙값 이하: 할인≥15% 비율 = {below_pct}%\n"
             f"4단계: 차이 = {med_pct_diff}\n최종 답: {med_pct_diff}"),

            # T3 (KEEP): Gold 카테고리 %
            (f"골드 고객 전체 매출 중 '{tgt_cat}' 카테고리 비율은 몇 %입니까? (소수점 버림)",
             gold_cat_pct, 1,
             f"1단계: 골드 고객: {len(gold_cids)}명\n"
             f"2단계: 골드 전체 매출 = {gold_total}\n"
             f"3단계: 골드 '{tgt_cat}' 매출 = {gold_cat_rev}\n"
             f"4단계: % = {gold_cat_pct}\n최종 답: {gold_cat_pct}"),

            # T4 (REPLACED): 매출 1위 지역 → 상위 2명 제외 → 평균 주문 가치
            (f"매출 1위 지역('{top_rev_region}')에서 소비 상위 2명 제외 후 주문당 평균 매출은 얼마입니까? (소수점 버림)",
             top_reg_excl_avg, 2,
             f"1단계: 매출 1위 지역 = '{top_rev_region}'\n"
             f"2단계: 상위 2명: {', '.join(top2_cids)}\n"
             f"3단계: '{top_rev_region}' 제외 후 주문: {len(top_reg_excl_orders)}건, 매출 = {top_reg_excl_rev}\n"
             f"4단계: 평균 = {top_reg_excl_avg}\n최종 답: {top_reg_excl_avg}"),

            # T5 (REPLACED): 3위 지역 → 최고 소비 고객 → 해당 지역 소비 %
            (f"주문 건수 3위 지역('{third_region}')의 최고 소비 고객이 해당 지역에서 차지하는 매출 비율은 몇 %입니까? (소수점 버림)",
             top_cust_third_pct, 2,
             f"1단계: 지역별 주문 건수: {', '.join(f'{k}({v})' for k,v in reg_count_ranked)}\n"
             f"2단계: 3위 = '{third_region}'\n"
             f"3단계: '{third_region}' 최고 소비 = '{top_cust_in_third}' ({top_cust_third_spend})\n"
             f"4단계: 전체 소비 = {top_cust_total_spend}, 비율 = {top_cust_third_pct}%\n최종 답: {top_cust_third_pct}"),

            # T6 (REPLACED): Gold+Silver vs Bronze+None 주문당 평균 매출 비율
            (f"골드+실버 고객의 주문당 평균 매출과 브론즈+없음 고객의 주문당 평균 매출의 비율은 얼마입니까? (골드실버평균×100/브론즈없음평균, 소수점 버림)",
             gs_bn_ratio, 2,
             f"1단계: 골드+실버 주문: {len(gs_orders)}건, 매출 = {gs_rev}, 평균 = {int(gs_avg)}\n"
             f"2단계: 브론즈+없음 주문: {len(bn_orders)}건, 매출 = {bn_rev}, 평균 = {int(bn_avg)}\n"
             f"3단계: 비율 = {gs_bn_ratio}\n최종 답: {gs_bn_ratio}"),
        ]

    question, answer, solution = _weighted_choice(
        rng, _calibrate_template_weights(config.difficulty, question_templates)
    )

    return {
        "type": ProblemType.MULTI_CONDITION.value,
        "difficulty": config.difficulty,
        "tables": _make_tables_dict(pt, st, ct),
        "question": question,
        "answer": answer,
        "answer_type": "number" if isinstance(answer, (int, float)) else "text",
        "solution": solution,
    }


# ============================================================
# 메인 생성 함수
# ============================================================

PROBLEM_GENERATORS = {
    ProblemType.LOOKUP_QUERY.value: generate_lookup_problem,
    ProblemType.CONDITIONAL_AGGREGATION.value: generate_conditional_aggregation_problem,
    ProblemType.ARRAY_COMPUTATION.value: generate_array_computation_problem,
    ProblemType.MULTI_CONDITION.value: generate_multi_condition_problem,
}


def _generation_difficulty_for_target(label_difficulty: str, rng: random.Random) -> str:
    """데이터셋 라벨을 보정된 생성 난이도 혼합으로 매핑."""
    roll = rng.random()
    if label_difficulty == "easy":
        return "medium" if roll < 0.82 else "easy"
    if label_difficulty == "medium":
        return "hard" if roll < 0.50 else "medium"
    return label_difficulty


def generate_puzzle(
    difficulty: str = "medium",
    problem_type: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    단일 퍼즐 생성

    Args:
        difficulty: 난이도 ("easy", "medium", "hard")
        problem_type: 문제 유형 (None이면 랜덤)
        seed: 난수 시드

    Returns:
        퍼즐 딕셔너리
    """
    if seed is None:
        seed = random.randint(1, 1000000)

    requested_difficulty = difficulty
    # Reject puzzles whose final answer is ambiguous (e.g. a membership lookup that
    # resolves to the "없음" tier, which is indistinguishable from "no answer").
    # Retry with a bumped seed; ids are reassigned downstream so this is safe.
    base_seed = seed
    for attempt in range(50):
        seed = base_seed + attempt * 7919
        rng = random.Random(seed)
        generation_difficulty = _generation_difficulty_for_target(requested_difficulty, rng)
        config = ArrayFormulaConfig(difficulty=generation_difficulty, seed=seed)

        ptype = problem_type if problem_type is not None else rng.choice(list(PROBLEM_GENERATORS.keys()))
        generator = PROBLEM_GENERATORS[ptype]
        puzzle = generator(config, rng)
        if str(puzzle.get("answer")).strip() not in AMBIGUOUS_ANSWERS:
            break

    problem_type = ptype
    puzzle["generation_difficulty"] = generation_difficulty
    puzzle["difficulty"] = requested_difficulty

    # ID 생성
    puzzle_hash = hashlib.md5(json.dumps(puzzle, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    puzzle["id"] = f"af_ko_{requested_difficulty}_{problem_type}_{puzzle_hash}"
    puzzle["seed"] = seed

    return puzzle


def generate_dataset(
    num_per_difficulty: int = 100,
    seed: int = 2025
) -> List[Dict[str, Any]]:
    """
    난이도별 데이터셋 생성

    Args:
        num_per_difficulty: 난이도별 퍼즐 수
        seed: 기본 시드

    Returns:
        퍼즐 리스트
    """
    puzzles = []
    difficulties = ["easy", "medium", "hard"]
    problem_types = list(PROBLEM_GENERATORS.keys())

    puzzle_seed = seed
    for difficulty in difficulties:
        per_ptype = num_per_difficulty // len(problem_types)
        remainder = num_per_difficulty % len(problem_types)
        diff_idx = 0
        for j, ptype in enumerate(problem_types):
            count = per_ptype + (1 if j < remainder else 0)
            for _ in range(count):
                puzzle = generate_puzzle(
                    difficulty=difficulty,
                    problem_type=ptype,
                    seed=puzzle_seed
                )
                puzzle["id"] = f"array_formula_ko_{difficulty}_{diff_idx:04d}"
                puzzles.append(puzzle)
                puzzle_seed += 1
                diff_idx += 1

    return puzzles


def format_table_for_prompt(table_name: str, table_data: Dict) -> str:
    """테이블을 프롬프트 문자열로 포맷"""
    columns = table_data["columns"]
    data = table_data["data"]

    lines = [f"[{table_name} 테이블]"]
    header = " | ".join(str(col) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))

    for row in data:
        row_str = " | ".join(str(row.get(col, "")) for col in columns)
        lines.append(row_str)

    return "\n".join(lines)


def puzzle_to_prompt(puzzle: Dict[str, Any]) -> str:
    """퍼즐을 LLM 프롬프트로 변환"""
    prompt_parts = []

    prompt_parts.append("다음은 스프레드시트 데이터입니다.\n")

    for table_name, table_data in puzzle["tables"].items():
        prompt_parts.append(format_table_for_prompt(table_name, table_data))
        prompt_parts.append("")

    prompt_parts.append(f"질문: {puzzle['question']}")

    if puzzle.get("answer_type") == "number":
        prompt_parts.append("\n숫자만 답하세요. (단위 없이)")
    else:
        prompt_parts.append("\n정확한 값만 답하세요.")

    return "\n".join(prompt_parts)


_SOLUTION_TYPE_LABELS_KO = {
    ProblemType.LOOKUP_QUERY.value: "조회형(INDEX/MATCH·VLOOKUP류)",
    ProblemType.CONDITIONAL_AGGREGATION.value: "조건부 집계(SUMIF/COUNTIF류)",
    ProblemType.ARRAY_COMPUTATION.value: "배열 연산(SUMPRODUCT류)",
    ProblemType.MULTI_CONDITION.value: "다중 조건(SUMIFS/MAXIFS류)",
}

_SFT_TYPE_REASONING_NUDGE_KO = {
    ProblemType.LOOKUP_QUERY.value: (
        "어떤 표(상품/주문/고객)을 어떤 키(상품명, 고객번호)로 잇는지 먼저 잡고, "
        "집계·필터 후 ‘1위/2위/제외/구간’ 같은 **순위·슬라이스**가 어디에 붙는지 읽는다."
    ),
    ProblemType.CONDITIONAL_AGGREGATION.value: (
        "질문이 요구한 **조건열**(지역, 분기, 등급, 할인율 …)에 맞는 주문(또는 상품) 행만 골라 "
        "COUNT/SUM/평균을 만든 뒤, **버림·반올림**이 문장에 있으면 마지막에만 적용한다."
    ),
    ProblemType.ARRAY_COMPUTATION.value: (
        "행·열이 맞닿는 곳(주문↔상품)에서 ‘수량×가격’ 같은 **원소곱/합**을 쌓고, "
        "최댓값·최솟값·한 행(상품)만 같은 **축 정렬**이 나오는지 먼저 짚는다."
    ),
    ProblemType.MULTI_CONDITION.value: (
        "AND/OR로 엮인 조건(지역+등급+기간+카테고리 …)을 **하나씩** 줄이며 부분집합을 잡고, "
        "필터가 겹칠 때는 “모든 주문/해당 주문만”이 어디에 해당하는지 구분해 집계한다."
    ),
}

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)

_SFT_PIPELINE_HINT_KO = {
    ProblemType.LOOKUP_QUERY.value: "필터 매칭 → 해당 행에서 원하는 열 추출",
    ProblemType.CONDITIONAL_AGGREGATION.value: "WHERE(조건) → 단일 열 집계(SUM/COUNT/AVG)",
    ProblemType.ARRAY_COMPUTATION.value: "조인(상품↔주문) → 원소곱/합(SUMPRODUCT)",
    ProblemType.MULTI_CONDITION.value: "다중 WHERE → 집계·정렬·랭크",
}


def _truncate_for_solution_prompt(text: str, max_len: int = 400) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


_STEP_PREFIX_KO = re.compile(r"^\s*(?:\d+\s*단계\s*[:：-]|Step\s*\d+\s*[:：-])\s*")
_FINAL_PREFIX_KO = re.compile(r"^\s*(?:최종\s*답|Final\s*answer)\s*[:：-]\s*", re.IGNORECASE)


def _worked_body_lines(solution: str) -> list:
    """생성기 'N단계: …' 줄을 [SEG n]으로 재번호; '최종 답:' 줄은 제거."""
    s = (solution or "").strip()
    if not s:
        return [
            "    [SEG 1] (이 자리엔 ‘1단계: …, 2단계: …’ 식 **중간 집계**가 온다. "
            "빈 경우엔 질문·유형에 맞춰 표에서 먼저 뽑을 행과 조인 키를 쓰면 된다.)"
        ]
    out, seg = [], 1
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _FINAL_PREFIX_KO.match(line):
            continue
        body = _STEP_PREFIX_KO.sub("", line)
        out.append(f"    [SEG {seg}] {body}")
        seg += 1
    if not out:
        out.append(
            "    [SEG 1] (원본 풀이에 단계 라벨이 없어, 질문→필터→집계 순서로 1~N 단계로 풀어 써라.)"
        )
    return out


def _solution_for_guided_distillation(
    solution: str,
    answer: Any,
    problem_type: str,
    difficulty: str,
    answer_type: str,
    question: str = "",
) -> str:
    """SFT: 해설이 아니라 추론 궤적을 강조."""
    type_label = _SOLUTION_TYPE_LABELS_KO.get(problem_type, problem_type)
    fmt = "숫자만(단위 없음)" if answer_type == "number" else "텍스트(문제와 동일 표기)"
    nudge = _SFT_TYPE_REASONING_NUDGE_KO.get(
        problem_type,
        "질문에서 **집계 단위(고객/주문/지역/상품)**를 먼저 정한 뒤, "
        "조인·필터·집계 순서로 풀어 쓴다(문제마다 중간 수는 다름).",
    )
    q_line = _truncate_for_solution_prompt(question) if question else (
        "(원문 질문은 상단 퍼즐 ‘질문:’에 있다.)"
    )
    lines = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 문제 유형: {type_label}",
        f"  - 난이도: {difficulty}",
        f"  - 정답 출력 형식: {fmt}",
        "  - (읽는 순서) " + nudge,
        "  - 최종 수치/문자는 [STEP3]에만 ‘검산용’으로 둔다. "
        "먼저 [STEP2]의 **단계 로그(중간값)** 를 따라갈 것.",
        "[STEP 1] 주어진 조건",
        f"  - **이번 질문(원문)**: {q_line}",
        "  - 데이터 스키마(질문·표에 나온 열과 같다):",
        "      상품: id, 상품명, 카테고리, 가격, 재고, 할인율",
        "      주문: 주문번호, 상품명, 지역, 수량, 분기, 고객번호",
        "      고객: 고객번호, 이름, 등급, 가입연도, 지역",
        "[STEP 2] 풀이 전개 (생성기가 쓴 **단계별 계산·중간 집계**; 문제마다 내용이 달라짐)",
    ]
    worked = _worked_body_lines(solution)
    pipeline = _SFT_PIPELINE_HINT_KO.get(
        problem_type,
        "조인 → 필터 → 집계/랭크",
    )
    lines.append(
        f"  · 요약: {type_label} · 파이프라인: {pipeline} · SEG {len(worked)}개")
    lines.append(
        "  · 머릿속으로: (1) 어떤 키로 조인 (2) 어떤 WHERE "
        "(3) 어떤 AGG/랭크 인지 — 그 다음이 아래 로그")
    lines.extend(worked)
    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답: {answer}",
        "  - 점검: 문장에 적힌 버림·반올림·할인가(할인이 가격/재고/매출 중 어디에 쓰였는지)를 "
        "일관되게 썼는지, 상품명·고객번호 등 **조인 키**가 질문 표와 맞는지 끝에서 다시 맞춘다.",
    ])
    return "\n".join(lines)


def save_dataset(
    puzzles: List[Dict],
    base_dir: str = "./data"
):
    """Save dataset as data/{task}_{tier}.jsonl (release layout)."""
    processed_puzzles = []
    for puzzle in puzzles:
        question = puzzle_to_prompt(puzzle)

        processed = {
            "id": puzzle["id"],
            "question": question,
            # 데이터셋의 answer 컬럼은 타입을 하나로 고정한다(문자열).
            # 숫자 정답도 문자열로 저장해야 CSV/HuggingFace 로더가 앞자리 0을
            # 잘라내거나 파일마다 다른 타입으로 추론하는 일이 없다.
            "answer": str(puzzle["answer"]),
            "solution": _solution_for_guided_distillation(
                puzzle.get("solution", ""),
                puzzle["answer"],
                puzzle["type"],
                puzzle["difficulty"],
                puzzle.get("answer_type", "number"),
                puzzle.get("question", ""),
            ),
            "difficulty": puzzle["difficulty"],
        }
        processed_puzzles.append(processed)

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("array_formula_ko", processed_puzzles)

    stats = {}
    for puzzle in puzzles:
        key = f"{puzzle['difficulty']}_{puzzle['type']}"
        stats[key] = stats.get(key, 0) + 1

    print("\nDataset Statistics:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")

    return None


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Array Formula Puzzle Generator (Korean)")
    parser.add_argument("--num", type=int, default=200, help="Number of puzzles per difficulty level")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output", type=str, default="./data", help="Output base directory")
    parser.add_argument("--demo", action="store_true", help="Print demo puzzles")

    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("Array Formula Puzzle Demo (Korean)")
        print("=" * 60)

        for ptype in PROBLEM_GENERATORS.keys():
            for difficulty in ["easy", "medium", "hard"]:
                puzzle = generate_puzzle(
                    difficulty=difficulty,
                    problem_type=ptype,
                    seed=42
                )
                print(f"\n[{ptype} - {difficulty}]")
                print("-" * 40)
                print(puzzle_to_prompt(puzzle))
                print(f"\n답: {puzzle['answer']}")
                if puzzle.get("solution"):
                    print(f"풀이: {puzzle['solution']}")
                print("=" * 60)
                break
    else:
        puzzles = generate_dataset(
            num_per_difficulty=args.num,
            seed=args.seed
        )
        save_dataset(puzzles, args.output)

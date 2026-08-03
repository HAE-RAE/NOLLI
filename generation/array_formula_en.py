"""
Array Formula Puzzle Generator
Excel array formula-based logical puzzle generator

Problem Types:
1. lookup_query: INDEX-MATCH, VLOOKUP style data lookup
2. conditional_aggregation: SUMIF, COUNTIF style conditional aggregation
3. array_computation: SUMPRODUCT style array computation
4. multi_condition: SUMIFS, MAXIFS style multi-condition problems

Difficulty Levels (calibrated against gemini-3-flash-preview):
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
    """Problem generation configuration"""
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
# Data Generation Utilities
# ============================================================

PRODUCT_NAMES = [
    "Apple", "Pear", "Grape", "Strawberry", "Banana", "Orange", "Watermelon", "Melon", "Peach", "Plum",
    "Milk", "Cheese", "Yogurt", "Butter", "IceCream", "Tofu", "Egg", "Ham", "Sausage", "Bacon",
    "Bread", "Rice", "Ramen", "Pasta", "Cereal", "Cookie", "Chocolate", "Candy", "Jelly", "Gum",
    "Cola", "Sprite", "Juice", "Coffee", "GreenTea", "Water", "Beer", "Soju", "Wine", "Makgeolli",
    "Kiwi", "Mango", "Cherry", "Lemon", "Lime", "Coconut", "Almond", "Walnut", "Peanut", "Cashew",
    "Tuna", "Salmon", "Shrimp", "Crab", "Squid", "Seaweed", "Onion", "Garlic", "Pepper", "Tomato",
    "Avocado", "Blueberry", "Raspberry", "Pomegranate", "Fig",
    "Mackerel", "Oyster", "Clam", "Anchovy", "Lobster",
    "Lettuce", "Spinach", "Carrot", "Cucumber", "Mushroom",
    "Dumpling", "FishCake", "Jerky", "Popcorn", "Chips",
    "Brandy", "Whiskey", "Milkshake", "Smoothie", "Kombucha",
]

CATEGORIES = ["Fruit", "Dairy", "Meat", "Grain", "Beverage", "Vegetable", "Seafood", "Processed"]
REGIONS = ["Seoul", "Busan", "Daegu", "Incheon", "Gwangju", "Daejeon", "Ulsan", "Sejong"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
MEMBERSHIPS = ["Gold", "Silver", "Bronze", "None"]
# Answers that read as "no answer" and are ungradeable if returned as the final answer.
# "None" is a valid membership tier (a non-member) and may appear in tables, but a
# membership-lookup question must never RESOLVE to it.
AMBIGUOUS_ANSWERS = {"None", ""}

CUSTOMER_NAMES = [
    "Kim Minsu", "Lee Younghee", "Park Cheolsu", "Choi Jiyoung", "Jung Donghyun",
    "Kang Sujin", "Cho Hyunwoo", "Yoon Mina", "Jang Junho", "Lim Seoyeon",
    "Han Jihoon", "Seo Yuna", "Son Taewoo", "Kwon Sooyeon", "Shin Woojin",
    "Oh Haeun", "Baek Sunwoo", "Hong Yerin", "Yoo Jaemin", "Moon Dain"
]


def generate_product_table(
    num_rows: int,
    num_categories: int,
    seed: int,
    difficulty: str = "easy"
) -> List[Dict[str, Any]]:
    """Generate product table"""
    rng = random.Random(seed)

    categories = rng.sample(CATEGORIES, min(num_categories, len(CATEGORIES)))
    products = rng.sample(PRODUCT_NAMES, num_rows)

    # Ensure uneven distribution of categories (important for avg-of-avg trap)
    table = []
    for i, product in enumerate(products):
        row = {
            "id": i + 1,
            "product": product,
            "category": rng.choice(categories),
            "price": rng.randint(5, 50) * 100,
            "stock": rng.randint(10, 200),
            "discount": rng.choice([0, 5, 10, 15, 20]),
        }
        if difficulty in ("medium", "hard"):
            row.update({
                "supplier": f"S-{rng.randint(1, 12):02d}",
                "warehouse": rng.choice(["North", "South", "East", "West"]),
            })
        if difficulty == "hard":
            row.update({
                "tax_rate": rng.choice([0, 3, 5, 8, 10]),
                "rating": rng.randint(1, 5),
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
    """Generate sales table, optionally with customer_id"""
    rng = random.Random(seed + 1000)

    regions = rng.sample(REGIONS, min(num_regions, len(REGIONS)))
    products = [p["product"] for p in product_table]

    table = []
    for i in range(num_orders):
        row = {
            "order_id": f"ORD-{i+1:03d}",
            "product": rng.choice(products),
            "region": rng.choice(regions),
            "quantity": rng.randint(1, 50),
            "quarter": rng.choice(QUARTERS),
        }
        if customer_table is not None:
            row["customer_id"] = rng.choice(customer_table)["customer_id"]
        if difficulty in ("medium", "hard"):
            row.update({
                "channel": rng.choice(["Online", "Store", "Partner", "Phone"]),
                "priority": rng.choice(["Low", "Normal", "High"]),
            })
        if difficulty == "hard":
            row.update({
                "shipping_fee": rng.randint(0, 20) * 100,
                "promo_rate": rng.choice([0, 5, 10, 15]),
            })
        table.append(row)

    return table


def generate_customer_table(
    num_customers: int,
    num_regions: int,
    seed: int,
    difficulty: str = "easy"
) -> List[Dict[str, Any]]:
    """Generate customer table"""
    rng = random.Random(seed + 3000)

    regions = rng.sample(REGIONS, min(num_regions, len(REGIONS)))
    names = rng.sample(CUSTOMER_NAMES, min(num_customers, len(CUSTOMER_NAMES)))

    table = []
    for i, name in enumerate(names):
        row = {
            "customer_id": f"CUST-{i+1:03d}",
            "name": name,
            "membership": rng.choice(MEMBERSHIPS),
            "join_year": rng.randint(2018, 2024),
            "region": rng.choice(regions),
        }
        if difficulty in ("medium", "hard"):
            row["segment"] = rng.choice(["Retail", "Corporate", "Education", "Public"])
        if difficulty == "hard":
            row.update({
                "age_group": rng.choice(["20s", "30s", "40s", "50s"]),
                "loyalty_points": rng.randint(0, 5000),
            })
        table.append(row)

    return table


# ============================================================
# Helper utilities
# ============================================================

def _group_sum(items, key_fn, val_fn):
    """Group items and sum values. Returns dict {key: sum}."""
    groups = {}
    for item in items:
        k = key_fn(item)
        groups[k] = groups.get(k, 0) + val_fn(item)
    return groups


def _rank_groups(group_dict, reverse=True):
    """Return sorted list of (key, value) by value. reverse=True means descending."""
    return sorted(group_dict.items(), key=lambda x: x[1], reverse=reverse)


def _group_count(items, key_fn):
    """Count items per group. Returns dict {key: count}."""
    groups = {}
    for item in items:
        k = key_fn(item)
        groups[k] = groups.get(k, 0) + 1
    return groups


def _group_avg(items, key_fn, val_fn):
    """Group-level averages. Returns dict {key: avg}."""
    sums = {}
    counts = {}
    for item in items:
        k = key_fn(item)
        sums[k] = sums.get(k, 0) + val_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def _median(values):
    """Median calculation with int truncation."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return int(s[n // 2])
    return int((s[n // 2 - 1] + s[n // 2]) / 2)


def _group_distinct_count(items, key_fn, val_fn):
    """Count distinct values per group. Returns dict {key: distinct_count}."""
    groups = {}
    for item in items:
        k = key_fn(item)
        if k not in groups:
            groups[k] = set()
        groups[k].add(val_fn(item))
    return {k: len(v) for k, v in groups.items()}


def _std_dev(values):
    """Population standard deviation."""
    if len(values) == 0:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _weighted_choice(rng, templates):
    """Choose from templates with optional weights. Each template is (question, answer, weight, solution)."""
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
    """Shift sampling toward multi-step numeric templates as difficulty increases."""
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
    """Build standard 3-table dict for return value."""
    def columns(base_columns, table):
        extra_columns = []
        if table:
            extra_columns = [col for col in table[0].keys() if col not in base_columns]
        return base_columns + extra_columns

    return {
        "Products": {
            "columns": columns(["id", "product", "category", "price", "stock", "discount"], product_table),
            "data": product_table
        },
        "Orders": {
            "columns": columns(["order_id", "product", "region", "quantity", "quarter", "customer_id"], sales_table),
            "data": sales_table
        },
        "Customers": {
            "columns": columns(["customer_id", "name", "membership", "join_year", "region"], customer_table),
            "data": customer_table
        }
    }


# ============================================================
# Data generation shared across generators
# ============================================================

_ORDER_COUNTS = {"easy": (35, 50), "medium": (120, 170), "hard": (220, 300)}
_CUSTOMER_COUNTS = {"easy": (10, 14), "medium": (18, 20), "hard": (20, 20)}


def _generate_all_tables(config, rng):
    """Generate all 3 tables + helper maps."""
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

    product_map = {p["product"]: p for p in product_table}
    customer_map = {c["customer_id"]: c for c in customer_table}

    return product_table, sales_table, customer_table, product_map, customer_map


# ============================================================
# Problem Type Generators
# ============================================================

def generate_lookup_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """Generate LOOKUP problem"""
    pt, st, ct, pm, cm = _generate_all_tables(config, rng)

    categories = list(set(p["category"] for p in pt))
    regions_in_orders = list(set(s["region"] for s in st))

    if config.difficulty == "easy":
        # T1: discount-price order value
        target_order = rng.choice(st)
        t1_price = pm[target_order["product"]]["price"]
        t1_disc = pm[target_order["product"]]["discount"]
        t1_disc_price = int(t1_price * (100 - t1_disc) / 100)
        t1_disc_value = t1_disc_price * target_order["quantity"]

        # T2: rank-2 by total qty → category
        product_qty = _group_sum(st, lambda s: s["product"], lambda s: s["quantity"])
        qty_ranked = _rank_groups(product_qty)
        rank2_name = qty_ranked[1][0] if len(qty_ranked) >= 2 else qty_ranked[0][0]
        rank2_cat = pm[rank2_name]["category"]

        # T3: category order count
        tgt_cat = rng.choice(categories)
        cat_prods = set(p["product"] for p in pt if p["category"] == tgt_cat)
        cat_order_count = len([s for s in st if s["product"] in cat_prods])

        # T4: category average order quantity (truncate)
        cat_orders = [s for s in st if s["product"] in cat_prods]
        cat_avg_qty = int(sum(s["quantity"] for s in cat_orders) / len(cat_orders)) if cat_orders else 0

        # T5: order → customer → region chain
        chain_order = rng.choice(st)
        chain_cust = cm[chain_order["customer_id"]]

        # T6: total discounted revenue for a specific category
        cat_disc_rev = sum(
            int(pm[s["product"]]["price"] * (100 - pm[s["product"]]["discount"]) / 100) * s["quantity"]
            for s in cat_orders
        )

        # T7: product with max stock in top-3 ordered categories → discount rate
        cat_qty = _group_sum(st, lambda s: pm[s["product"]]["category"], lambda s: s["quantity"])
        cat_qty_ranked = _rank_groups(cat_qty)
        top3_cats = set(c for c, _ in cat_qty_ranked[:3])
        top3_cat_prods = [p for p in pt if p["category"] in top3_cats]
        if top3_cat_prods:
            max_stock_prod = max(top3_cat_prods, key=lambda p: p["stock"])
            t7_disc = max_stock_prod["discount"]
        else:
            max_stock_prod = pt[0]
            t7_disc = pt[0]["discount"]

        # T8: region with highest order count → total discounted revenue
        reg_count = _group_count(st, lambda s: s["region"])
        reg_count_ranked = _rank_groups(reg_count)
        top_count_reg = reg_count_ranked[0][0] if reg_count_ranked else regions_in_orders[0]
        top_reg_orders = [s for s in st if s["region"] == top_count_reg]
        top_reg_disc_rev = sum(
            int(pm[s["product"]]["price"] * (100 - pm[s["product"]]["discount"]) / 100) * s["quantity"]
            for s in top_reg_orders
        )

        question_templates = [
            (f"What is the discounted value of order '{target_order['order_id']}'? (discounted_price = price × (100 - discount) / 100, truncate decimals; value = discounted_price × quantity)",
             t1_disc_value, 1,
             f"Step 1: '{target_order['order_id']}' → product = '{target_order['product']}'\n"
             f"Step 2: price = {t1_price}, discount = {t1_disc}%\n"
             f"Step 3: discounted_price = {t1_price} × (100-{t1_disc})/100 = {t1_disc_price}\n"
             f"Step 4: value = {t1_disc_price} × {target_order['quantity']} = {t1_disc_value}\n"
             f"Final answer: {t1_disc_value}"),

            (f"What is the category of the product ranked 2nd in total order quantity?",
             rank2_cat, 1,
             f"Step 1: Group orders by product, sum quantities\n"
             f"Step 2: {', '.join(f'{k}({v})' for k,v in qty_ranked[:5])}\n"
             f"Step 3: 2nd = '{rank2_name}'\n"
             f"Step 4: category = '{rank2_cat}'\nFinal answer: {rank2_cat}"),

            (f"How many orders are there for '{tgt_cat}' category products?",
             cat_order_count, 1,
             f"Step 1: '{tgt_cat}' products: {', '.join(list(cat_prods)[:5])}\n"
             f"Step 2: Filter orders\n"
             f"Step 3: count = {cat_order_count}\nFinal answer: {cat_order_count}"),

            (f"What is the average order quantity for '{tgt_cat}' category products? (truncate decimals)",
             cat_avg_qty, 1,
             f"Step 1: '{tgt_cat}' products: {', '.join(list(cat_prods)[:5])}\n"
             f"Step 2: Filter orders: {len(cat_orders)} orders\n"
             f"Step 3: average quantity = {cat_avg_qty}\nFinal answer: {cat_avg_qty}"),

            (f"What is the region of the customer who placed order '{chain_order['order_id']}'?",
             chain_cust["region"], 1,
             f"Step 1: '{chain_order['order_id']}' → customer_id = '{chain_order['customer_id']}'\n"
             f"Step 2: Look up in Customers table\n"
             f"Step 3: region = '{chain_cust['region']}'\n"
             f"Final answer: {chain_cust['region']}"),

            (f"What is the total discounted revenue for '{tgt_cat}' category products? (For each order: discounted_price = price × (100-discount)/100, truncate decimals, then multiply by quantity. Sum all.)",
             cat_disc_rev, 2,
             f"Step 1: '{tgt_cat}' products: {', '.join(list(cat_prods)[:5])}\n"
             f"Step 2: Filter orders: {len(cat_orders)} orders\n"
             f"Step 3: For each: discounted_price (truncated) × quantity\n"
             f"Step 4: Sum = {cat_disc_rev}\nFinal answer: {cat_disc_rev}"),

            (f"Among the top 3 categories by total order quantity, which product has the highest stock? What is its discount rate?",
             t7_disc, 2,
             f"Step 1: Category order qty: {', '.join(f'{k}({v})' for k,v in cat_qty_ranked[:5])}\n"
             f"Step 2: Top 3 categories: {', '.join(top3_cats)}\n"
             f"Step 3: Products in these categories: {len(top3_cat_prods)}\n"
             f"Step 4: Max stock = '{max_stock_prod['product']}' (stock={max_stock_prod['stock']})\n"
             f"Step 5: Discount = {t7_disc}%\nFinal answer: {t7_disc}"),

            (f"Which region has the most orders? What is the total discounted revenue for that region? (discounted_price = price × (100-discount)/100, truncate per item, then × quantity)",
             top_reg_disc_rev, 2,
             f"Step 1: Orders per region: {', '.join(f'{k}({v})' for k,v in reg_count_ranked[:5])}\n"
             f"Step 2: Top region = '{top_count_reg}'\n"
             f"Step 3: '{top_count_reg}' orders: {len(top_reg_orders)}\n"
             f"Step 4: Discounted revenue = {top_reg_disc_rev}\nFinal answer: {top_reg_disc_rev}"),
        ]

    elif config.difficulty == "medium":
        # T1: revenue rank-2 product's category
        prod_revenue = _group_sum(st, lambda s: s["product"],
                                  lambda s: pm[s["product"]]["price"] * s["quantity"])
        rev_ranked = _rank_groups(prod_revenue)
        rank2_name = rev_ranked[1][0] if len(rev_ranked) >= 2 else rev_ranked[0][0]
        rank2_cat = pm[rank2_name]["category"]

        # T2: top spender in a specific region (3-table join)
        tgt_region = rng.choice(regions_in_orders)
        reg_orders = [s for s in st if s["region"] == tgt_region]
        reg_cust_spend = _group_sum(reg_orders, lambda s: s["customer_id"],
                                     lambda s: pm[s["product"]]["price"] * s["quantity"])
        reg_spend_ranked = _rank_groups(reg_cust_spend)
        reg_top_cid = reg_spend_ranked[0][0] if reg_spend_ranked else ct[0]["customer_id"]
        reg_top_cust_name = cm[reg_top_cid]["name"]

        # T3: top product by qty for a specific membership grade → discount
        tgt_grade = rng.choice(["Gold", "Silver", "Bronze"])
        grade_cids = set(c["customer_id"] for c in ct if c["membership"] == tgt_grade)
        grade_orders = [s for s in st if s["customer_id"] in grade_cids]
        grade_prod_qty = _group_sum(grade_orders, lambda s: s["product"], lambda s: s["quantity"])
        grade_qty_ranked = _rank_groups(grade_prod_qty)
        grade_top_prod = grade_qty_ranked[0][0] if grade_qty_ranked else pt[0]["product"]
        grade_top_disc = pm[grade_top_prod]["discount"]

        # T4: quarter top product → oldest join year among its customers
        tgt_q = rng.choice(QUARTERS)
        q_orders = [s for s in st if s["quarter"] == tgt_q]
        q_prod_rev = _group_sum(q_orders, lambda s: s["product"],
                                lambda s: pm[s["product"]]["price"] * s["quantity"])
        q_rev_ranked = _rank_groups(q_prod_rev)
        q_top_prod = q_rev_ranked[0][0] if q_rev_ranked else pt[0]["product"]
        q_top_orders = [s for s in q_orders if s["product"] == q_top_prod]
        q_top_cids = set(s["customer_id"] for s in q_top_orders)
        earliest_year = min((cm[cid]["join_year"] for cid in q_top_cids), default=2020)

        # T5: category with 2nd-highest distinct products ordered → total revenue
        cat_distinct = _group_distinct_count(st, lambda s: pm[s["product"]]["category"], lambda s: s["product"])
        cat_dist_ranked = _rank_groups(cat_distinct)
        t5_cat = cat_dist_ranked[1][0] if len(cat_dist_ranked) >= 2 else cat_dist_ranked[0][0]
        t5_cat_prods = set(p["product"] for p in pt if p["category"] == t5_cat)
        t5_cat_orders = [s for s in st if s["product"] in t5_cat_prods]
        t5_cat_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in t5_cat_orders)

        # T6: total revenue from orders where qty > avg qty
        all_qtys = [s["quantity"] for s in st]
        avg_qty = sum(all_qtys) / len(all_qtys) if all_qtys else 0
        above_avg_orders = [s for s in st if s["quantity"] > avg_qty]
        above_avg_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in above_avg_orders)

        question_templates = [
            (f"What is the category of the product ranked 2nd in total revenue? (revenue = price × quantity)",
             rank2_cat, 1,
             f"Step 1: Compute revenue per order\n"
             f"Step 2: Group by product, sum revenue\n"
             f"Step 3: {', '.join(f'{k}({v})' for k,v in rev_ranked[:5])}\n"
             f"Step 4: 2nd = '{rank2_name}' → category = '{rank2_cat}'\nFinal answer: {rank2_cat}"),

            (f"Who is the customer that spent the most in '{tgt_region}'? (spending = price × quantity, answer with customer name)",
             reg_top_cust_name, 1,
             f"Step 1: Filter '{tgt_region}' orders: {len(reg_orders)} orders\n"
             f"Step 2: Group by customer, sum spending\n"
             f"Step 3: {', '.join(f'{k}({v})' for k,v in reg_spend_ranked[:5])}\n"
             f"Step 4: Top = '{reg_top_cid}' → '{reg_top_cust_name}'\nFinal answer: {reg_top_cust_name}"),

            (f"What is the discount rate of the product most ordered (by quantity) by '{tgt_grade}' membership customers?",
             grade_top_disc, 1,
             f"Step 1: '{tgt_grade}' customers: {len(grade_cids)}\n"
             f"Step 2: Filter orders: {len(grade_orders)} orders\n"
             f"Step 3: Product qty: {', '.join(f'{k}({v})' for k,v in grade_qty_ranked[:5])}\n"
             f"Step 4: Top = '{grade_top_prod}' → discount = {grade_top_disc}%\nFinal answer: {grade_top_disc}"),

            (f"Among customers who ordered the top-selling product in {tgt_q} (by revenue), what is the earliest join year?",
             earliest_year, 1,
             f"Step 1: {tgt_q} orders: {len(q_orders)}\n"
             f"Step 2: Product revenue: {', '.join(f'{k}({v})' for k,v in q_rev_ranked[:3])}\n"
             f"Step 3: Top = '{q_top_prod}'\n"
             f"Step 4: Customers: {', '.join(q_top_cids)}\n"
             f"Step 5: Earliest join year = {earliest_year}\nFinal answer: {earliest_year}"),

            (f"Which category has the 2nd-highest number of distinct products ordered? What is its total revenue? (revenue = price × quantity)",
             t5_cat_rev, 2,
             f"Step 1: Count distinct products ordered per category: {', '.join(f'{k}({v})' for k,v in cat_dist_ranked[:5])}\n"
             f"Step 2: 2nd = '{t5_cat}'\n"
             f"Step 3: '{t5_cat}' products: {', '.join(list(t5_cat_prods)[:5])}\n"
             f"Step 4: Revenue = {t5_cat_rev}\nFinal answer: {t5_cat_rev}"),

            (f"What is the total revenue from orders where the quantity exceeds the average order quantity ({int(avg_qty)})? (revenue = price × quantity)",
             above_avg_rev, 2,
             f"Step 1: Average order quantity = {int(avg_qty)}\n"
             f"Step 2: Filter orders with qty > {int(avg_qty)}: {len(above_avg_orders)} orders\n"
             f"Step 3: Sum revenue = {above_avg_rev}\nFinal answer: {above_avg_rev}"),
        ]

    else:  # hard
        # T1 (Pattern A): above-avg spenders → rank by distinct categories → 2nd → region
        cust_spend = _group_sum(st, lambda s: s["customer_id"],
                                lambda s: pm[s["product"]]["price"] * s["quantity"])
        avg_cust_spend = sum(cust_spend.values()) / len(cust_spend) if cust_spend else 0
        above_avg_cids = {cid for cid, v in cust_spend.items() if v > avg_cust_spend}
        above_avg_distinct = _group_distinct_count(
            [s for s in st if s["customer_id"] in above_avg_cids],
            lambda s: s["customer_id"],
            lambda s: pm[s["product"]]["category"]
        )
        above_avg_dist_ranked = _rank_groups(above_avg_distinct)
        t1_cid = above_avg_dist_ranked[1][0] if len(above_avg_dist_ranked) >= 2 else above_avg_dist_ranked[0][0]
        t1_region = cm[t1_cid]["region"]

        # T2: among customers with 3+ orders, rank-2 by avg order value → membership
        cust_order_counts = _group_count(st, lambda s: s["customer_id"])
        cust_3plus = {cid for cid, cnt in cust_order_counts.items() if cnt >= 3}
        if len(cust_3plus) < 2:
            cust_3plus = {cid for cid, cnt in cust_order_counts.items() if cnt >= 2}
        if len(cust_3plus) < 2:
            cust_3plus = set(cid for cid, _ in sorted(cust_order_counts.items(), key=lambda x: x[1], reverse=True)[:2])
        cust_avg_val = {}
        for cid in cust_3plus:
            cid_orders = [s for s in st if s["customer_id"] == cid]
            total_val = sum(pm[s["product"]]["price"] * s["quantity"] for s in cid_orders)
            cust_avg_val[cid] = total_val / len(cid_orders) if cid_orders else 0
        avg_val_ranked = _rank_groups(cust_avg_val)
        t2_cid = avg_val_ranked[1][0] if len(avg_val_ranked) >= 2 else avg_val_ranked[0][0]
        t2_grade = cm[t2_cid]["membership"]

        # T3: bottom-3 categories by revenue → max avg qty product → top orderer → membership
        cat_rev = _group_sum(st, lambda s: pm[s["product"]]["category"],
                             lambda s: pm[s["product"]]["price"] * s["quantity"])
        cat_rev_ranked = _rank_groups(cat_rev)
        bottom3_cats = set(c for c, _ in cat_rev_ranked[-3:]) if len(cat_rev_ranked) >= 3 else set(c for c, _ in cat_rev_ranked)
        bottom3_prod_avg_qty = {}
        for p_name in set(p["product"] for p in pt if p["category"] in bottom3_cats):
            p_orders = [s for s in st if s["product"] == p_name]
            if p_orders:
                bottom3_prod_avg_qty[p_name] = sum(s["quantity"] for s in p_orders) / len(p_orders)
        if bottom3_prod_avg_qty:
            t3_prod = max(bottom3_prod_avg_qty, key=bottom3_prod_avg_qty.get)
        else:
            t3_prod = pt[0]["product"]
        t3_orders = [s for s in st if s["product"] == t3_prod]
        t3_cust_qty = _group_sum(t3_orders, lambda s: s["customer_id"], lambda s: s["quantity"])
        t3_cust_ranked = _rank_groups(t3_cust_qty)
        t3_top_cid = t3_cust_ranked[0][0] if t3_cust_ranked else ct[0]["customer_id"]
        t3_membership = cm[t3_top_cid]["membership"]

        # T4: join<2021 → top per-order avg spender → most ordered category → avg stock of that category
        old_cids = set(c["customer_id"] for c in ct if c["join_year"] < 2021)
        old_orders = [s for s in st if s["customer_id"] in old_cids]
        old_cust_avg = {}
        old_cust_counts = _group_count(old_orders, lambda s: s["customer_id"])
        old_cust_rev = _group_sum(old_orders, lambda s: s["customer_id"],
                                   lambda s: pm[s["product"]]["price"] * s["quantity"])
        for cid in old_cust_rev:
            old_cust_avg[cid] = old_cust_rev[cid] / old_cust_counts.get(cid, 1)
        old_avg_ranked = _rank_groups(old_cust_avg)
        t4_cid = old_avg_ranked[0][0] if old_avg_ranked else ct[0]["customer_id"]
        t4_orders = [s for s in st if s["customer_id"] == t4_cid]
        t4_cat_qty = _group_sum(t4_orders, lambda s: pm[s["product"]]["category"], lambda s: s["quantity"])
        t4_cat_ranked = _rank_groups(t4_cat_qty)
        t4_top_cat = t4_cat_ranked[0][0] if t4_cat_ranked else categories[0]
        t4_cat_prods = [p for p in pt if p["category"] == t4_top_cat]
        t4_avg_stock = int(sum(p["stock"] for p in t4_cat_prods) / len(t4_cat_prods)) if t4_cat_prods else 0

        # T5: post-2021 customers → top spender → most ordered category
        recent_cids = set(c["customer_id"] for c in ct if c["join_year"] >= 2021)
        recent_orders = [s for s in st if s["customer_id"] in recent_cids]
        recent_spend = _group_sum(recent_orders, lambda s: s["customer_id"],
                                   lambda s: pm[s["product"]]["price"] * s["quantity"])
        recent_ranked = _rank_groups(recent_spend)
        t5_cid = recent_ranked[0][0] if recent_ranked else ct[0]["customer_id"]
        t5_orders = [s for s in st if s["customer_id"] == t5_cid]
        t5_cat_qty = _group_sum(t5_orders, lambda s: pm[s["product"]]["category"], lambda s: s["quantity"])
        t5_cat_ranked = _rank_groups(t5_cat_qty)
        t5_top_cat = t5_cat_ranked[0][0] if t5_cat_ranked else categories[0]

        question_templates = [
            (f"Among customers whose total spending exceeds the average per-customer spending, rank them by the number of distinct categories they ordered from. What is the region of the customer ranked 2nd?",
             t1_region, 1,
             f"Step 1: Per-customer spending\n"
             f"Step 2: Average = {int(avg_cust_spend)}\n"
             f"Step 3: Above average: {len(above_avg_cids)} customers\n"
             f"Step 4: Distinct categories per above-avg customer: {', '.join(f'{k}({v})' for k,v in above_avg_dist_ranked[:5])}\n"
             f"Step 5: 2nd = '{t1_cid}' → region = '{t1_region}'\nFinal answer: {t1_region}"),

            (f"Among customers with 3 or more orders, what is the membership of the customer ranked 2nd in average order value (price × quantity / number of orders)?",
             t2_grade, 1,
             f"Step 1: Count orders per customer\n"
             f"Step 2: 3+ orders: {', '.join(cust_3plus)}\n"
             f"Step 3-4: Compute avg order value per customer\n"
             f"Step 5: {', '.join(f'{k}({int(v)})' for k,v in avg_val_ranked[:5])}\n"
             f"Step 6: 2nd = '{t2_cid}' → membership = '{t2_grade}'\nFinal answer: {t2_grade}"),

            (f"Among the bottom 3 categories by total revenue, find the product with the highest average order quantity. Who ordered that product the most (by qty)? What is their membership?",
             t3_membership, 1,
             f"Step 1: Category revenue: {', '.join(f'{k}({v})' for k,v in cat_rev_ranked)}\n"
             f"Step 2: Bottom 3: {', '.join(bottom3_cats)}\n"
             f"Step 3: Avg qty per product in bottom-3 cats\n"
             f"Step 4: Max avg qty product = '{t3_prod}'\n"
             f"Step 5: Top orderer: '{t3_top_cid}' → membership = '{t3_membership}'\nFinal answer: {t3_membership}"),

            (f"Among customers who joined before 2021, find the one with the highest average order value (total revenue / number of orders). What is the average stock of products in their most-ordered category? (truncate decimals)",
             t4_avg_stock, 1,
             f"Step 1: Pre-2021 customers: {len(old_cids)}\n"
             f"Step 2: Avg order value: {', '.join(f'{k}({int(v)})' for k,v in old_avg_ranked[:5])}\n"
             f"Step 3: Top = '{t4_cid}'\n"
             f"Step 4: Most ordered category = '{t4_top_cat}'\n"
             f"Step 5: '{t4_top_cat}' products: {len(t4_cat_prods)}, avg stock = {t4_avg_stock}\nFinal answer: {t4_avg_stock}"),

            (f"Among customers who joined in 2021 or later, what is the most ordered category (by quantity) for the top spender? (answer with category name)",
             t5_top_cat, 1,
             f"Step 1: Post-2021 customers: {len(recent_cids)}\n"
             f"Step 2: Spending: {', '.join(f'{k}({int(v)})' for k,v in recent_ranked[:5])}\n"
             f"Step 3: Top = '{t5_cid}'\n"
             f"Step 4: '{t5_cid}' category qty: {', '.join(f'{k}({v})' for k,v in t5_cat_ranked)}\n"
             f"Step 5: Top category = '{t5_top_cat}'\nFinal answer: {t5_top_cat}"),
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
    """Generate conditional aggregation problem"""
    pt, st, ct, pm, cm = _generate_all_tables(config, rng)

    categories = list(set(p["category"] for p in pt))
    regions_in_orders = list(set(s["region"] for s in st))
    tgt_cat = rng.choice(categories)
    cat_products = [p for p in pt if p["category"] == tgt_cat]
    cat_prod_names = set(p["product"] for p in cat_products)

    if config.difficulty == "easy":
        # T1: category revenue
        cat_orders = [s for s in st if s["product"] in cat_prod_names]
        cat_revenue = sum(pm[s["product"]]["price"] * s["quantity"] for s in cat_orders)

        # T2: discount inventory value
        disc_inv = sum(int(p["price"] * (100 - p["discount"]) / 100) * p["stock"] for p in cat_products)

        # T3: category average price
        avg_price = int(sum(p["price"] for p in cat_products) / len(cat_products)) if cat_products else 0

        # T4: quarter total quantity
        tgt_q = rng.choice(QUARTERS)
        q_orders = [s for s in st if s["quarter"] == tgt_q]
        q_total_qty = sum(s["quantity"] for s in q_orders)

        # T5: discount >= 10% order count
        disc_prods = set(p["product"] for p in pt if p["discount"] >= 10)
        disc_order_count = len([s for s in st if s["product"] in disc_prods])

        # T6: revenue from orders where qty > median qty
        all_qtys = sorted(s["quantity"] for s in st)
        med_qty = _median(all_qtys)
        above_med_orders = [s for s in st if s["quantity"] > med_qty]
        above_med_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in above_med_orders)

        # T7: products above avg price → discounted inventory value
        all_avg_price = int(sum(p["price"] for p in pt) / len(pt)) if pt else 0
        above_avg_prods = [p for p in pt if p["price"] > all_avg_price]
        above_avg_disc_inv = sum(int(p["price"] * (100 - p["discount"]) / 100) * p["stock"] for p in above_avg_prods)

        # T8: category with highest avg discount → order count
        cat_avg_disc = _group_avg(pt, lambda p: p["category"], lambda p: p["discount"])
        cat_disc_ranked = _rank_groups(cat_avg_disc)
        top_disc_cat = cat_disc_ranked[0][0] if cat_disc_ranked else categories[0]
        top_disc_cat_prods = set(p["product"] for p in pt if p["category"] == top_disc_cat)
        top_disc_cat_orders = len([s for s in st if s["product"] in top_disc_cat_prods])

        question_templates = [
            (f"What is the total sales revenue for '{tgt_cat}' category products? (revenue = price * quantity, look up price from Products table)",
             cat_revenue, 1,
             f"Step 1: '{tgt_cat}' products: {', '.join(list(cat_prod_names)[:5])}\n"
             f"Step 2: Filter orders: {len(cat_orders)} orders\n"
             f"Step 3: Sum price * qty = {cat_revenue}\nFinal answer: {cat_revenue}"),

            (f"What is the total discounted inventory value for '{tgt_cat}' category? (discounted_price = price * (100 - discount) / 100, truncate decimals, then multiply by stock and sum)",
             disc_inv, 1,
             f"Step 1: '{tgt_cat}' products: {len(cat_products)}\n"
             f"Step 2: For each product: discounted_price = price * (100-discount)/100 (truncate)\n"
             f"Step 3: discounted_price * stock\n"
             f"Step 4: Sum = {disc_inv}\nFinal answer: {disc_inv}"),

            (f"What is the average price of '{tgt_cat}' category products? (truncate decimals)",
             avg_price, 1,
             f"Step 1: '{tgt_cat}' products: {len(cat_products)}\n"
             f"Step 2: Sum prices / count = {avg_price}\nFinal answer: {avg_price}"),

            (f"What is the total quantity ordered in {tgt_q}?",
             q_total_qty, 1,
             f"Step 1: Filter {tgt_q} orders: {len(q_orders)} orders\n"
             f"Step 2: Sum quantities = {q_total_qty}\nFinal answer: {q_total_qty}"),

            (f"How many orders are for products with discount rate 10% or higher?",
             disc_order_count, 1,
             f"Step 1: Products with discount >= 10%: {len(disc_prods)}\n"
             f"Step 2: Filter matching orders\n"
             f"Step 3: Count = {disc_order_count}\nFinal answer: {disc_order_count}"),

            (f"What is the total revenue from orders where quantity exceeds the median order quantity ({med_qty})? (revenue = price × quantity)",
             above_med_rev, 2,
             f"Step 1: Median order quantity = {med_qty}\n"
             f"Step 2: Filter orders with qty > {med_qty}: {len(above_med_orders)} orders\n"
             f"Step 3: Sum revenue = {above_med_rev}\nFinal answer: {above_med_rev}"),

            (f"For products with price above the overall average price ({all_avg_price}), what is their total discounted inventory value? (discounted_price = price × (100-discount)/100, truncate, then × stock)",
             above_avg_disc_inv, 2,
             f"Step 1: Average price = {all_avg_price}\n"
             f"Step 2: Products above avg: {len(above_avg_prods)}\n"
             f"Step 3: For each: discounted_price (truncated) × stock\n"
             f"Step 4: Sum = {above_avg_disc_inv}\nFinal answer: {above_avg_disc_inv}"),

            (f"Which category has the highest average discount rate? How many orders are there for that category's products?",
             top_disc_cat_orders, 2,
             f"Step 1: Avg discount per category: {', '.join(f'{k}({v:.1f})' for k,v in cat_disc_ranked[:5])}\n"
             f"Step 2: Highest = '{top_disc_cat}'\n"
             f"Step 3: '{top_disc_cat}' products: {len(top_disc_cat_prods)}\n"
             f"Step 4: Order count = {top_disc_cat_orders}\nFinal answer: {top_disc_cat_orders}"),
        ]

    elif config.difficulty == "medium":
        # T1: category revenue % of total
        cat_rev = _group_sum(st, lambda s: pm[s["product"]]["category"],
                             lambda s: pm[s["product"]]["price"] * s["quantity"])
        total_rev = sum(cat_rev.values())
        cat_rev_ranked = _rank_groups(cat_rev)
        tgt_cat_rev = cat_rev.get(tgt_cat, 0)
        tgt_cat_pct = int(tgt_cat_rev * 100 / total_rev) if total_rev > 0 else 0

        # T2: customers above average spending
        cust_spend = _group_sum(st, lambda s: s["customer_id"],
                                lambda s: pm[s["product"]]["price"] * s["quantity"])
        avg_spend = sum(cust_spend.values()) / len(cust_spend) if cust_spend else 0
        custs_above = sum(1 for v in cust_spend.values() if v > avg_spend)

        # T3: membership grade revenue %
        tgt_grade = rng.choice(["Gold", "Silver", "Bronze"])
        grade_cids = set(c["customer_id"] for c in ct if c["membership"] == tgt_grade)
        grade_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in st if s["customer_id"] in grade_cids)
        grade_pct = int(grade_rev * 100 / total_rev) if total_rev > 0 else 0

        # T4: median-stock revenue %
        stocks = [p["stock"] for p in pt]
        median_stock = _median(stocks)
        high_stock_prods = set(p["product"] for p in pt if p["stock"] > median_stock)
        high_stock_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in st if s["product"] in high_stock_prods)
        high_stock_pct = int(high_stock_rev * 100 / total_rev) if total_rev > 0 else 0

        # T5: in quarter with most orders → Gold customers' revenue %
        q_counts = _group_count(st, lambda s: s["quarter"])
        q_count_ranked = _rank_groups(q_counts)
        top_q = q_count_ranked[0][0] if q_count_ranked else "Q1"
        top_q_orders = [s for s in st if s["quarter"] == top_q]
        top_q_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in top_q_orders)
        gold_cids = set(c["customer_id"] for c in ct if c["membership"] == "Gold")
        top_q_gold_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in top_q_orders if s["customer_id"] in gold_cids)
        top_q_gold_pct = int(top_q_gold_rev * 100 / top_q_rev) if top_q_rev > 0 else 0

        # T6: qty-weighted avg price vs simple avg price difference
        total_qty = sum(s["quantity"] for s in st)
        qty_weighted_price = int(sum(pm[s["product"]]["price"] * s["quantity"] for s in st) / total_qty) if total_qty > 0 else 0
        simple_avg_price = int(sum(p["price"] for p in pt) / len(pt)) if pt else 0
        price_diff = abs(qty_weighted_price - simple_avg_price)

        question_templates = [
            (f"What percentage of total revenue does the '{tgt_cat}' category represent? (truncate decimals, revenue = price * quantity)",
             tgt_cat_pct, 1,
             f"Step 1: Revenue by category: {', '.join(f'{k}({v})' for k,v in cat_rev_ranked)}\n"
             f"Step 2: Total revenue = {total_rev}\n"
             f"Step 3: '{tgt_cat}' revenue = {tgt_cat_rev}\n"
             f"Step 4: % = {tgt_cat_pct}\nFinal answer: {tgt_cat_pct}"),

            (f"How many customers have total spending above the average per-customer spending? (spending = price * quantity)",
             custs_above, 1,
             f"Step 1: Compute spending per customer ({len(cust_spend)} customers)\n"
             f"Step 2: Average = {int(avg_spend)}\n"
             f"Step 3: Count above average: {custs_above}\nFinal answer: {custs_above}"),

            (f"What percentage of total revenue comes from '{tgt_grade}' membership customers? (truncate decimals)",
             grade_pct, 1,
             f"Step 1: '{tgt_grade}' customers: {len(grade_cids)}\n"
             f"Step 2: '{tgt_grade}' revenue = {grade_rev}\n"
             f"Step 3: Total revenue = {total_rev}\n"
             f"Step 4: % = {grade_pct}\nFinal answer: {grade_pct}"),

            (f"What percentage of total revenue comes from products with stock above the median ({median_stock})? (truncate decimals)",
             high_stock_pct, 1,
             f"Step 1: Median stock = {median_stock}\n"
             f"Step 2: Products with stock > {median_stock}: {len(high_stock_prods)}\n"
             f"Step 3: Their revenue = {high_stock_rev}\n"
             f"Step 4: Total = {total_rev}, % = {high_stock_pct}\nFinal answer: {high_stock_pct}"),

            (f"In the quarter with the most orders ({top_q}), what percentage of that quarter's revenue comes from Gold membership customers? (truncate decimals)",
             top_q_gold_pct, 2,
             f"Step 1: Orders per quarter: {', '.join(f'{k}({v})' for k,v in q_count_ranked)}\n"
             f"Step 2: Most orders: {top_q}\n"
             f"Step 3: {top_q} revenue = {top_q_rev}, Gold revenue in {top_q} = {top_q_gold_rev}\n"
             f"Step 4: % = {top_q_gold_pct}\nFinal answer: {top_q_gold_pct}"),

            (f"What is the difference between the quantity-weighted average price (sum(price×qty)/sum(qty)) and the simple average product price? (absolute value, truncate decimals)",
             price_diff, 2,
             f"Step 1: Qty-weighted avg price = sum(price×qty)/sum(qty) = {qty_weighted_price}\n"
             f"Step 2: Simple avg product price = {simple_avg_price}\n"
             f"Step 3: Difference = {price_diff}\nFinal answer: {price_diff}"),
        ]

    else:  # hard
        # T1: avg-of-avg trap (category avg prices)
        cat_avg = {}
        for cat in categories:
            prods = [p for p in pt if p["category"] == cat]
            if prods:
                cat_avg[cat] = sum(p["price"] for p in prods) / len(prods)
        avg_of_avgs = int(sum(cat_avg.values()) / len(cat_avg)) if cat_avg else 0

        # T2 (Pattern C): Gold-only avg-of-avg vs overall avg-of-avg → difference
        gold_cids = set(c["customer_id"] for c in ct if c["membership"] == "Gold")
        gold_orders = [s for s in st if s["customer_id"] in gold_cids]
        gold_cat_avg = {}
        for cat in categories:
            cat_prods_set = set(p["product"] for p in pt if p["category"] == cat)
            cat_gold = [s for s in gold_orders if s["product"] in cat_prods_set]
            if cat_gold:
                gold_cat_avg[cat] = sum(pm[s["product"]]["price"] * s["quantity"] for s in cat_gold) / len(cat_gold)
        gold_avg_of_avg = int(sum(gold_cat_avg.values()) / len(gold_cat_avg)) if gold_cat_avg else 0
        all_cat_avg_rev = {}
        for cat in categories:
            cat_prods_set = set(p["product"] for p in pt if p["category"] == cat)
            cat_all = [s for s in st if s["product"] in cat_prods_set]
            if cat_all:
                all_cat_avg_rev[cat] = sum(pm[s["product"]]["price"] * s["quantity"] for s in cat_all) / len(cat_all)
        overall_avg_of_avg = int(sum(all_cat_avg_rev.values()) / len(all_cat_avg_rev)) if all_cat_avg_rev else 0
        avg_of_avg_diff = abs(gold_avg_of_avg - overall_avg_of_avg)

        # T3: per-quarter discount revenue % → max quarter vs overall %
        total_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in st)
        total_disc_rev = int(sum(
            pm[s["product"]]["price"] * (100 - pm[s["product"]]["discount"]) / 100 * s["quantity"]
            for s in st
        ))
        overall_disc_pct = int(total_disc_rev * 100 / total_rev) if total_rev > 0 else 0
        q_disc_pcts = {}
        for q in QUARTERS:
            q_orders = [s for s in st if s["quarter"] == q]
            q_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in q_orders)
            q_disc = int(sum(pm[s["product"]]["price"] * (100 - pm[s["product"]]["discount"]) / 100 * s["quantity"] for s in q_orders))
            q_disc_pcts[q] = int(q_disc * 100 / q_rev) if q_rev > 0 else 0
        q_disc_ranked = _rank_groups(q_disc_pcts)
        max_q_disc_pct = q_disc_ranked[0][1] if q_disc_ranked else 0
        disc_pct_diff = abs(max_q_disc_pct - overall_disc_pct)

        # T4 (Pattern D): counterfactual − discount reduced by 5pp
        counterfactual_rev = int(sum(
            pm[s["product"]]["price"] * (100 - max(0, pm[s["product"]]["discount"] - 5)) / 100 * s["quantity"]
            for s in st
        ))
        actual_disc_rev = total_disc_rev
        counterfactual_diff = abs(counterfactual_rev - actual_disc_rev)

        # T5: quarter avg-of-avg vs overall avg difference (kept)
        q_rev_map = _group_sum(st, lambda s: s["quarter"],
                           lambda s: pm[s["product"]]["price"] * s["quantity"])
        q_counts = _group_count(st, lambda s: s["quarter"])
        q_avgs = {q: q_rev_map.get(q, 0) / q_counts.get(q, 1) for q in q_rev_map}
        avg_of_q_avgs = int(sum(q_avgs.values()) / len(q_avgs)) if q_avgs else 0
        overall_avg = int(total_rev / len(st)) if st else 0
        diff_avgs = abs(avg_of_q_avgs - overall_avg)

        # T6: coefficient of variation of category revenues
        cat_rev = _group_sum(st, lambda s: pm[s["product"]]["category"],
                             lambda s: pm[s["product"]]["price"] * s["quantity"])
        cat_rev_vals = list(cat_rev.values())
        cat_rev_mean = sum(cat_rev_vals) / len(cat_rev_vals) if cat_rev_vals else 1
        cat_rev_sd = _std_dev(cat_rev_vals)
        cv = int(cat_rev_sd * 100 / cat_rev_mean) if cat_rev_mean > 0 else 0

        question_templates = [
            (f"What is the overall average of each category's average product price? (First compute average price per category, then average those values. Truncate decimals.)",
             avg_of_avgs, 1,
             f"Step 1: Group products by category\n"
             f"Step 2: Avg per category: {', '.join(f'{k}({v:.0f})' for k,v in cat_avg.items())}\n"
             f"Step 3: Avg of avgs = {avg_of_avgs}\nFinal answer: {avg_of_avgs}"),

            (f"Compute the average of per-category average order revenue for Gold customers only, and for all customers. What is the absolute difference? (Per-category avg = total category revenue / number of orders in that category. Then average those. Truncate decimals.)",
             avg_of_avg_diff, 1,
             f"Step 1: Gold-only per-category avg revenue: {', '.join(f'{k}({int(v)})' for k,v in gold_cat_avg.items())}\n"
             f"Step 2: Gold avg-of-avg = {gold_avg_of_avg}\n"
             f"Step 3: Overall per-category avg revenue: {', '.join(f'{k}({int(v)})' for k,v in all_cat_avg_rev.items())}\n"
             f"Step 4: Overall avg-of-avg = {overall_avg_of_avg}\n"
             f"Step 5: Difference = {avg_of_avg_diff}\nFinal answer: {avg_of_avg_diff}"),

            (f"For each quarter, compute the discounted revenue as a percentage of regular revenue (discounted_price = price×(100-discount)/100, truncate per-item). Find the quarter with the highest such percentage. What is the difference between that percentage and the overall discounted revenue percentage? (absolute value)",
             disc_pct_diff, 1,
             f"Step 1: Per-quarter disc revenue %: {', '.join(f'{k}({v})' for k,v in q_disc_ranked)}\n"
             f"Step 2: Max = {max_q_disc_pct}%\n"
             f"Step 3: Overall disc % = {overall_disc_pct}%\n"
             f"Step 4: Difference = {disc_pct_diff}\nFinal answer: {disc_pct_diff}"),

            (f"If every product's discount rate were reduced by 5 percentage points (minimum 0%), what would be the difference between actual discounted revenue and counterfactual discounted revenue? (absolute value, truncate per-item discounted_price = price×(100-discount)/100)",
             counterfactual_diff, 1,
             f"Step 1: Actual discounted revenue = {actual_disc_rev}\n"
             f"Step 2: Counterfactual (discount-5pp, min 0) revenue = {counterfactual_rev}\n"
             f"Step 3: Difference = {counterfactual_diff}\nFinal answer: {counterfactual_diff}"),

            (f"What is the difference between the average of quarterly per-order average revenue and the overall per-order average revenue? (absolute value, truncate decimals)",
             diff_avgs, 1,
             f"Step 1: Per-quarter avg revenue/order: {', '.join(f'{k}({int(v)})' for k,v in q_avgs.items())}\n"
             f"Step 2: Avg of quarterly avgs = {avg_of_q_avgs}\n"
             f"Step 3: Overall avg per order = {overall_avg}\n"
             f"Step 4: Difference = {diff_avgs}\nFinal answer: {diff_avgs}"),

            (f"What is the coefficient of variation of category revenues? (CV = standard_deviation / mean × 100, truncate to integer. Use population std dev.)",
             cv, 1,
             f"Step 1: Revenue per category: {', '.join(f'{k}({v})' for k,v in _rank_groups(cat_rev))}\n"
             f"Step 2: Mean = {int(cat_rev_mean)}\n"
             f"Step 3: Std dev = {int(cat_rev_sd)}\n"
             f"Step 4: CV = {cv}\nFinal answer: {cv}"),
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
    """Generate array computation problem"""
    pt, st, ct, pm, cm = _generate_all_tables(config, rng)

    categories = list(set(p["category"] for p in pt))
    regions_in_orders = list(set(s["region"] for s in st))

    if config.difficulty == "easy":
        product_prices = {p["product"]: p["price"] for p in pt}
        product_discounts = {p["product"]: p["discount"] for p in pt}

        # T1: total sales
        total_sales = sum(product_prices[s["product"]] * s["quantity"] for s in st)

        # T2: total discounted sales
        total_disc_sales = int(sum(
            product_prices[s["product"]] * (100 - product_discounts[s["product"]]) / 100 * s["quantity"]
            for s in st
        ))

        # T3: category sales
        tgt_cat = rng.choice(categories)
        cat_prods = set(p["product"] for p in pt if p["category"] == tgt_cat)
        cat_orders = [s for s in st if s["product"] in cat_prods]
        cat_sales = sum(product_prices[s["product"]] * s["quantity"] for s in cat_orders)

        # T4: high-qty sales
        qty_thresh = rng.choice([15, 20, 25, 30])
        high_qty_orders = [s for s in st if s["quantity"] > qty_thresh]
        high_qty_sales = sum(product_prices[s["product"]] * s["quantity"] for s in high_qty_orders)

        # T5: region revenue cross-lookup
        tgt_region = rng.choice(list(set(s["region"] for s in st)))
        reg_orders = [s for s in st if s["region"] == tgt_region]
        reg_revenue = sum(product_prices[s["product"]] * s["quantity"] for s in reg_orders)

        # T6: total vs discounted revenue difference
        rev_diff = total_sales - total_disc_sales

        # T7: above-median-stock product revenue
        stocks = sorted(p["stock"] for p in pt)
        median_stock = stocks[len(stocks) // 2]
        above_med_prods = set(p["product"] for p in pt if p["stock"] > median_stock)
        above_med_orders = [s for s in st if s["product"] in above_med_prods]
        above_med_rev = sum(product_prices[s["product"]] * s["quantity"] for s in above_med_orders)

        question_templates = [
            (f"What is the total sales amount across all orders? (Look up price from Products table, revenue = price * quantity)",
             total_sales, 1,
             f"Step 1: For each order, look up price\n"
             f"Step 2: Compute price * quantity\n"
             f"Step 3: Sum = {total_sales}\nFinal answer: {total_sales}"),

            (f"What is the total discounted sales amount? (discounted_price = price * (100-discount)/100, then multiply by quantity. Truncate final answer.)",
             total_disc_sales, 1,
             f"Step 1: For each order, look up price and discount\n"
             f"Step 2: discounted_price * quantity for each\n"
             f"Step 3: Sum = {total_disc_sales}\nFinal answer: {total_disc_sales}"),

            (f"What is the total sales amount for '{tgt_cat}' category products only?",
             cat_sales, 1,
             f"Step 1: '{tgt_cat}' products: {', '.join(list(cat_prods)[:5])}\n"
             f"Step 2: Filter orders: {len(cat_orders)}\n"
             f"Step 3: Sum = {cat_sales}\nFinal answer: {cat_sales}"),

            (f"What is the total sales revenue for orders with quantity greater than {qty_thresh}?",
             high_qty_sales, 1,
             f"Step 1: Filter orders with qty > {qty_thresh}: {len(high_qty_orders)}\n"
             f"Step 2: Look up prices, compute revenue\n"
             f"Step 3: Sum = {high_qty_sales}\nFinal answer: {high_qty_sales}"),

            # region revenue cross-lookup (weight=2)
            (f"What is the total sales revenue for orders placed in '{tgt_region}'? (Look up price from Products table)",
             reg_revenue, 2,
             f"Step 1: Filter orders in '{tgt_region}': {len(reg_orders)}\n"
             f"Step 2: For each, look up price, compute price * quantity\n"
             f"Step 3: Sum = {reg_revenue}\nFinal answer: {reg_revenue}"),

            # total vs discounted revenue difference (weight=2)
            (f"What is the difference between total sales and total discounted sales? (total_sales - discounted_sales, truncate discounted sales before subtracting)",
             rev_diff, 2,
             f"Step 1: Total sales = {total_sales}\n"
             f"Step 2: Total discounted sales = {total_disc_sales}\n"
             f"Step 3: Difference = {rev_diff}\nFinal answer: {rev_diff}"),

            # above-median-stock product revenue (weight=2)
            (f"What is the total sales revenue from products whose stock is above the median stock level? (median = middle value when sorted)",
             above_med_rev, 2,
             f"Step 1: Sort stock values, median = {median_stock}\n"
             f"Step 2: Products with stock > {median_stock}: {len(above_med_prods)}\n"
             f"Step 3: Filter orders for those products: {len(above_med_orders)}\n"
             f"Step 4: Sum revenue = {above_med_rev}\nFinal answer: {above_med_rev}"),
        ]

    elif config.difficulty == "medium":
        product_prices = {p["product"]: p["price"] for p in pt}
        product_discounts = {p["product"]: p["discount"] for p in pt}

        total_rev = sum(product_prices[s["product"]] * s["quantity"] for s in st)

        # T1: above-avg-qty orders' discounted revenue
        avg_qty = total_rev / len(st) if st else 0
        order_qtys = [s["quantity"] for s in st]
        avg_order_qty = sum(order_qtys) / len(order_qtys) if order_qtys else 0
        above_avg_qty_orders = [s for s in st if s["quantity"] > avg_order_qty]
        above_avg_disc_rev = int(sum(
            product_prices[s["product"]] * (100 - product_discounts[s["product"]]) / 100 * s["quantity"]
            for s in above_avg_qty_orders
        ))

        # T2: best region revenue
        reg_rev = _group_sum(st, lambda s: s["region"],
                             lambda s: product_prices[s["product"]] * s["quantity"])
        reg_ranked = _rank_groups(reg_rev)
        top_reg = reg_ranked[0][0] if reg_ranked else ""
        top_reg_rev = reg_ranked[0][1] if reg_ranked else 0

        # T3: Gold-Silver diff
        gold_cids = set(c["customer_id"] for c in ct if c["membership"] == "Gold")
        gold_orders = [s for s in st if s["customer_id"] in gold_cids]
        gold_sales = sum(product_prices[s["product"]] * s["quantity"] for s in gold_orders)
        silver_cids = set(c["customer_id"] for c in ct if c["membership"] == "Silver")
        silver_orders = [s for s in st if s["customer_id"] in silver_cids]
        silver_sales = sum(product_prices[s["product"]] * s["quantity"] for s in silver_orders)
        gold_silver_diff = gold_sales - silver_sales

        # T4: quarter discount revenue
        tgt_q = rng.choice(QUARTERS)
        q_orders = [s for s in st if s["quarter"] == tgt_q]
        q_disc_rev = int(sum(
            product_prices[s["product"]] * (100 - product_discounts[s["product"]]) / 100 * s["quantity"]
            for s in q_orders
        ))

        # T5: per-category revenue range (max - min)
        cat_rev = _group_sum(st, lambda s: pm[s["product"]]["category"],
                             lambda s: product_prices[s["product"]] * s["quantity"])
        cat_ranked = _rank_groups(cat_rev)
        cat_rev_range = cat_ranked[0][1] - cat_ranked[-1][1] if len(cat_ranked) >= 2 else 0

        # T6: per-quarter discounted revenue max-min diff
        q_disc_revs = {}
        for q in QUARTERS:
            qo = [s for s in st if s["quarter"] == q]
            q_disc_revs[q] = int(sum(
                product_prices[s["product"]] * (100 - product_discounts[s["product"]]) / 100 * s["quantity"]
                for s in qo
            ))
        q_disc_ranked = _rank_groups(q_disc_revs)
        q_disc_diff = q_disc_ranked[0][1] - q_disc_ranked[-1][1] if len(q_disc_ranked) >= 2 else 0

        question_templates = [
            # T1: above-avg-qty discounted revenue
            (f"What is the total discounted revenue from orders whose quantity exceeds the average order quantity? (avg qty = {int(avg_order_qty)}, discounted_price = price*(100-discount)/100, truncate final answer)",
             above_avg_disc_rev, 2,
             f"Step 1: Average order qty = {int(avg_order_qty)}\n"
             f"Step 2: Orders with qty > avg: {len(above_avg_qty_orders)}\n"
             f"Step 3: Compute discounted revenue per order\n"
             f"Step 4: Sum = {above_avg_disc_rev}\nFinal answer: {above_avg_disc_rev}"),

            (f"What is the revenue of the region with the highest total sales? (revenue = price * quantity)",
             top_reg_rev, 1,
             f"Step 1: Compute revenue per order\n"
             f"Step 2: By region: {', '.join(f'{k}({v})' for k,v in reg_ranked)}\n"
             f"Step 3: Top = '{top_reg}' ({top_reg_rev})\nFinal answer: {top_reg_rev}"),

            (f"What is the sales difference between Gold and Silver membership customers? (Gold - Silver)",
             gold_silver_diff, 1,
             f"Step 1: Gold sales = {gold_sales}\n"
             f"Step 2: Silver sales = {silver_sales}\n"
             f"Step 3: Difference = {gold_silver_diff}\nFinal answer: {gold_silver_diff}"),

            (f"What is the total discounted sales revenue for {tgt_q} orders? (discounted_price = price * (100-discount)/100, truncate final answer)",
             q_disc_rev, 1,
             f"Step 1: {tgt_q} orders: {len(q_orders)}\n"
             f"Step 2: Look up price and discount\n"
             f"Step 3: Sum = {q_disc_rev}\nFinal answer: {q_disc_rev}"),

            # T5: per-category revenue range
            (f"What is the difference between the highest and lowest category revenue? (revenue = price * quantity for each category)",
             cat_rev_range, 2,
             f"Step 1: Revenue per category: {', '.join(f'{k}({v})' for k,v in cat_ranked)}\n"
             f"Step 2: Max = {cat_ranked[0][1] if cat_ranked else 0}, Min = {cat_ranked[-1][1] if cat_ranked else 0}\n"
             f"Step 3: Range = {cat_rev_range}\nFinal answer: {cat_rev_range}"),

            # T6: per-quarter discounted revenue max-min diff
            (f"What is the difference between the quarter with the highest and lowest total discounted revenue? (discounted_price = price*(100-discount)/100, truncate per-quarter totals)",
             q_disc_diff, 2,
             f"Step 1: Compute discounted revenue per order\n"
             f"Step 2: Sum by quarter: {', '.join(f'{k}({v})' for k,v in q_disc_ranked)}\n"
             f"Step 3: Max - Min = {q_disc_diff}\nFinal answer: {q_disc_diff}"),
        ]

    else:  # hard
        product_prices = {p["product"]: p["price"] for p in pt}
        product_discounts = {p["product"]: p["discount"] for p in pt}
        total_rev = sum(product_prices[s["product"]] * s["quantity"] for s in st)

        # T1: revenue-weighted discount
        weighted_disc_sum = sum(
            product_prices[s["product"]] * s["quantity"] * product_discounts[s["product"]]
            for s in st
        )
        rev_weighted_disc = int(weighted_disc_sum / total_rev) if total_rev > 0 else 0

        # T2: membership x category pivot max/min ratio
        mem_cat_rev = {}
        for s in st:
            mem = cm[s["customer_id"]]["membership"]
            cat = pm[s["product"]]["category"]
            key = (mem, cat)
            rev = product_prices[s["product"]] * s["quantity"]
            mem_cat_rev[key] = mem_cat_rev.get(key, 0) + rev
        if mem_cat_rev:
            mc_max = max(mem_cat_rev.values())
            mc_min = min(v for v in mem_cat_rev.values() if v > 0) if any(v > 0 for v in mem_cat_rev.values()) else 1
            mc_ratio = int(mc_max * 100 / mc_min) if mc_min > 0 else 0
        else:
            mc_max, mc_min, mc_ratio = 0, 1, 0

        # T3: per-region disc revenue comparison (all vs 2021+ customers only)
        post2021_cids = set(c["customer_id"] for c in ct if c["join_year"] >= 2021)
        reg_disc_all = {}
        reg_disc_post = {}
        for s in st:
            reg = s["region"]
            disc_rev = product_prices[s["product"]] * (100 - product_discounts[s["product"]]) / 100 * s["quantity"]
            reg_disc_all[reg] = reg_disc_all.get(reg, 0) + disc_rev
            if s["customer_id"] in post2021_cids:
                reg_disc_post[reg] = reg_disc_post.get(reg, 0) + disc_rev
        reg_diff_vals = {}
        for reg in reg_disc_all:
            reg_diff_vals[reg] = int(reg_disc_all[reg]) - int(reg_disc_post.get(reg, 0))
        max_reg_diff = max(reg_diff_vals.values()) if reg_diff_vals else 0

        # T4: median-derived revenue filter → % of total
        order_revs = sorted(product_prices[s["product"]] * s["quantity"] for s in st)
        median_rev = order_revs[len(order_revs) // 2] if order_revs else 0
        above_med_orders = [s for s in st if product_prices[s["product"]] * s["quantity"] > median_rev]
        above_med_total = sum(product_prices[s["product"]] * s["quantity"] for s in above_med_orders)
        above_med_pct = int(above_med_total * 100 / total_rev) if total_rev > 0 else 0

        # T5: per-category disc/full revenue ratio range
        cat_full_rev = _group_sum(st, lambda s: pm[s["product"]]["category"],
                                  lambda s: product_prices[s["product"]] * s["quantity"])
        cat_disc_rev = _group_sum(st, lambda s: pm[s["product"]]["category"],
                                  lambda s: product_prices[s["product"]] * (100 - product_discounts[s["product"]]) / 100 * s["quantity"])
        cat_ratios = {}
        for cat in cat_full_rev:
            if cat_full_rev[cat] > 0:
                cat_ratios[cat] = int(cat_disc_rev.get(cat, 0) * 100 / cat_full_rev[cat])
        ratio_range = max(cat_ratios.values()) - min(cat_ratios.values()) if len(cat_ratios) >= 2 else 0

        # T6: disc vs nodisc price gap
        disc_prods = [p for p in pt if p["discount"] > 0]
        nodisc_prods = [p for p in pt if p["discount"] == 0]
        if disc_prods:
            disc_weighted_price = int(sum(
                p["price"] * (100 - p["discount"]) / 100 for p in disc_prods
            ) / len(disc_prods))
        else:
            disc_weighted_price = 0
        nodisc_avg_price = int(sum(p["price"] for p in nodisc_prods) / len(nodisc_prods)) if nodisc_prods else 0
        price_gap = abs(disc_weighted_price - nodisc_avg_price)

        question_templates = [
            # T1: revenue-weighted discount
            (f"What is the revenue-weighted average discount rate? (= sum(revenue * discount) / sum(revenue), truncate to integer)",
             rev_weighted_disc, 1,
             f"Step 1: For each order: revenue * discount\n"
             f"Step 2: Weighted sum = {weighted_disc_sum}\n"
             f"Step 3: Total revenue = {total_rev}\n"
             f"Step 4: Average = {rev_weighted_disc}\nFinal answer: {rev_weighted_disc}"),

            # T2: membership x category pivot ratio
            (f"In a membership * category revenue table, what is the ratio of the maximum cell value to the minimum non-zero cell value? (max/min * 100, truncate to integer)",
             mc_ratio, 2,
             f"Step 1: Compute revenue for each (membership, category) pair\n"
             f"Step 2: Max cell = {mc_max}, Min non-zero cell = {mc_min}\n"
             f"Step 3: Ratio = {mc_ratio}\nFinal answer: {mc_ratio}"),

            # T3: per-region disc revenue, all vs post-2021 customers
            (f"For each region, compute total discounted revenue for ALL customers and for customers who joined in 2021 or later. What is the maximum difference (all - post-2021) across regions? (discounted_price = price*(100-discount)/100, truncate each region total)",
             max_reg_diff, 2,
             f"Step 1: Identify post-2021 customers: {len(post2021_cids)}\n"
             f"Step 2: Per-region discounted revenue (all vs post-2021)\n"
             f"Step 3: Differences: {', '.join(f'{k}({v})' for k,v in sorted(reg_diff_vals.items(), key=lambda x: -x[1]))}\n"
             f"Step 4: Max diff = {max_reg_diff}\nFinal answer: {max_reg_diff}"),

            # T4: median-filtered revenue percentage
            (f"What percentage of total revenue comes from orders whose individual revenue (price * quantity) exceeds the median order revenue? (median = middle value when sorted, truncate percentage)",
             above_med_pct, 2,
             f"Step 1: Compute revenue per order, sort\n"
             f"Step 2: Median order revenue = {median_rev}\n"
             f"Step 3: Orders above median: {len(above_med_orders)}, total rev = {above_med_total}\n"
             f"Step 4: Percentage = {above_med_pct}%\nFinal answer: {above_med_pct}"),

            # T5: per-category disc/full ratio range
            (f"For each category, compute the discount retention ratio = (discounted revenue / full revenue) * 100 (truncate per category). What is the range (max ratio - min ratio)?",
             ratio_range, 2,
             f"Step 1: Per category: full rev and disc rev\n"
             f"Step 2: Ratios: {', '.join(f'{k}({v})' for k,v in sorted(cat_ratios.items(), key=lambda x: -x[1]))}\n"
             f"Step 3: Range = {ratio_range}\nFinal answer: {ratio_range}"),

            # T6: disc vs nodisc price gap
            (f"What is the difference between the average discounted price of discounted products (discount > 0) and the average price of non-discounted products (discount = 0)? (absolute value, truncate decimals)",
             price_gap, 1,
             f"Step 1: Discounted products ({len(disc_prods)}): avg discounted price = {disc_weighted_price}\n"
             f"Step 2: Non-discounted products ({len(nodisc_prods)}): avg price = {nodisc_avg_price}\n"
             f"Step 3: Difference = {price_gap}\nFinal answer: {price_gap}"),
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
    """Generate multi-condition problem"""
    pt, st, ct, pm, cm = _generate_all_tables(config, rng)

    categories = list(set(p["category"] for p in pt))
    regions_in_orders = list(set(s["region"] for s in st))
    tgt_cat = rng.choice(categories)

    if config.difficulty == "easy":
        tgt_region = rng.choice(regions_in_orders)
        product_cats = {p["product"]: p["category"] for p in pt}

        # T1: region+category count
        filtered_rc = [s for s in st
                       if s["region"] == tgt_region
                       and product_cats.get(s["product"]) == tgt_cat]
        rc_count = len(filtered_rc)

        # T2: region+quarter qty
        tgt_quarter = rng.choice(QUARTERS)
        filtered_rq = [s for s in st
                       if s["region"] == tgt_region and s["quarter"] == tgt_quarter]
        rq_qty = sum(s["quantity"] for s in filtered_rq)

        # T3: region revenue
        reg_orders = [s for s in st if s["region"] == tgt_region]
        reg_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in reg_orders)

        # T4: Gold total qty
        gold_cids = set(c["customer_id"] for c in ct if c["membership"] == "Gold")
        gold_orders = [s for s in st if s["customer_id"] in gold_cids]
        gold_qty = sum(s["quantity"] for s in gold_orders)

        # T5: region+category revenue
        rc_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in filtered_rc)

        # T6: category+quarter order count
        cat_prods_set = set(p["product"] for p in pt if p["category"] == tgt_cat)
        cat_q_orders = [s for s in st if s["product"] in cat_prods_set and s["quarter"] == tgt_quarter]
        cat_q_count = len(cat_q_orders)

        # T7: region discounted revenue
        reg_disc_rev = int(sum(
            pm[s["product"]]["price"] * (100 - pm[s["product"]]["discount"]) / 100 * s["quantity"]
            for s in reg_orders
        ))

        # T8: above-avg-price products in region count
        avg_price = sum(p["price"] for p in pt) / len(pt) if pt else 0
        high_price_prods = set(p["product"] for p in pt if p["price"] > avg_price)
        high_price_reg_orders = [s for s in st if s["product"] in high_price_prods and s["region"] == tgt_region]
        high_price_reg_count = len(high_price_reg_orders)

        question_templates = [
            (f"How many orders for '{tgt_cat}' products were placed in '{tgt_region}'?",
             rc_count, 1,
             f"Step 1: Identify '{tgt_cat}' products from Products table\n"
             f"Step 2: Filter region='{tgt_region}' AND category='{tgt_cat}'\n"
             f"Step 3: Count = {rc_count}\nFinal answer: {rc_count}"),

            (f"What is the total order quantity in '{tgt_region}' during {tgt_quarter}?",
             rq_qty, 1,
             f"Step 1: Filter region='{tgt_region}' AND quarter='{tgt_quarter}': {len(filtered_rq)} orders\n"
             f"Step 2: Sum qty = {rq_qty}\nFinal answer: {rq_qty}"),

            (f"What is the total sales revenue for all orders in '{tgt_region}'? (Look up prices from Products table)",
             reg_rev, 1,
             f"Step 1: Filter region='{tgt_region}': {len(reg_orders)} orders\n"
             f"Step 2: Sum price * qty = {reg_rev}\nFinal answer: {reg_rev}"),

            (f"What is the total order quantity for Gold membership customers?",
             gold_qty, 1,
             f"Step 1: Gold customers: {len(gold_cids)}\n"
             f"Step 2: Gold orders: {len(gold_orders)}\n"
             f"Step 3: Sum qty = {gold_qty}\nFinal answer: {gold_qty}"),

            (f"What is the total sales revenue for '{tgt_cat}' products in '{tgt_region}'? (Look up prices from Products table)",
             rc_rev, 1,
             f"Step 1: '{tgt_cat}' products in '{tgt_region}' orders: {len(filtered_rc)}\n"
             f"Step 2: Sum price * qty = {rc_rev}\nFinal answer: {rc_rev}"),

            # category+quarter order count (weight=2)
            (f"How many orders for '{tgt_cat}' products were placed during {tgt_quarter}?",
             cat_q_count, 2,
             f"Step 1: Identify '{tgt_cat}' products\n"
             f"Step 2: Filter quarter='{tgt_quarter}' AND category='{tgt_cat}'\n"
             f"Step 3: Count = {cat_q_count}\nFinal answer: {cat_q_count}"),

            # region discounted revenue (weight=2)
            (f"What is the total discounted revenue for orders in '{tgt_region}'? (discounted_price = price*(100-discount)/100, truncate final answer)",
             reg_disc_rev, 2,
             f"Step 1: Filter region='{tgt_region}': {len(reg_orders)} orders\n"
             f"Step 2: For each, compute discounted_price * quantity\n"
             f"Step 3: Sum = {reg_disc_rev}\nFinal answer: {reg_disc_rev}"),

            # above-avg-price products in region (weight=2)
            (f"How many orders in '{tgt_region}' are for products with price above the average product price ({int(avg_price)})?",
             high_price_reg_count, 2,
             f"Step 1: Average product price = {int(avg_price)}\n"
             f"Step 2: Products with price > avg: {len(high_price_prods)}\n"
             f"Step 3: Filter those products in '{tgt_region}': {high_price_reg_count}\nFinal answer: {high_price_reg_count}"),
        ]

    elif config.difficulty == "medium":
        # T1: category region top
        cat_prods = set(p["product"] for p in pt if p["category"] == tgt_cat)
        cat_orders = [s for s in st if s["product"] in cat_prods]
        reg_qty = _group_sum(cat_orders, lambda s: s["region"], lambda s: s["quantity"])
        reg_ranked = _rank_groups(reg_qty)
        top_region = reg_ranked[0][0] if reg_ranked else ""

        # T2: region revenue 2nd
        region_rev = _group_sum(st, lambda s: s["region"],
                                lambda s: pm[s["product"]]["price"] * s["quantity"])
        rev_ranked = _rank_groups(region_rev)
        second_rev = rev_ranked[1][1] if len(rev_ranked) >= 2 else 0

        # T3: quarter most orders
        q_counts = _group_count(st, lambda s: s["quarter"])
        q_count_ranked = _rank_groups(q_counts)
        top_q = q_count_ranked[0][0] if q_count_ranked else "Q1"

        # T4: Gold category with most distinct products ordered
        gold_cids = set(c["customer_id"] for c in ct if c["membership"] == "Gold")
        gold_orders = [s for s in st if s["customer_id"] in gold_cids]
        gold_cat_prods = {}
        for s in gold_orders:
            cat = pm[s["product"]]["category"]
            if cat not in gold_cat_prods:
                gold_cat_prods[cat] = set()
            gold_cat_prods[cat].add(s["product"])
        gold_cat_distinct = {k: len(v) for k, v in gold_cat_prods.items()}
        gold_cat_ranked = _rank_groups(gold_cat_distinct)
        top_gold_cat = gold_cat_ranked[0][0] if gold_cat_ranked else ""

        # T5: category with most orders → discounted revenue of that category
        cat_order_counts = _group_count(st, lambda s: pm[s["product"]]["category"])
        cat_count_ranked = _rank_groups(cat_order_counts)
        top_count_cat = cat_count_ranked[0][0] if cat_count_ranked else ""
        top_count_cat_prods = set(p["product"] for p in pt if p["category"] == top_count_cat)
        top_count_cat_orders = [s for s in st if s["product"] in top_count_cat_prods]
        top_count_cat_disc_rev = int(sum(
            pm[s["product"]]["price"] * (100 - pm[s["product"]]["discount"]) / 100 * s["quantity"]
            for s in top_count_cat_orders
        ))

        # T6: per-region Gold customer order count difference vs overall
        tgt_region = rng.choice(regions_in_orders)
        reg_total_orders = len([s for s in st if s["region"] == tgt_region])
        reg_gold_orders = len([s for s in st if s["region"] == tgt_region and s["customer_id"] in gold_cids])
        reg_non_gold = reg_total_orders - reg_gold_orders

        question_templates = [
            (f"In the '{tgt_cat}' category, which region has the most orders by quantity? (Answer with region name)",
             top_region, 1,
             f"Step 1: '{tgt_cat}' products\n"
             f"Step 2: Filter orders: {len(cat_orders)}\n"
             f"Step 3: By region: {', '.join(f'{k}({v})' for k,v in reg_ranked)}\n"
             f"Step 4: Top = '{top_region}'\nFinal answer: {top_region}"),

            (f"What is the sales revenue of the region with the 2nd highest total sales?",
             second_rev, 1,
             f"Step 1-2: Revenue per region: {', '.join(f'{k}({v})' for k,v in rev_ranked)}\n"
             f"Step 3: 2nd = {second_rev}\nFinal answer: {second_rev}"),

            (f"Which quarter has the highest number of orders? (Answer with quarter name, e.g. Q1)",
             top_q, 1,
             f"Step 1: Count per quarter: {', '.join(f'{k}({v})' for k,v in q_count_ranked)}\n"
             f"Step 2: Top = '{top_q}'\nFinal answer: {top_q}"),

            # T4: Gold distinct products by category
            (f"Among Gold membership customers, which category has the most distinct products ordered? (Answer with category name)",
             top_gold_cat, 2,
             f"Step 1: Gold customers: {len(gold_cids)}\n"
             f"Step 2: Gold orders: {len(gold_orders)}\n"
             f"Step 3: Distinct products per category: {', '.join(f'{k}({v})' for k,v in gold_cat_ranked)}\n"
             f"Step 4: Top = '{top_gold_cat}'\nFinal answer: {top_gold_cat}"),

            # T5: top-count category → discounted revenue
            (f"Which category has the most orders? What is the total discounted revenue for that category? (discounted_price = price*(100-discount)/100, truncate final answer)",
             top_count_cat_disc_rev, 2,
             f"Step 1: Orders per category: {', '.join(f'{k}({v})' for k,v in cat_count_ranked)}\n"
             f"Step 2: Top category = '{top_count_cat}'\n"
             f"Step 3: Discounted revenue for '{top_count_cat}': {top_count_cat_disc_rev}\nFinal answer: {top_count_cat_disc_rev}"),

            # T6: region Gold vs non-Gold order count difference
            (f"In '{tgt_region}', how many more non-Gold orders are there compared to Gold membership orders? (non_Gold - Gold)",
             reg_non_gold - reg_gold_orders, 2,
             f"Step 1: '{tgt_region}' total orders: {reg_total_orders}\n"
             f"Step 2: Gold orders in '{tgt_region}': {reg_gold_orders}\n"
             f"Step 3: Non-Gold = {reg_non_gold}\n"
             f"Step 4: Difference = {reg_non_gold - reg_gold_orders}\nFinal answer: {reg_non_gold - reg_gold_orders}"),
        ]

    else:  # hard
        # T1: top-3 spender exclude + region
        tgt_region = rng.choice(regions_in_orders)
        cust_spend = _group_sum(st, lambda s: s["customer_id"],
                                lambda s: pm[s["product"]]["price"] * s["quantity"])
        spend_ranked = _rank_groups(cust_spend)
        top3_cids = set(cid for cid, _ in spend_ranked[:3])
        excl_top3_reg = [s for s in st if s["customer_id"] not in top3_cids and s["region"] == tgt_region]
        excl_top3_reg_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in excl_top3_reg)

        # T2: above-median spenders vs below-median → disc≥15% revenue % difference
        all_spends = sorted(cust_spend.values())
        median_spend = all_spends[len(all_spends) // 2] if all_spends else 0
        above_med_cids = set(cid for cid, sp in cust_spend.items() if sp > median_spend)
        below_med_cids = set(cid for cid, sp in cust_spend.items() if sp <= median_spend)
        high_disc_prods_15 = set(p["product"] for p in pt if p["discount"] >= 15)
        above_med_disc15_rev = sum(pm[s["product"]]["price"] * s["quantity"]
                                   for s in st if s["customer_id"] in above_med_cids and s["product"] in high_disc_prods_15)
        above_med_total = sum(pm[s["product"]]["price"] * s["quantity"]
                              for s in st if s["customer_id"] in above_med_cids)
        below_med_disc15_rev = sum(pm[s["product"]]["price"] * s["quantity"]
                                   for s in st if s["customer_id"] in below_med_cids and s["product"] in high_disc_prods_15)
        below_med_total = sum(pm[s["product"]]["price"] * s["quantity"]
                              for s in st if s["customer_id"] in below_med_cids)
        above_pct = int(above_med_disc15_rev * 100 / above_med_total) if above_med_total > 0 else 0
        below_pct = int(below_med_disc15_rev * 100 / below_med_total) if below_med_total > 0 else 0
        med_pct_diff = abs(above_pct - below_pct)

        # T3: Gold category %
        gold_cids = set(c["customer_id"] for c in ct if c["membership"] == "Gold")
        gold_orders = [s for s in st if s["customer_id"] in gold_cids]
        gold_total = sum(pm[s["product"]]["price"] * s["quantity"] for s in gold_orders)
        cat_prods = set(p["product"] for p in pt if p["category"] == tgt_cat)
        gold_cat_orders = [s for s in gold_orders if s["product"] in cat_prods]
        gold_cat_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in gold_cat_orders)
        gold_cat_pct = int(gold_cat_rev * 100 / gold_total) if gold_total > 0 else 0

        # T4: top revenue region → exclude top-2 spenders → avg order value
        region_rev = _group_sum(st, lambda s: s["region"],
                                lambda s: pm[s["product"]]["price"] * s["quantity"])
        reg_rev_ranked = _rank_groups(region_rev)
        top_rev_region = reg_rev_ranked[0][0] if reg_rev_ranked else tgt_region
        top2_cids = set(cid for cid, _ in spend_ranked[:2])
        top_reg_excl_orders = [s for s in st if s["region"] == top_rev_region and s["customer_id"] not in top2_cids]
        top_reg_excl_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in top_reg_excl_orders)
        top_reg_excl_avg = int(top_reg_excl_rev / len(top_reg_excl_orders)) if top_reg_excl_orders else 0

        # T5: 3rd region by count → top customer → % of their spending in that region
        reg_counts = _group_count(st, lambda s: s["region"])
        reg_count_ranked = _rank_groups(reg_counts)
        third_region = reg_count_ranked[2][0] if len(reg_count_ranked) >= 3 else (reg_count_ranked[-1][0] if reg_count_ranked else tgt_region)
        third_reg_orders = [s for s in st if s["region"] == third_region]
        third_reg_cust_spend = _group_sum(third_reg_orders, lambda s: s["customer_id"],
                                          lambda s: pm[s["product"]]["price"] * s["quantity"])
        third_reg_spend_ranked = _rank_groups(third_reg_cust_spend)
        top_cust_in_third = third_reg_spend_ranked[0][0] if third_reg_spend_ranked else ""
        top_cust_total_spend = cust_spend.get(top_cust_in_third, 0)
        top_cust_third_spend = third_reg_cust_spend.get(top_cust_in_third, 0)
        top_cust_third_pct = int(top_cust_third_spend * 100 / top_cust_total_spend) if top_cust_total_spend > 0 else 0

        # T6: Gold+Silver avg rev/order vs Bronze+None → ratio×100
        gs_cids = set(c["customer_id"] for c in ct if c["membership"] in ("Gold", "Silver"))
        bn_cids = set(c["customer_id"] for c in ct if c["membership"] in ("Bronze", "None"))
        gs_orders = [s for s in st if s["customer_id"] in gs_cids]
        bn_orders = [s for s in st if s["customer_id"] in bn_cids]
        gs_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in gs_orders)
        bn_rev = sum(pm[s["product"]]["price"] * s["quantity"] for s in bn_orders)
        gs_avg = gs_rev / len(gs_orders) if gs_orders else 0
        bn_avg = bn_rev / len(bn_orders) if bn_orders else 1
        gs_bn_ratio = int(gs_avg * 100 / bn_avg) if bn_avg > 0 else 0

        question_templates = [
            # T1
            (f"Excluding the top 3 customers by total spending, what is the sales revenue in '{tgt_region}'?",
             excl_top3_reg_rev, 1,
             f"Step 1: Spending per customer: {', '.join(f'{k}({v})' for k,v in spend_ranked[:5])}\n"
             f"Step 2: Top 3 = {', '.join(top3_cids)}\n"
             f"Step 3: Excluding top 3, '{tgt_region}' orders: {len(excl_top3_reg)}\n"
             f"Step 4: Revenue = {excl_top3_reg_rev}\nFinal answer: {excl_top3_reg_rev}"),

            # T2: above vs below median spender disc≥15% revenue %
            (f"Split customers into above-median and below-median spenders. For each group, what % of revenue comes from products with discount >= 15%? What is the absolute difference? (truncate each %, median = middle value)",
             med_pct_diff, 2,
             f"Step 1: Customer spending sorted, median = {median_spend}\n"
             f"Step 2: Above-median: disc≥15% rev = {above_med_disc15_rev}, total = {above_med_total}, % = {above_pct}\n"
             f"Step 3: Below-median: disc≥15% rev = {below_med_disc15_rev}, total = {below_med_total}, % = {below_pct}\n"
             f"Step 4: Diff = {med_pct_diff}\nFinal answer: {med_pct_diff}"),

            # T3: Gold category %
            (f"What percentage of Gold membership customers' total spending is on '{tgt_cat}' products? (truncate decimals)",
             gold_cat_pct, 1,
             f"Step 1: Gold customers: {len(gold_cids)}\n"
             f"Step 2: Gold total = {gold_total}\n"
             f"Step 3: Gold '{tgt_cat}' = {gold_cat_rev}\n"
             f"Step 4: % = {gold_cat_pct}\nFinal answer: {gold_cat_pct}"),

            # T4: top revenue region → exclude top-2 → avg order value
            (f"In the region with the highest total revenue ('{top_rev_region}'), excluding orders from the top 2 spenders overall, what is the average order value? (truncate decimals)",
             top_reg_excl_avg, 2,
             f"Step 1: Top revenue region = '{top_rev_region}'\n"
             f"Step 2: Top 2 spenders: {', '.join(top2_cids)}\n"
             f"Step 3: '{top_rev_region}' orders excl top 2: {len(top_reg_excl_orders)}, rev = {top_reg_excl_rev}\n"
             f"Step 4: Avg = {top_reg_excl_avg}\nFinal answer: {top_reg_excl_avg}"),

            # T5: 3rd region → top customer → % in that region
            (f"In the region with the 3rd highest order count ('{third_region}'), who is the top spender? What percentage of their total spending is in '{third_region}'? (truncate decimals)",
             top_cust_third_pct, 2,
             f"Step 1: Orders per region: {', '.join(f'{k}({v})' for k,v in reg_count_ranked)}\n"
             f"Step 2: 3rd region = '{third_region}'\n"
             f"Step 3: Top spender in '{third_region}' = '{top_cust_in_third}' ({top_cust_third_spend})\n"
             f"Step 4: Their total = {top_cust_total_spend}, % = {top_cust_third_pct}\nFinal answer: {top_cust_third_pct}"),

            # T6: Gold+Silver vs Bronze+None avg rev/order ratio
            (f"What is the ratio of average revenue per order for Gold+Silver customers vs Bronze+None customers? (Gold+Silver avg / Bronze+None avg * 100, truncate)",
             gs_bn_ratio, 2,
             f"Step 1: Gold+Silver orders: {len(gs_orders)}, rev = {gs_rev}, avg = {int(gs_avg)}\n"
             f"Step 2: Bronze+None orders: {len(bn_orders)}, rev = {bn_rev}, avg = {int(bn_avg)}\n"
             f"Step 3: Ratio = {gs_bn_ratio}\nFinal answer: {gs_bn_ratio}"),
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
# Main Generation Functions
# ============================================================

PROBLEM_GENERATORS = {
    ProblemType.LOOKUP_QUERY.value: generate_lookup_problem,
    ProblemType.CONDITIONAL_AGGREGATION.value: generate_conditional_aggregation_problem,
    ProblemType.ARRAY_COMPUTATION.value: generate_array_computation_problem,
    ProblemType.MULTI_CONDITION.value: generate_multi_condition_problem,
}


def _generation_difficulty_for_target(label_difficulty: str, rng: random.Random) -> str:
    """Map dataset labels to calibrated generation difficulty mixtures."""
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
    """Generate a single puzzle (problem_type=None picks one at random)."""
    if seed is None:
        seed = random.randint(1, 1000000)

    requested_difficulty = difficulty
    # Reject puzzles whose final answer is ambiguous (e.g. a membership lookup that
    # resolves to the "None" tier, which is indistinguishable from "no answer").
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

    puzzle_hash = hashlib.md5(json.dumps(puzzle, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    puzzle["id"] = f"af_{requested_difficulty}_{problem_type}_{puzzle_hash}"
    puzzle["seed"] = seed

    return puzzle


def generate_dataset(
    num_per_difficulty: int = 100,
    seed: int = 2025
) -> List[Dict[str, Any]]:
    """Generate a dataset spanning all difficulty levels."""
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
                puzzle["id"] = f"array_formula_en_{difficulty}_{diff_idx:04d}"
                puzzles.append(puzzle)
                puzzle_seed += 1
                diff_idx += 1

    return puzzles


def format_table_for_prompt(table_name: str, table_data: Dict) -> str:
    """Format table as prompt string"""
    columns = table_data["columns"]
    data = table_data["data"]

    lines = [f"[{table_name} Table]"]
    header = " | ".join(str(col) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))

    for row in data:
        row_str = " | ".join(str(row.get(col, "")) for col in columns)
        lines.append(row_str)

    return "\n".join(lines)


def puzzle_to_prompt(puzzle: Dict[str, Any]) -> str:
    """Convert puzzle to LLM prompt"""
    prompt_parts = []

    prompt_parts.append("The following is spreadsheet data.\n")

    for table_name, table_data in puzzle["tables"].items():
        prompt_parts.append(format_table_for_prompt(table_name, table_data))
        prompt_parts.append("")

    prompt_parts.append(f"Question: {puzzle['question']}")

    if puzzle.get("answer_type") == "number":
        prompt_parts.append("\nAnswer with only a number. (no units)")
    else:
        prompt_parts.append("\nAnswer with the exact value.")

    return "\n".join(prompt_parts)


_SOLUTION_TYPE_LABELS_EN = {
    ProblemType.LOOKUP_QUERY.value: "Lookup (INDEX/MATCH, VLOOKUP-style)",
    ProblemType.CONDITIONAL_AGGREGATION.value: "Conditional aggregation (SUMIF, COUNTIF-style)",
    ProblemType.ARRAY_COMPUTATION.value: "Array computation (SUMPRODUCT-style)",
    ProblemType.MULTI_CONDITION.value: "Multi-criteria (SUMIFS, MAXIFS-style)",
}

_SFT_TYPE_REASONING_NUDGE_EN = {
    ProblemType.LOOKUP_QUERY.value: (
        "Decide which tables to join (products/orders/customers) and on which keys, "
        "then see where **ranking / top-k / exclusions** (1st, 2nd, exclude) attach after filters."
    ),
    ProblemType.CONDITIONAL_AGGREGATION.value: (
        "Select rows that match the question’s **condition columns** (region, quarter, tier, …), "
        "then COUNT/SUM/avg; apply **truncation/rounding** only as stated, usually at the end."
    ),
    ProblemType.ARRAY_COMPUTATION.value: (
        "Align rows (orders↔products), build **per-element** quantities like qty×price, and note if "
        "the question is max/min/sum on a product row or a joint slice."
    ),
    ProblemType.MULTI_CONDITION.value: (
        "With AND/OR conditions, **narrow the row set stepwise**; when sets overlap, be clear whether "
        "you aggregate over all orders or only those matching every clause."
    ),
}

SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)

_SFT_PIPELINE_HINT_EN = {
    ProblemType.LOOKUP_QUERY.value: "filter match -> pick the target column",
    ProblemType.CONDITIONAL_AGGREGATION.value: (
        "WHERE(condition) -> single-column aggregate (SUM/COUNT/AVG)"),
    ProblemType.ARRAY_COMPUTATION.value: (
        "join (products<->orders) -> elementwise multiply then sum "
        "(SUMPRODUCT)"),
    ProblemType.MULTI_CONDITION.value: (
        "multi-condition WHERE -> aggregate/sort/rank"),
}


def _truncate_for_solution_prompt_en(text: str, max_len: int = 400) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


_STEP_PREFIX_EN = re.compile(r"^\s*(?:Step\s*\d+\s*[:：-]|\d+\s*단계\s*[:：-])\s*", re.IGNORECASE)
_FINAL_PREFIX_EN = re.compile(r"^\s*(?:Final\s*answer|최종\s*답)\s*[:：-]\s*", re.IGNORECASE)


def _worked_body_lines_en(solution: str) -> list:
    """Re-number the generator's 'Step N: …' lines as [SEG n]; drop 'Final answer:' line."""
    s = (solution or "").strip()
    if not s:
        return [
            "    [SEG 1] (Generator would place **Step 1, Step 2, …** intermediate work here. "
            "If empty, write your own: join keys, filters, then aggregates.)"
        ]
    out, seg = [], 1
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _FINAL_PREFIX_EN.match(line):
            continue
        body = _STEP_PREFIX_EN.sub("", line)
        out.append(f"    [SEG {seg}] {body}")
        seg += 1
    if not out:
        out.append(
            "    [SEG 1] (No labelled steps in the raw solution; walk question→filters→aggregates yourself.)"
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
    """SFT: emphasize reasoning trace, not a dry answer key."""
    type_label = _SOLUTION_TYPE_LABELS_EN.get(problem_type, problem_type)
    fmt = (
        "numeric only (no units)"
        if answer_type == "number"
        else "text (match the problem wording exactly)"
    )
    nudge = _SFT_TYPE_REASONING_NUDGE_EN.get(
        problem_type,
        "First pick the **grain** (customer vs order vs region vs product), then join, filter, aggregate; "
        "the numbers in each line differ by instance.",
    )
    q_line = _truncate_for_solution_prompt_en(question) if question else (
        "(The exact question is under ‘Question:’ in the prompt above.)"
    )
    lines = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Problem type: {type_label}",
        f"  - Difficulty: {difficulty}",
        f"  - How to state the final answer: {fmt}",
        "  - (How to read the task) " + nudge,
        "  - The numeric/text answer is only fixed in [STEP3] (verification). "
        "Follow the **step log** in [STEP2] first.",
        "[STEP 1] Given",
        f"  - **This question (verbatim, shortened if long)**: {q_line}",
        "  - Schema (same columns as the tables in the prompt):",
        "      Products: id, product, category, price, stock, discount",
        "      Orders: order_id, product, region, quantity, quarter, customer_id",
        "      Customers: customer_id, name, membership, join_year, region",
        "[STEP 2] Worked solution (per-instance **intermediate work**; varies by problem)",
    ]
    worked = _worked_body_lines_en(solution)
    pipeline = _SFT_PIPELINE_HINT_EN.get(
        problem_type,
        "join -> filter -> aggregate/rank",
    )
    lines.append(
        f"  · Summary: {type_label} · pipeline: {pipeline} · "
        f"{len(worked)} SEGs")
    lines.append(
        "  · Mentally: (1) join keys (2) filters (3) aggregations or ranks "
        "— the lines below implement that")
    lines.extend(worked)
    lines.extend([
        "[STEP 3] Answer and verification",
        f"  - Final answer: {answer}",
        "  - Checks: truncation/rounding; where discount applies (price, stock, or revenue); "
        "and join keys (product name, customer_id, etc.) line up with the prompt tables.",
    ])
    return "\n".join(lines)


def save_dataset(
    puzzles: List[Dict],
    base_dir: str = "./data"
):
    """Save dataset as data/{task}_{tier}.jsonl (release layout)."""
    # Release layout: data/{task}_{tier}.jsonl
    processed_puzzles = []
    for puzzle in puzzles:
        question = puzzle_to_prompt(puzzle)

        processed = {
            "id": puzzle["id"],
            "question": question,
            # answer is always stringified: keeps the column one type across CSV/
            # HuggingFace loaders, and avoids losing leading zeros on numeric answers
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

    write_release_jsonl("array_formula_en", processed_puzzles)

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

    parser = argparse.ArgumentParser(description="Array Formula Puzzle Generator")
    parser.add_argument("--num", type=int, default=200, help="Number of puzzles per difficulty level")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output", type=str, default="./data", help="Output base directory")
    parser.add_argument("--demo", action="store_true", help="Print demo puzzles")

    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("Array Formula Puzzle Demo")
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
                print(f"\nAnswer: {puzzle['answer']}")
                if puzzle.get("solution"):
                    print(f"Solution: {puzzle['solution']}")
                print("=" * 60)
                break
    else:
        puzzles = generate_dataset(
            num_per_difficulty=args.num,
            seed=args.seed
        )
        save_dataset(puzzles, args.output)

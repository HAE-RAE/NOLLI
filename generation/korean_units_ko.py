"""
한국 전통 단위 환산 퍼즐 생성기 (한국어) — gemini-3-flash 기준 난이도 보정판

되·말·섬(부피), 푼·치·자·장(길이), 돈·냥·근·관(무게), 평·단·정보·결(넓이) 등
전통 도량형을 **문제에 함께 제시된 환산표만 보고** 기준 단위로 환산/합산하는 퍼즐.
실제 단위 지식이 아니라 주어진 표로만 풀게 하는 자기완결형. 정답은 항상 기준 단위
단일 정수(예: "373돈")라 기존 KoreanUnitsEvaluator 로 그대로 채점된다.

본질이 다단계 산술이라 frontier 모델은 ×10·소규모로는 거의 100% 포화된다. 그래서
난이도를 다음 **질적·양적 레버**의 조합으로 스케일한다(아래 _CONFIG):
  [질적]
  - ratio_mode: 환산비의 더러움. decimal(×10, 자리이동=공짜) → semi(일부 비-십진) →
    ugly(매 단계 7·11·13·16·17·19·23 등 서로 다른 소수성 비율 → 매 단계 실제 곱셈 강제).
  - scalar: 항목별 '×k 배' (환산값에 곱하는 단계 추가).
  - subtract: '더함/뺌' 부호 혼합 (부호 추적 + 음수 중간값 위험).
  [양적]
  - depth: 단위 깊이(연쇄 환산 길이·배수 크기).
  - n_quant: 합산할 항목 수.
  - terms: 항목당 단위 항 수.
  - cmax: 각 단위 계수의 최댓값(피연산자 자릿수).

난이도 보정 전략: 먼저 hard 를 '최대한 어렵게' 잡아 gemini-3-flash 를 100%에서
끌어내린 뒤, 거기서 레버를 줄여 medium/easy 를 세팅한다(_CONFIG 숫자만 바꾸면 됨).

[gemini-3-flash 보정 결과] (2026-06-23, n=100/난이도, max_tokens=65536, reasoning=medium)
- easy   : 74% (74/100)  — ugly·5depth·17항목·계수≤1249·배수·뺌  (house 밴드 70–90 ✓)
- medium : 58% (58/100)  — ugly·5depth·24항목·계수≤2499·배수·뺌  (house 밴드 40–60 ✓)
- hard   : 18% (18/100)  — ugly·5depth·40항목·5항·계수≤9999·배수·뺌 (house 밴드 20–40, 2%p 하회)
주 난이도 레버는 n_quant(합산 항목 수)와 cmax(계수 크기); 정답률 절벽이 n_quant≈22~26 구간.
n=30 표본은 노이즈가 커(SE~8%p) n=100 로 확정함. 더 쉽게/어렵게는 _CONFIG 의
n_quant·cmax·depth·ratio_mode·scalar·subtract 조절.
"""

import random
import json
from pathlib import Path

# 도량형별 단위명 (작은→큰 순). 환산비는 난이도(ratio_mode)가 동적으로 결정한다.
UNIT_SYSTEMS = {
    '무게': ['돈', '냥', '근', '관', '섬'],
    '길이': ['푼', '치', '자', '장', '리'],
    '부피': ['홉', '되', '말', '동이', '섬'],
    '넓이': ['평', '단', '마지기', '정보', '결'],
}

# 비-십진(더러운) 연쇄 환산비 풀 — 매 단계 실제 곱셈을 강제한다.
UGLY_RATIOS = [7, 11, 12, 13, 16, 17, 19, 23]

# 난이도별 레버 프리셋. (gemini-3-flash 기준 보정 대상)
# hard 보정 이력 (flash 기준, 합격 밴드 15~35%):
#   n_quant 40 + scalar(2,49) 아님, cmax 9999 -> 0.15
#   n_quant 30 + scalar(2,49), cmax 9999      -> 0.11
#   n_quant 30 + scalar(2,9),  cmax 9999      -> 0.13
#   → 항목 수(40→30)는 표준오차 내로 거의 무효. 이 구간의 지배 변수는
#     계수 크기(cmax)와 항목당 단위 수(terms)이므로 cmax를 직접 낮춘다.
#   n_quant 26 + cmax 3999                    -> 0.36 (밴드 상한 0.35 초과)
#   → cmax 감도가 커서(9999:0.13, 3999:0.36) 두 점 보간으로 0.25 지점인
#     cmax 6999를 채택.
_CONFIG = {
    'easy': dict(
        depth=5, ratio_mode='ugly', n_quant=17, terms=(4, 5),
        cmax=1249, scalar=True, scalar_range=(2, 9), subtract=True,
        subtract_p=0.25,
    ),
    'medium': dict(
        depth=5, ratio_mode='ugly', n_quant=24, terms=(4, 5),
        cmax=2499, scalar=True, scalar_range=(2, 9), subtract=True,
        subtract_p=0.25,
    ),
    'hard': dict(
        depth=5, ratio_mode='ugly', n_quant=26, terms=(5, 5),
        cmax=6999, scalar=True, scalar_range=(2, 9), subtract=True,
        subtract_p=0.25,
    ),
}

_LABELS = list('갑을병정무기경신임계')

# 다른 KO task 와 동일한 4단 해설 형식. 항목별 환산 줄(`sol`)은 그대로 두고
# STEP 0~3 뼈대만 씌운다.
SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_solution(sol: str, difficulty: str, answer: str,
                    n_items: int, total: int, base: str) -> str:
    # 합계는 [STEP 3]에서 다시 보고하므로 본문 끝의 '최종 합 = ...' 줄은 뺀다.
    lines = sol.splitlines()
    if lines and lines[-1].startswith("최종 합 = "):
        lines = lines[:-1]
    sol = "\n".join(lines)
    return "\n".join([
        SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {difficulty}",
        f"  - 합산 항목 수: {n_items}",
        "[STEP 1] 주어진 조건",
        "  - 문제에 제시된 환산표만 사용한다(사전 지식 불필요).",
        "  - 각 항목을 기준 단위로 환산하고 배수를 적용한 뒤, 부호(더함/뺌)대로 합산한다.",
        "[STEP 2] 풀이 전개",
        "  · 요약: 항목별 기준단위 환산 → 배수 적용 → 부호대로 합산",
        "[환산]",
        sol,
        "[STEP 3] 답·검산",
        f"  - 최종 답: {answer}",
        f"  - 검산: 각 항목의 환산값 × 배수를 부호대로 합산하면 {total}{base} 이다.",
    ])


def _labels(n):
    if n <= len(_LABELS):
        return _LABELS[:n]
    return _LABELS + [f'제{i}호' for i in range(len(_LABELS) + 1, n + 1)]


def _build_ratios(rng, depth, ratio_mode):
    """연쇄 환산비(인접 단위 간) depth-1개를 난이도에 맞게 생성."""
    n = depth - 1
    if ratio_mode == 'decimal':
        return [10] * n
    if ratio_mode == 'semi':
        ratios = [10] * n
        ratios[rng.randrange(n)] = rng.choice([12, 16])
        return ratios
    # ugly: 매 단계 서로 다른 소수성 비율
    return rng.sample(UGLY_RATIOS, n)


def _cumulative_values(ratios):
    """기준 단위(=1)부터 각 단위의 기준단위 환산값 누적곱."""
    values = [1]
    for r in ratios:
        values.append(values[-1] * r)
    return values


def _make_quantity(rng, depth, terms, cmax):
    """단위 인덱스(0..depth-1)에서 k개 항을 골라 {level: count} 생성."""
    lo, hi = terms
    k = rng.randint(lo, min(hi, depth))
    k = max(2, k)
    chosen = rng.sample(range(depth), k)
    return {lvl: rng.randint(1, cmax) for lvl in chosen}


def _base_total(values, q):
    return sum(cnt * values[lvl] for lvl, cnt in q.items())


def _render_quantity(names, q):
    return ' '.join(f"{q[lvl]}{names[lvl]}" for lvl in sorted(q, reverse=True))


def generate_puzzle(difficulty='easy', seed=None):
    cfg = _CONFIG[difficulty]
    rng = random.Random(seed if seed is not None else random.randint(0, 10**9))

    for _ in range(400):
        system = rng.choice(list(UNIT_SYSTEMS.keys()))
        names = UNIT_SYSTEMS[system]
        depth = min(cfg['depth'], len(names))
        ratios = _build_ratios(rng, depth, cfg['ratio_mode'])
        values = _cumulative_values(ratios)
        base = names[0]

        # 항목 생성: 각 항목 = (수량, 부호, 배수)
        items = []
        for _i in range(cfg['n_quant']):
            q = _make_quantity(rng, depth, cfg['terms'], cfg['cmax'])
            scalar = rng.randint(*cfg['scalar_range']) if cfg['scalar'] else 1
            sign = -1 if (cfg['subtract'] and rng.random() < cfg['subtract_p']) else 1
            items.append({'q': q, 'scalar': scalar, 'sign': sign})

        # 각 항목 1개라도 단위 2개 미만이면 배제
        if any(len(it['q']) < 2 for it in items):
            continue

        signed_vals = [it['sign'] * it['scalar'] * _base_total(values, it['q']) for it in items]
        total = sum(signed_vals)

        # 부호 혼합 시 최종 합이 양의 정수이고 너무 시시하지 않도록
        if total <= 0:
            continue
        if cfg['subtract'] and not any(it['sign'] < 0 for it in items):
            continue  # 뺌이 최소 1회는 등장하도록

        answer = f"{total}{base}"
        table = '\n'.join(f"- 1{names[i]} = {ratios[i-1]}{names[i-1]}" for i in range(1, depth))

        simple = (cfg['n_quant'] == 1 and not cfg['scalar'] and not cfg['subtract'])
        if simple:
            body = (f"다음 양은 모두 몇 {base}입니까?\n\n"
                    f"  {_render_quantity(names, items[0]['q'])}")
            sol = f"{_render_quantity(names, items[0]['q'])}\n= {total}{base}"
            note = ""
        else:
            labels = _labels(cfg['n_quant'])
            lines = []
            sol_lines = []
            for i, it in enumerate(items):
                op = '뺌' if it['sign'] < 0 else '더함'
                lines.append(f"  {labels[i]} [{op}, {it['scalar']}배]: {_render_quantity(names, it['q'])}")
                bt = _base_total(values, it['q'])
                sv = it['sign'] * it['scalar'] * bt
                sol_lines.append(
                    f"{labels[i]}: {_render_quantity(names, it['q'])} = {bt}{base}"
                    f" → {it['scalar']}배 = {it['scalar']*bt}{base} ({op})")
            body = ("다음 각 항목을 환산표에 따라 '" + base + "'(으)로 바꾼 뒤, "
                    "[더함]은 더하고 [뺌]은 빼서 최종 합을 구하세요.\n\n"
                    + '\n'.join(lines))
            sol = '\n'.join(sol_lines) + f"\n최종 합 = {total}{base}"
            note = ("\n표기: 'k배'는 그 항목의 환산값에 곱하는 수, "
                    "[더함]/[뺌]은 최종 합에 더할지 뺄지를 뜻합니다.")

        question = f"""아래 환산표를 보고 물음에 답하세요. (제시된 표만 사용하세요)

[{system} 환산표]
{table}{note}

[문제]
{body}

풀이 마지막 줄에 `Answer: N{base}` 형식으로 답하세요."""

        return {
            'question': question,
            'answer': answer,
            'solution': _build_solution(sol, difficulty, answer, len(items), total, base),
            'difficulty': difficulty,
            'meta': {
                'system': system, 'total': total, 'base': base,
                'depth': depth, 'ratios': ratios, 'n_quant': cfg['n_quant'],
            },
        }

    return None


TASK = 'korean_units_ko'


def create_dataset_files(num_questions=100):
    from collections import Counter
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    difficulties = ['easy', 'medium', 'hard']
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    answer_cap = max(4, num_questions // 3)

    total = 0
    for diff in difficulties:
        rows = []
        seen = set()
        ans_count = Counter()
        s = 0
        while len(rows) < num_questions and s < num_questions * 400 + 4000:
            item = generate_puzzle(diff, seed=s)
            s += 1
            if item is None:
                continue
            key = item['question']
            if key in seen:
                continue
            if ans_count[item['answer']] >= answer_cap:
                continue
            seen.add(key)
            ans_count[item['answer']] += 1
            idx = len(rows)
            rows.append({
                'id': f'{TASK}_{diff}_{idx:04d}',
                'difficulty': diff,
                'question': item['question'],
                'answer': item['answer'],
                'solution': item['solution'],
            })

        path = data_dir / f'{TASK}_{diff}.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        total += len(rows)
        print(f"[{diff}] {len(rows)}개 생성 → {path}")

    print(f"총 {total}개 생성 완료")
    return total


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='한국 전통 단위 환산 퍼즐 생성기')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        for diff in ['easy', 'medium', 'hard']:
            p = generate_puzzle(diff, seed=42)
            print(f"\n{'='*60}\n[{diff.upper()}]")
            print(p['question'])
            print(f"\n정답: {p['answer']}")
            print(f"풀이:\n{p['solution']}")
    else:
        create_dataset_files(args.num)

"""Hangul-based Cipher Puzzle Generator
[진행도] ☑ 완료 / ☐ 미완성
[파일명] hangul_cipher.py
[목적] 한글의 자모 구조를 활용한 고난도 암호 퍼즐 생성
"""

import random
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import math
import itertools

# ============================================================================
# 난이도 설정 - 한글 자모 구조 활용
# ============================================================================

DIFFICULTY_CONFIG = {
    "LEVEL_0": {
        "name": "easy",
        "cipher_stack": ["cho_shift", "jung_sub", "reverse", "cho_shift", "jung_sub", "cho_shift"],
        "keyword_logic": "positional",
        "hint_count": 0,
        "answer_length": (8, 8),
        "description": "Easy (Target ~75%): Cho + Jung + Reverse + Cho + Jung + Cho / Positional / 8-char Real Korean Words"
    },
    "LEVEL_1": {
        "name": "medium",
        "cipher_stack": ["cho_shift", "jung_sub", "jong_shift", "reverse", "cho_shift", "jung_sub", "reverse", "jong_shift", "cho_shift", "jung_sub", "jong_shift", "reverse"],
        "keyword_logic": "positional",
        "hint_count": 0,
        "answer_length": (7, 9),
        "description": "Medium (Target ~50%): 12-layer Jamo Stack / Positional / Real Korean Words"
    },
    "LEVEL_2": {
        "name": "hard",
        "cipher_stack": ["cho_shift", "jung_sub", "jong_shift", "reverse", "cho_shift", "jung_sub", "reverse", "jong_shift", "cho_shift", "jung_sub", "jong_shift", "reverse"],
        "keyword_logic": "extraction",
        "hint_count": 0,
        "answer_length": (8, 10),
        "description": "Hard (Target ~25%): 12-layer Jamo Stack / Extraction / Real Korean Words"
    }
}

# ============================================================================
# 한글 처리 시스템
# ============================================================================

class HangulCipherSystem:
    def __init__(self):
        self.CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.BASE = 0xAC00

    def decompose(self, char: str) -> Tuple[int, int, int]:
        if not char or not (0xAC00 <= ord(char) <= 0xD7A3):
            return -1, -1, -1
        code = ord(char) - self.BASE
        cho = code // (21 * 28)
        jung = (code % (21 * 28)) // 28
        jong = code % 28
        return cho, jung, jong

    def compose(self, cho: int, jung: int, jong: int) -> str:
        if cho < 0 or jung < 0: return ""
        code = self.BASE + (cho * 21 * 28) + (jung * 28) + jong
        return chr(code)

    def cho_shift_encrypt(self, text: str, keyword: str) -> str:
        result = []
        for i, char in enumerate(text):
            key_char = keyword[i % len(keyword)]
            k_cho, _, _ = self.decompose(key_char)
            shift = k_cho if k_cho >= 0 else ord(key_char) % 19
            
            c_cho, c_jung, c_jong = self.decompose(char)
            if c_cho >= 0:
                new_cho = (c_cho + shift) % 19
                result.append(self.compose(new_cho, c_jung, c_jong))
            else:
                result.append(char)
        return "".join(result)

    def jung_sub_encrypt(self, text: str, keyword: str) -> str:
        # Generate substitution table based on keyword's vowels
        keyword_jungs = []
        for char in keyword:
            _, jung, _ = self.decompose(char)
            if jung >= 0 and jung not in keyword_jungs:
                keyword_jungs.append(jung)
        
        mapping = keyword_jungs.copy()
        for i in range(21):
            if i not in mapping:
                mapping.append(i)
        
        result = []
        for char in text:
            cho, jung, jong = self.decompose(char)
            if cho >= 0:
                result.append(self.compose(cho, mapping[jung], jong))
            else:
                result.append(char)
        return "".join(result)

    def jong_shift_encrypt(self, text: str, keyword: str) -> str:
        result = []
        for i, char in enumerate(text):
            key_char = keyword[i % len(keyword)]
            k_cho, _, k_jong = self.decompose(key_char)
            shift = k_jong if k_jong > 0 else (k_cho + 1 if k_cho >= 0 else ord(key_char) % 28)

            c_cho, c_jung, c_jong = self.decompose(char)
            if c_cho >= 0:
                new_jong = (c_jong + shift) % 28
                result.append(self.compose(c_cho, c_jung, new_jong))
            else:
                result.append(char)
        return "".join(result)

# ============================================================================
# 가상 로그 생성기
# ============================================================================

class KoreanMissionLogGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.statuses = ["심각", "안정", "동기화", "잠금", "과부하"]
        self.targets = ["위성", "데이터베이스", "그리드", "단말기", "코어"]
        self.actions = ["오프라인", "준비완료", "대기", "침입됨"]
        self.key_labels = ["주요키", "인증코드", "시드값", "벡터값", "암호키", "접속토큰"]

    def generate_log(self) -> Tuple[str, str]:
        keyword = self.rng.choice(["하늘", "바다", "나무", "구름", "태양", "달빛", "지도", "열쇠", "비밀"])
        label = self.rng.choice(self.key_labels)
        
        sentences = [
            f"시스템 보고서 ID {self.rng.randint(100, 999)}: 상태 {self.rng.choice(self.statuses)}.",
            f"대상 {self.rng.choice(self.targets)} 상태는 {self.rng.choice(self.actions)}.",
            f"암호화 {label} 설정 단계에서 키워드 {keyword} 부수어짐 없이 적용됨."
        ]
        self.rng.shuffle(sentences)
        return " ".join(sentences), keyword

# ============================================================================
# Guided-distillation style solution (teacher trace)
# ============================================================================

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_cipher_ko_solution(
    config: Dict,
    process: List[str],
    answer: str,
    keyword: str,
    encrypted: str,
    kw_logic: str,
    kw_instruction: str,
    pos_for_solution: int = None,
) -> str:
    """SFT용: 메타 → 키워드 → 역파이프라인 → 검산."""
    stack = config["cipher_stack"]
    lines = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {config['name']}",
        f"  - 암호문: '{encrypted}' (길이 {len(encrypted)}자)",
        f"  - 평문(정답): {answer}",
        f"  - 암호화 적용 순서(평문→암호문): {' → '.join(process)}",
        "  - 복호화: **마지막에 적용된 변환부터** 역순으로 역연산을 곱한다.",
        "[STEP 1] 주어진 조건 (키워드·지문 규칙)",
        f"  - 지문 규칙: {kw_instruction}",
    ]
    if kw_logic == "positional" and pos_for_solution is not None:
        lines.append(
            f"  - 절차: 로그에서 구두점(., :, ,) 제거 → 공백 단위 토큰화 → "
            f"{pos_for_solution}번째 단어 = '{keyword}'")
    elif kw_logic == "extraction":
        lines.append(
            "  - 절차: 라벨(주요키·인증코드·시드값·벡터값·암호키·접속토큰 등) 바로 뒤 "
            "토큰이 키워드.")
    else:
        lines.append(f"  - 키워드는 지문에 명시: '{keyword}'")

    rev = list(reversed(stack))
    _cipher_op_name_ko = {
        "cho_shift": "초성 시프트⁻¹",
        "jung_sub": "중성 치환⁻¹",
        "jong_shift": "종성 시프트⁻¹",
        "reverse": "문자열 뒤집기⁻¹",
    }
    decrypt_pipeline = " → ".join(_cipher_op_name_ko.get(s, s) for s in rev)
    lines.append("[STEP 2] 풀이 전개 (암호문 → 평문, 역연산)")
    lines.append(
        f"  · 요약: 스택 {len(stack)}단 · 키워드 '{keyword}' · "
        f"복호 파이프라인: {decrypt_pipeline} · SEG {len(rev)}개"
    )
    for i, st in enumerate(rev, 1):
        if st == "cho_shift":
            lines.append(
                f"    [SEG {i}] 초성 시프트 역연산: 키워드 '{keyword}'의 각 글자 초성 인덱스를 "
                f"키로 쓰되, 암호화 때 더했던 만큼 **빼서** mod 19로 초성 복원 "
                f"(비한글·공백은 그대로).")
        elif st == "jung_sub":
            lines.append(
                f"    [SEG {i}] 중성 치환 역연산: 키워드에서 등장 순서대로 모음 인덱스를 앞에 두고 "
                f"나머지 21개 중성 인덱스를 채운 치환표의 **역치환**으로 중성 복원.")
        elif st == "jong_shift":
            lines.append(
                f"    [SEG {i}] 종성 시프트 역연산: 키워드 '{keyword}'의 각 글자 종성 인덱스를 "
                f"우선 키로 쓰고, 종성이 없으면 초성 인덱스+1을 키로 써서 더했던 만큼 **빼서** "
                f"mod 28로 종성 복원.")
        elif st == "reverse":
            lines.append(f"    [SEG {i}] 문자열 **전체 뒤집기**로 역순 단계 해제.")
        else:
            lines.append(f"    [SEG {i}] {st} 역연산 적용.")

    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답: '{answer}' (공백 없는 한글)",
        f"  - 복호 결과가 '{answer}'와 일치하는지 확인.",
        "  - 예제 행(문제 본문)을 같은 키·같은 스택으로 암호화해 암호문과 규칙이 맞는지 대조.",
    ])
    return "\n".join(lines)


# ============================================================================
# Generator
# ============================================================================

class HangulCipherGenerator:
    def __init__(self):
        self.cipher = HangulCipherSystem()
        self.answer_pools = self.build_answer_pools()
        self.hint_pool = ["사과", "바다", "친구", "공부", "사랑", "지도", "구름", "나무", "시계", "전화"]

    def build_answer_pools(self) -> Dict[str, List[str]]:
        """의미 있는 복합명사 후보를 크게 만들어 정답 중복을 줄인다."""
        easy_seed = [
            "대한민국", "정보보안", "미래기술", "평화통일", "민주주의",
            "산업혁명", "자연과학", "문화유산", "교통신호", "기상예보",
            "해양생물", "우주탐사", "전통시장", "도서관", "자전거",
            "컴퓨터", "운동회", "박물관", "비행기", "지하철",
        ]
        easy = easy_seed + [
            left + right
            for left, right in itertools.product(
                ["국가", "지역", "학교", "환경", "해양", "우주", "문화", "교통", "기상", "전통", "공공", "사회", "자연", "생활", "도시", "농업", "의료", "정보", "산업", "교육"],
                ["보안", "통신", "연구", "관리", "탐사", "예보", "시장", "도서", "신호", "자원", "정책", "기술", "지도", "관측", "분석", "지원", "훈련", "기록"],
            )
        ] + [
            left + mid + right
            for left, mid, right in itertools.product(
                ["지역", "학교", "환경", "해양", "우주", "문화", "교통", "기상", "전통", "공공", "사회", "자연", "생활", "도시", "농업", "의료", "정보", "산업", "교육"],
                ["보안", "통신", "연구", "관리", "탐사", "예보", "시장", "자원", "정책", "기술", "관측", "분석", "지원", "훈련"],
                ["계획", "센터", "체계", "사업", "자료", "기록", "모델", "지도", "절차", "목표"],
            )
        ] + [
            left + mid_a + mid_b + right
            for left, mid_a, mid_b, right in itertools.product(
                ["지역", "학교", "환경", "해양", "우주", "문화", "교통", "기상", "전통", "공공", "사회", "자연", "생활", "도시", "농업", "의료", "정보", "산업", "교육"],
                ["보안", "통신", "연구", "관리", "탐사", "예보", "시장", "자원", "정책", "기술", "관측", "분석", "지원", "훈련"],
                ["평가", "분석", "검증", "관리", "지원", "기록", "탐색", "예측", "보호", "복원"],
                ["계획", "센터", "체계", "사업", "자료", "기록", "모델", "지도", "절차", "목표"],
            )
        ]

        medium_seed = [
            "데이터베이스", "네트워크보안", "인공지능연구", "클라우드서버", "디지털신호",
            "암호해독", "위성통신", "자동제어", "분산처리", "시스템분석",
            "정보검색", "기계번역", "패턴인식", "음성합성", "문서분류",
            "경로탐색", "확률모델", "논리회로", "자료구조", "운영체제",
        ]
        medium = medium_seed + [
            left + right
            for left, right in itertools.product(
                ["데이터", "네트워크", "인공지능", "클라우드", "디지털", "암호", "위성", "자동", "분산", "시스템", "정보", "기계", "패턴", "음성", "문서", "경로", "확률", "논리", "자료", "보안"],
                ["분석", "보안", "검색", "처리", "제어", "모델", "회로", "구조", "서버", "통신", "해독", "분류", "탐색", "합성", "학습"],
            )
        ]

        hard_seed = [
            "양자암호통신", "다중계층보안", "비선형최적화", "확률그래프모델", "분산합의프로토콜",
            "지식그래프추론", "형태소분석엔진", "신경망압축기법", "대규모언어모델", "검색증강생성",
            "동형암호연산", "차분프라이버시", "블록체인검증", "자동정리증명", "복합추론평가",
            "기호논리시스템", "문맥인식검색", "계층적계획수립", "다국어평가셋", "절차기억검증",
        ]
        hard = hard_seed + [
            left + right
            for left, right in itertools.product(
                ["양자암호", "다중계층", "비선형", "확률그래프", "분산합의", "지식그래프", "형태소분석", "신경망압축", "대규모언어", "검색증강", "동형암호", "차분보호", "블록체인", "자동정리", "복합추론", "기호논리", "문맥인식", "계층계획", "다국어평가", "절차기억"],
                ["통신", "보안", "최적화", "모델", "추론", "엔진", "기법", "생성", "연산", "검증", "증명", "평가", "검색", "수립", "시스템"],
            )
        ]
        return {
            "easy": self.filter_pool(easy, 8, 8),
            "medium": self.filter_pool(medium, 4, 6),
            "hard": self.filter_pool(hard, 6, 8),
        }

    def filter_pool(self, words: List[str], min_len: int, max_len: int) -> List[str]:
        unique_words = list(dict.fromkeys(word.replace(" ", "") for word in words))
        return [word for word in unique_words if min_len <= len(word) <= max_len]

    def generate_answer(self, rng: random.Random, config: Dict) -> str:
        return rng.choice(self.answer_pools[config["name"]]).replace(" ", "")

    def generate_problem(self, config: Dict, seed: int = None) -> Dict:
        rng = random.Random(seed)
        log_gen = KoreanMissionLogGenerator(rng)
        
        answer = self.generate_answer(rng, config)
        
        log_text, keyword = log_gen.generate_log()
        
        current_text = answer
        process = []
        for stage in config["cipher_stack"]:
            if stage == "cho_shift":
                current_text = self.cipher.cho_shift_encrypt(current_text, keyword)
                process.append(f"초성 시프트(키={keyword})")
            elif stage == "jung_sub":
                current_text = self.cipher.jung_sub_encrypt(current_text, keyword)
                process.append(f"중성 치환(키={keyword})")
            elif stage == "jong_shift":
                current_text = self.cipher.jong_shift_encrypt(current_text, keyword)
                process.append(f"종성 시프트(키={keyword})")
            elif stage == "reverse":
                current_text = current_text[::-1]
                process.append("역순")
        
        encrypted = current_text
        
        # Build hints
        hint_examples = []
        for _ in range(config["hint_count"]):
            test_word = rng.choice(self.hint_pool)
            temp = test_word
            for stage in config["cipher_stack"]:
                if stage == "cho_shift": temp = self.cipher.cho_shift_encrypt(temp, keyword)
                elif stage == "jung_sub": temp = self.cipher.jung_sub_encrypt(temp, keyword)
                elif stage == "jong_shift": temp = self.cipher.jong_shift_encrypt(temp, keyword)
                elif stage == "reverse": temp = temp[::-1]
            hint_examples.append(f"  - {test_word} -> {temp}")

        kw_logic = config["keyword_logic"]
        pos_for_solution = None
        if kw_logic == "direct":
            kw_instruction = f"암호화 키워드는 '{keyword}'입니다."
        elif kw_logic == "positional":
            clean_log = log_text.replace(".", "").replace(":", "").replace(",", "")
            words = clean_log.split()
            pos_for_solution = words.index(keyword) + 1
            kw_instruction = (
                f"암호화 키워드는 아래 로그 지문의 {pos_for_solution}번째 단어입니다. "
                f"(문장 부호 제외)")
        else:
            kw_instruction = (
                "암호화 키워드는 로그 지문 내에 숨겨져 있습니다. '주요키', '인증코드', "
                "'시드값' 등의 라벨 다음에 오는 단어가 키워드입니다.")

        algo_details = []
        if "cho_shift" in config["cipher_stack"]:
            algo_details.append(
                "- CHO_SHIFT: 한글 음절을 초성/중성/종성으로 분해합니다. "
                "초성 순서는 ㄱ, ㄲ, ㄴ, ㄷ, ㄸ, ㄹ, ㅁ, ㅂ, ㅃ, ㅅ, ㅆ, ㅇ, ㅈ, ㅉ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ입니다. "
                "키워드를 반복해 각 위치의 키 글자 초성 인덱스만큼 평문 초성을 더해(mod 19) 암호화합니다."
            )
        if "jung_sub" in config["cipher_stack"]:
            algo_details.append(
                "- JUNG_SUB: 중성 순서는 ㅏ, ㅐ, ㅑ, ㅒ, ㅓ, ㅔ, ㅕ, ㅖ, ㅗ, ㅘ, ㅙ, ㅚ, ㅛ, ㅜ, ㅝ, ㅞ, ㅟ, ㅠ, ㅡ, ㅢ, ㅣ입니다. "
                "키워드에서 처음 등장하는 중성 인덱스를 앞에 놓고, 나머지 중성 인덱스를 순서대로 붙여 치환표를 만듭니다. "
                "암호화는 원래 중성 인덱스 j를 치환표[j]로 바꿉니다."
            )
        if "jong_shift" in config["cipher_stack"]:
            algo_details.append(
                "- JONG_SHIFT: 종성 순서는 없음, ㄱ, ㄲ, ㄳ, ㄴ, ㄵ, ㄶ, ㄷ, ㄹ, ㄺ, ㄻ, ㄼ, ㄽ, ㄾ, ㄿ, ㅀ, ㅁ, ㅂ, ㅄ, ㅅ, ㅆ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ입니다. "
                "키 글자에 종성이 있으면 그 종성 인덱스를, 없으면 키 글자의 초성 인덱스+1을 사용합니다. "
                "키워드를 반복해 각 위치의 종성 인덱스에 키 값을 더해(mod 28) 암호화합니다."
            )
        if "reverse" in config["cipher_stack"]:
            algo_details.append("- REVERSE: 문자열 전체 순서를 뒤집습니다. 복호화할 때도 다시 뒤집으면 됩니다.")

        problem_text = f"--- [복구된 미션 로그] ---\n{log_text}\n---------------------------\n\n"
        problem_text += f"암호문: '{encrypted}'\n\n"
        problem_text += f"암호화 가이드:\n1. {kw_instruction}\n2. 적용된 알고리즘: {' -> '.join(config['cipher_stack']).upper()}\n"
        if algo_details:
            problem_text += "알고리즘 정의:\n" + "\n".join(algo_details) + "\n"
        
        if hint_examples:
            problem_text += "\n예제:\n" + "\n".join(hint_examples) + "\n"
        
        problem_text += "\n복호화된 원문을 입력하세요 (공백 없이 한글로)."

        solution = _build_cipher_ko_solution(
            config=config,
            process=process,
            answer=answer,
            keyword=keyword,
            encrypted=encrypted,
            kw_logic=kw_logic,
            kw_instruction=kw_instruction,
            pos_for_solution=pos_for_solution,
        )

        return {
            "difficulty": config["name"],
            "problem": problem_text,
            "answer": answer,
            "solution": solution,
        }

def create_hangul_dataset(num_per_level: int = 3):
    print(f"Hangul-based Cipher 문제 생성 중...")
    print(f"난이도별 {num_per_level}개씩 생성")
    print("="*70)

    generator = HangulCipherGenerator()
    all_problems = []

    # 쉬운 난이도부터 어려운 순서로 생성 (LEVEL_0: easy -> LEVEL_2: hard)
    for level_key in sorted(DIFFICULTY_CONFIG.keys()):
        config = DIFFICULTY_CONFIG[level_key]
        difficulty = config["name"]
        used_answers = set()
        
        print(f"\n[{difficulty}] {config['description']}")

        for i in range(num_per_level):
            problem = None
            for attempt in range(1000):
                seed = 5000 + len(all_problems) + attempt * 10000
                candidate = generator.generate_problem(config, seed)
                if candidate["answer"] not in used_answers or len(used_answers) >= len(generator.answer_pools[difficulty]):
                    problem = candidate
                    break
            if problem is None:
                raise RuntimeError(f"cipher_ko_{difficulty} 문제를 생성하지 못했습니다.")
            used_answers.add(problem["answer"])
            all_problems.append({
                "id": f"cipher_ko_{difficulty}_{i:04d}",
                "question": problem["problem"],
                "answer": problem["answer"],
                "solution": problem["solution"],
                "difficulty": difficulty,
            })

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("cipher_ko", all_problems)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Hangul Cipher Puzzles')
    parser.add_argument('--num', type=int, default=2, help='Number of puzzles per difficulty level')
    args = parser.parse_args()
    
    # 각 난이도별 n개씩 생성
    create_hangul_dataset(num_per_level=args.num)

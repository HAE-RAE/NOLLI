"""Cipher puzzles v260112 - Intermediate Layer Added"""

import random
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import math

# ============================================================================
# Difficulty settings - algorithmic complexity + information asymmetry
# ============================================================================

DIFFICULTY_CONFIG = {
    "LEVEL_0": {
        "name": "easy",
        "cipher_stack": ["vigenere"],
        "keyword_logic": "direct",
        "hint_count": 1,
        "answer_length": (20, 24),
        "show_algorithm_details": True,
        "show_decryption_order_hint": False,
        "description": "Easy (Target ~75%): Vigenere / Direct / Algorithm Detail / Slightly Shorter Text"
    },
    "LEVEL_1": {
        "name": "medium",
        "cipher_stack": ["vigenere"],
        "keyword_logic": "extraction",
        "hint_count": 2,
        "answer_length": (20, 24),
        "show_algorithm_details": True,
        "show_decryption_order_hint": True,
        "description": "Medium (Target ~50%): Vigenere / Extraction / 2 Hints"
    },
    "LEVEL_2": {
        "name": "hard",
        "cipher_stack": ["transposition", "vigenere"],
        "keyword_logic": "positional",
        "hint_count": 10,
        "answer_length": (6, 10),
        "show_algorithm_details": True,
        "show_decryption_order_hint": True,
        "description": "Hard (Target ~25%): Transposition + Vigenere / Positional / 10 Hints"
    }
}

# ============================================================================
# Synthetic context generator (blocks external knowledge)
# ============================================================================

class MissionLogGenerator:
    """Fully randomized prompt context generator."""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.components = {
            "id": lambda: str(self.rng.randint(100, 9999)),
            "status": lambda: self.rng.choice(["CRITICAL", "STABLE", "SYNCING", "LOCKED", "OVERLOADED"]),
            "target": lambda: self.rng.choice(["SATELLITE", "DATABASE", "GRID", "TERMINAL", "CORE"]),
            "action": lambda: self.rng.choice(["OFFLINE", "READY", "STANDBY", "BREACHED"]),
            "coord": lambda: f"{self.rng.randint(0, 180)}.{self.rng.randint(0, 99)}",
            "hex": lambda: hex(self.rng.randint(4096, 65535)).upper()[2:],
        }
        self.key_labels = ["PRIMARY KEY", "AUTH CODE", "SEED", "VECTOR"]

    def generate_log(self) -> Tuple[str, str]:
        """Generate the mission log and random keyword."""
        keyword = self.rng.choice(["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL", "INDIA"])
        label = self.rng.choice(self.key_labels)
        
        sentences = [
            f"SYSTEM REPORT ID {self.components['id']()}: STATUS {self.components['status']()}.",
            f"TARGET {self.components['target']()} IS {self.components['action']()}.",
            f"COORDINATES SET TO {self.components['coord']()} | SECTOR {self.components['hex']()}.",
            f"ENCRYPTION {label} IS DETERMINED AS {keyword}."
        ]
        self.rng.shuffle(sentences)
        return " ".join(sentences), keyword

# ============================================================================
# Advanced Cipher System
# ============================================================================

class AdvancedCipher:
    def __init__(self):
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def vigenere_encrypt(self, text: str, keyword: str) -> str:
        result = []
        keyword_repeated = (keyword * (len(text) // len(keyword) + 1))[:len(text)]
        for i, char in enumerate(text.upper()):
            if char in self.alphabet:
                new_idx = (self.alphabet.index(char) + self.alphabet.index(keyword_repeated[i])) % 26
                result.append(self.alphabet[new_idx])
            else: result.append(char)
        return ''.join(result)

    def substitution_encrypt(self, text: str, keyword: str) -> str:
        seen = set()
        key_chars = []
        for char in keyword.upper():
            if char in self.alphabet and char not in seen:
                key_chars.append(char); seen.add(char)
        for char in self.alphabet:
            if char not in seen: key_chars.append(char)
        key = ''.join(key_chars)
        
        result = []
        for char in text.upper():
            if char in self.alphabet:
                result.append(key[self.alphabet.index(char)])
            else: result.append(char)
        return ''.join(result)

    def playfair_encrypt(self, text: str, keyword: str) -> str:
        """Playfair cipher (digraph substitution)."""
        # Build a 5x5 matrix (J is merged with I).
        matrix = []
        seen = set(['J'])
        chars = []
        for char in (keyword.upper() + self.alphabet):
            if char not in seen and char in self.alphabet:
                chars.append(char)
                seen.add(char)
        
        matrix = [chars[i:i+5] for i in range(0, 25, 5)]
        
        # Preprocess text: even length and repeated-letter handling.
        text = text.upper().replace('J', 'I')
        processed = ""
        i = 0
        while i < len(text):
            a = text[i]
            b = text[i+1] if i+1 < len(text) else 'X'
            if a == b:
                processed += a + 'X'
                i += 1
            else:
                processed += a + b
                i += 2
        
        def find_pos(char):
            for r in range(5):
                for c in range(5):
                    if matrix[r][c] == char: return r, c
            return 0, 0

        result = ""
        for i in range(0, len(processed), 2):
            r1, c1 = find_pos(processed[i])
            r2, c2 = find_pos(processed[i+1])
            
            if r1 == r2: # Same row
                result += matrix[r1][(c1+1)%5] + matrix[r2][(c2+1)%5]
            elif c1 == c2: # Same column
                result += matrix[(r1+1)%5][c1] + matrix[(r2+1)%5][c2]
            else: # Rectangle
                result += matrix[r1][c2] + matrix[r2][c1]
        return result

    def columnar_transpose(self, text: str, key: str) -> str:
        # Keep the existing columnar transposition logic.
        text = text.replace(" ", "")
        key_order = sorted(range(len(key)), key=lambda k: key[k])
        cols = len(key)
        rows = math.ceil(len(text) / cols)
        padded_text = text.ljust(rows * cols, 'X')
        grid = [padded_text[i:i+cols] for i in range(0, len(padded_text), cols)]
        result = ""
        for k_idx in key_order:
            for r in range(rows):
                result += grid[r][k_idx]
        return result

# ============================================================================
# Guided-distillation style solution (teacher trace)
# ============================================================================

SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


def _build_cipher_en_solution(
    config: Dict,
    process_log: List[str],
    answer: str,
    keyword: str,
    encrypted: str,
    kw_logic: str,
    kw_instruction: str,
    pos_for_solution: int = None,
    log_id: int = 0,
) -> str:
    """Structured English solution for SFT guided distillation."""
    stack = config["cipher_stack"]
    lines = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {config['name']}",
        f"  - Ciphertext: '{encrypted}' (length {len(encrypted)})",
        f"  - Plaintext (answer): {answer}",
        f"  - Encryption pipeline (plaintext → ciphertext): "
        f"{' → '.join(process_log)}",
        "  - Decryption: apply the **inverse** of each layer in **reverse order** "
        "(undo the last encryption step first).",
        "[STEP 1] Given (keyword + log rules)",
        f"  - Rule from prompt: {kw_instruction}",
    ]
    if kw_logic == "positional" and pos_for_solution is not None:
        lines.append(
            f"  - Procedure: strip punctuation (., :, ,) → split on whitespace → "
            f"take the {pos_for_solution}-th token = '{keyword}'.")
    elif kw_logic == "extraction":
        lines.append(
            "  - Procedure: the token immediately after a label such as "
            "'PRIMARY KEY', 'AUTH CODE', 'SEED', or 'VECTOR' is the keyword.")
    else:
        lines.append(f"  - Keyword is stated explicitly: '{keyword}'.")

    if "playfair" in stack:
        pf_key = keyword if log_id % 2 == 0 else keyword[::-1]
        lines.append(
            f"  - Playfair key rule: mission log numeric ID is {log_id} "
            f"({'even' if log_id % 2 == 0 else 'odd'}); use keyword "
            f"{'forward' if log_id % 2 == 0 else 'reversed'} → effective key "
            f"'{pf_key}'.")

    rev = list(reversed(stack))
    _cipher_op_name_en = {
        "vigenere": "Vigenère⁻¹",
        "playfair": "Playfair⁻¹",
        "transposition": "columnar transposition⁻¹",
        "reverse": "reverse⁻¹",
        "substitution": "keyed substitution⁻¹",
    }
    decrypt_pipeline = " -> ".join(_cipher_op_name_en.get(s, s) for s in rev)
    lines.append("[STEP 2] Worked solution (decrypt, ciphertext → plaintext)")
    lines.append(
        f"  · Summary: stack depth {len(stack)} · keyword '{keyword}' · "
        f"decrypt pipeline: {decrypt_pipeline} · {len(rev)} SEGs"
    )
    for i, st in enumerate(rev, 1):
        if st == "vigenere":
            lines.append(
                f"    [SEG {i}] Vigenère **decrypt**: repeat '{keyword}'; subtract each key "
                "letter from ciphertext (mod 26, A=0).")
        elif st == "playfair":
            lines.append(
                f"    [SEG {i}] Playfair **decrypt**: build the 5×5 square from the effective "
                "key (I/J merged), undo digraph rules (same row → shift left, same column "
                "→ shift up, rectangle → swap columns back).")
        elif st == "transposition":
            lines.append(
                f"    [SEG {i}] Columnar transposition **decrypt**: split ciphertext into columns "
                f"ordered by sorting the keyword '{keyword}'; fill the grid and read "
                "rows left-to-right, then strip padding 'X' as needed.")
        elif st == "reverse":
            lines.append(f"    [SEG {i}] Reverse the string end-to-end.")
        elif st == "substitution":
            lines.append(
                f"    [SEG {i}] Substitution **decrypt**: build the keyed alphabet from "
                f"'{keyword}' then map ciphertext letters back to plaintext using the "
                "inverse mapping.")
        else:
            lines.append(f"    [SEG {i}] apply inverse of {st}.")

    lines.extend([
        "[STEP 3] Answer and verification",
        f"  - Final answer: '{answer}' (uppercase A–Z, no spaces).",
        f"  - Plaintext must match '{answer}'.",
        "  - Re-encrypt the sample pairs from the prompt with the same keyword and "
        "stack to verify the rules match the ciphertext.",
    ])
    return "\n".join(lines)


# ============================================================================
# Generator
# ============================================================================

class SelfContainedCipherGenerator:
    def __init__(self):
        self.cipher = AdvancedCipher()

    def generate_random_string(self, rng: random.Random, length: int, allow_j: bool = True) -> str:
        """Generate a purely random string with no semantic content."""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if allow_j else "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        return ''.join(rng.choice(alphabet) for _ in range(length))

    def generate_plaintext(self, rng: random.Random, config: Dict) -> str:
        """Generate a fair plaintext for the selected cipher stack.

        Playfair merges I/J and inserts X between repeated digraph letters, so
        hard-mode plaintext avoids those cases to keep the expected answer
        uniquely recoverable.
        """
        min_len, max_len = config.get("answer_length", (10, 14))
        length = rng.randint(min_len, max_len)

        if "playfair" not in config["cipher_stack"]:
            return self.generate_random_string(rng, length)

        if length % 2 == 1:
            length += 1
        chars = []
        alphabet = "ABCDEFGHIKLMNOPQRSTUVWYZ"  # no J, no X padding ambiguity
        while len(chars) < length:
            char = rng.choice(alphabet)
            if len(chars) % 2 == 1 and chars[-1] == char:
                continue
            chars.append(char)
        return ''.join(chars)

    def generate_problem(self, config: Dict, seed: int = None) -> Dict:
        rng = random.Random(seed)
        log_gen = MissionLogGenerator(rng)
        
        # 1. Generate a fully random answer with no semantic content.
        answer = self.generate_plaintext(rng, config)
        
        log_text, keyword = log_gen.generate_log()
        
        # 2. Apply encryption layers with rule-based branching.
        current_text = answer
        process_log = []
        
        # Extract the numeric ID from the mission log.
        log_id_match = re.search(r'SYSTEM REPORT ID (\d+)', log_text)
        log_id = int(log_id_match.group(1)) if log_id_match else int(re.search(r'\d+', log_text).group())
        
        for stage in config["cipher_stack"]:
            if stage == "vigenere":
                current_text = self.cipher.vigenere_encrypt(current_text, keyword)
                process_log.append(f"Vigenere(key={keyword})")
            elif stage == "playfair":
                # Branching rule: even log ID uses the keyword as-is; odd log ID reverses it.
                actual_key = keyword if log_id % 2 == 0 else keyword[::-1]
                current_text = self.cipher.playfair_encrypt(current_text, actual_key)
                process_log.append(f"Playfair(key={actual_key})")
            elif stage == "transposition":
                current_text = self.cipher.columnar_transpose(current_text, keyword)
                process_log.append(f"Columnar Transposition(key={keyword})")
            elif stage == "reverse":
                current_text = current_text[::-1]
                process_log.append("Reverse")
            elif stage == "substitution":
                current_text = self.cipher.substitution_encrypt(current_text, keyword)
                process_log.append(f"Substitution(key={keyword})")
        
        encrypted = current_text
        
        # 3. Generate hints using fully random strings.
        hint_examples = []
        num_hints = config["hint_count"]
        
        for _ in range(num_hints):
            # Hint strings are also meaningless 4-6 character strings.
            hint_config = {**config, "answer_length": (4, 6)}
            test_word = self.generate_plaintext(rng, hint_config)
            temp = test_word
            for stage in config["cipher_stack"]:
                if stage == "vigenere": temp = self.cipher.vigenere_encrypt(temp, keyword)
                elif stage == "playfair":
                    # Branching rule: even log ID uses the keyword as-is; odd log ID reverses it.
                    actual_key = keyword if log_id % 2 == 0 else keyword[::-1]
                    temp = self.cipher.playfair_encrypt(temp, actual_key)
                elif stage == "transposition": temp = self.cipher.columnar_transpose(temp, keyword)
                elif stage == "reverse": temp = temp[::-1]
                elif stage == "substitution": temp = self.cipher.substitution_encrypt(temp, keyword)
            hint_examples.append(f"  - {test_word} -> {temp}")

        # 4. Build the prompt.
        kw_logic = config["keyword_logic"]
        pos_for_solution = None
        if kw_logic == "direct":
            kw_instruction = f"The encryption keyword is '{keyword}'."
        elif kw_logic == "positional":
            # Remove punctuation, then split into word tokens.
            clean_log = log_text.replace(".", "").replace(":", "").replace(",", "")
            words = clean_log.split()
            try:
                pos_for_solution = words.index(keyword) + 1
                kw_instruction = (
                    f"The encryption keyword is the {pos_for_solution}-th word in the mission log below "
                    f"(excluding punctuation).")
            except ValueError:
                kw_instruction = f"The encryption keyword is '{keyword}'."
        else:  # extraction (Extreme)
            kw_instruction = (
                "The encryption keyword is hidden in the mission log. The word immediately after labels "
                "such as 'PRIMARY KEY', 'AUTH CODE', 'SEED', or 'VECTOR' is the keyword.")

        stack_desc = " -> ".join([s.upper() for s in config["cipher_stack"]])
        
        # Add conditional rule description.
        logic_hint = ""
        if "playfair" in config["cipher_stack"]:
            logic_hint = "\n[SPECIAL RULE]: When using the Playfair algorithm, reverse the keyword if the numeric log ID is odd; use it as-is if the log ID is even."

        # Add algorithm details for lower difficulties.
        algo_details = ""
        if config.get("show_algorithm_details", False):
            details = []
            for s in config["cipher_stack"]:
                if s == "substitution":
                    details.append("- SUBSTITUTION: Put the keyword's unique letters first, then append the remaining alphabet in order to build a 26-letter mapping.")
                elif s == "vigenere":
                    details.append("- VIGENERE: Repeat the keyword and add it to the plaintext letters (A=0, B=1...).")
                elif s == "transposition":
                    details.append("- TRANSPOSITION: This is columnar transposition; read columns vertically in the order produced by sorting the keyword letters alphabetically.")
                elif s == "playfair":
                    details.append("- PLAYFAIR: Build a 5x5 square with I/J merged, then substitute pairs of letters.")
                elif s == "reverse":
                    details.append("- REVERSE: Reverse the string order.")
            if details:
                algo_details = "\nAlgorithm details:\n" + "\n".join(details) + "\n"

        problem_text = f"--- [RECOVERED MISSION LOG] ---\n{log_text}\n-------------------------------\n\n"
        problem_text += f"Ciphertext: '{encrypted}'\n\n"
        problem_text += f"Encryption guide:\n1. {kw_instruction}\n2. Applied algorithms: {stack_desc}{logic_hint}\n{algo_details}"
        if config.get("show_decryption_order_hint", False):
            problem_text += "3. To decrypt, undo the algorithms in reverse order, starting from the last applied algorithm.\n"
        
        if hint_examples:
            problem_text += "\nExamples encrypted with the same keyword and algorithms:\n" + "\n".join(hint_examples) + "\n"
        
        problem_text += "\nEnter the decrypted plaintext (uppercase, no spaces)."

        answer_clean = answer.replace(" ", "")
        solution = _build_cipher_en_solution(
            config=config,
            process_log=process_log,
            answer=answer_clean,
            keyword=keyword,
            encrypted=encrypted,
            kw_logic=kw_logic,
            kw_instruction=kw_instruction,
            pos_for_solution=pos_for_solution,
            log_id=log_id,
        )

        return {
            "difficulty": config["name"],
            "problem": problem_text,
            "answer": answer_clean,
            "solution": solution,
        }

# ============================================================================
# Dataset generation and saving
# ============================================================================

def create_advanced_dataset(num_per_level: int = 2):
    """Generate a self-contained cipher dataset."""
    import pandas as pd
    import json

    print(f"Generating algorithm-focused cipher puzzles...")
    print(f"Generating {num_per_level} puzzles per difficulty")
    print("="*70)

    generator = SelfContainedCipherGenerator()
    all_problems = []

    # Generate in ascending difficulty order (LEVEL_0: EASY -> LEVEL_2: HARD).
    for level_key in sorted(DIFFICULTY_CONFIG.keys()):
        config = DIFFICULTY_CONFIG[level_key]
        difficulty = config["name"]

        print(f"\n[{difficulty}] {config['description']}")

        for i in range(num_per_level):
            # Use 4000-series seeds.
            seed = 4000 + len(all_problems)
            problem = generator.generate_problem(config, seed)

            all_problems.append({
                "id": f"cipher_en_{difficulty}_{i:04d}",
                "question": problem["problem"],
                "answer": problem["answer"],
                "solution": problem["solution"],
                "difficulty": difficulty,
            })

            print(f"  {i+1}. {problem['answer'][:15]}... generated")

    cols = ["id", "question", "answer", "solution", "difficulty"]
    df = pd.DataFrame([{k: p[k] for k in cols} for p in all_problems])

    import sys
    from pathlib import Path as _Path
    _gen = str(_Path(__file__).resolve().parent)
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from dataset_io import write_release_jsonl

    write_release_jsonl("cipher_en", all_problems)

    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Cipher Puzzles')
    parser.add_argument('--num', type=int, default=2, help='Number of puzzles per difficulty level')
    args = parser.parse_args()

    create_advanced_dataset(num_per_level=args.num)

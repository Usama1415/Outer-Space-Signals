from __future__ import annotations
import argparse
import json
import math
import random
import re
import string
from collections import Counter
from typing import Dict, Tuple, List

# --- Constants ---
ALPHA = string.ascii_uppercase
ALPHABET_SET = set(ALPHA)

# Classic English frequency order (highest→lowest)
ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
# Normalized English letter frequencies (rough; used for chi-squared & scoring)
ENGLISH_FREQ = {
    'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75,
    'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
    'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
    'P': 1.93, 'B': 1.49, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
    'Q': 0.10, 'Z': 0.07,
}

# Small, curated wordlist for scoring (kept compact to avoid extra files)
COMMON_WORDS = set(
    (
        # 1-letter
        "A I".split()
        +
        # 2-letter
        "OF TO IN IS IT AS AT ON BY HE WE OR IF DO ME MY UP AN GO NO US AM SO BE".split()
        +
        # 3-letter (very common)
        "THE AND FOR ARE BUT NOT YOU ALL ANY CAN HER WAS ONE OUR OUT DAY GET HAS HIM HIS HOW MAN NEW NOW OLD SEE TWO WAY WHO BOY DID ITS LET PUT SAY SHE TOO USE".split()
        +
        # Some 4–10 letter frequent words (general, not task-specific)
        "THIS THAT WITH FROM THEY WERE WHAT WHEN YOUR THERE THEIR ABOUT WHICH WOULD THESE OTHER COULD FIRST AFTER WHERE BEING BEFORE PEOPLE THROUGH LITTLE AGAINST SHOULD BETWEEN BECAUSE UNDER AROUND WITHOUT WHILE NEVER ALWAYS".split()
        +
        # Helpful domain-neutral words often present in long messages
        "VOLUME FREQUENCY FREQUENCIES LOCATION EMISSIONS REQUEST OUTPUT LEVEL REDUCTION RESPECTFULLY COMMUNICATING SPEAKER MISTAKEN SEVERAL ANOTHER ALREADY CLUSTER EXPLORATION DRAMATIC EXPERIENCE CONFUSION MEMBERS INTERNAL STABILIZERS IMPROVISED CONTAINERS EMOTIONAL THUNDERSTORM WATERFALL".split()
    )
)

# Frequent English bigrams and trigrams with weights (heuristic log-boosts)
BIGRAM_WEIGHTS = {
    # bigram: weight
    "TH": 2.2, "HE": 2.0, "IN": 1.5, "ER": 1.3, "AN": 1.2, "RE": 1.2,
    "ON": 1.0, "AT": 1.0, "EN": 1.0, "ND": 1.0, "TI": 0.9, "ES": 0.9,
    "OR": 0.9, "TE": 0.9, "OF": 1.0, "ED": 0.8, "IS": 0.8, "IT": 0.8,
    "AL": 0.7, "AR": 0.7, "ST": 0.8, "TO": 0.9, "NT": 0.7, "NG": 0.7,
    "SE": 0.7, "HA": 0.7, "AS": 0.7, "OU": 0.9, "IO": 0.8, "LE": 0.6,
    "VE": 0.6, "EA": 0.6, "RO": 0.6, "RI": 0.6, "NE": 0.6, "NO": 0.5,
    "QU": 1.2,
}
TRIGRAM_WEIGHTS = {
    "THE": 3.5, "ING": 2.8, "AND": 2.6, "HER": 1.6, "ERE": 1.2,
    "ENT": 1.5, "THA": 1.4, "NTH": 1.1, "WAS": 1.2, "ETH": 1.0,
    "FOR": 1.2, "TED": 0.9, "EST": 1.0, "ION": 1.1, "TIO": 1.2,
}

# --- Utilities ---
def load_signal(path: str) -> str:
    """Load and clean signal file (keep A–Z and spaces)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    return ''.join(ch for ch in data if ch == ' ' or ('A' <= ch <= 'Z'))

def chi_squared_distance(text: str) -> float:
    """Chi-squared distance between text letter freq and English."""
    counts = Counter(ch for ch in text if ch in ALPHABET_SET)
    total = sum(counts.values()) or 1
    chi2 = 0.0
    for letter in ALPHA:
        observed = counts[letter]
        expected = ENGLISH_FREQ[letter] / 100.0 * total
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected
    return chi2

def frequency_seed_mapping(cipher_text: str) -> Dict[str, str]:
    counts = Counter(ch for ch in cipher_text if ch in ALPHABET_SET)
    cipher_sorted = [p[0] for p in counts.most_common()]
    for L in ALPHA:
        if L not in cipher_sorted:
            cipher_sorted.append(L)
    mapping: Dict[str, str] = {}
    for i, c in enumerate(cipher_sorted):
        if i < 26:
            mapping[c] = ENGLISH_FREQ_ORDER[i]
    used = set(mapping.values())
    leftover_plain = [L for L in ALPHA if L not in used]
    for c in ALPHA:
        if c not in mapping:
            mapping[c] = leftover_plain.pop() if leftover_plain else c
    return mapping

def apply_mapping(text: str, m: Dict[str, str]) -> str:
    return ''.join(m.get(ch, ' ') if ch != ' ' else ' ' for ch in text)

def english_like_score(plain: str) -> float:
    """Heuristic language score combining:
    - common word hits (weighted),
    - bigram & trigram boosts,
    - letter-frequency fit (negative chi2),
    - vowel/consonant mix sanity,
    - penalties for improbable patterns (e.g., Q without U, VV/JJ/QQ/WW…)
    """
    score = 0.0

    # Word score
    words = [w for w in plain.split(' ') if w]
    for w in words:
        if w in COMMON_WORDS:
            score += min(4.0, 0.5 + 0.15 * len(w))
    score += 1.5 * plain.count(' THE ')

    # N-gram boosts
    for i in range(len(plain) - 1):
        bg = plain[i:i+2]
        if ' ' in bg:
            continue
        score += BIGRAM_WEIGHTS.get(bg, 0.0)
    for i in range(len(plain) - 2):
        tg = plain[i:i+3]
        if ' ' in tg:
            continue
        score += TRIGRAM_WEIGHTS.get(tg, 0.0)

    # Penalize Q not followed by U (English heuristic)
    for i, ch in enumerate(plain):
        if ch == 'Q':
            if i + 1 >= len(plain) or plain[i+1] != 'U':
                score -= 2.5

    # Penalize very rare double letters
    BAD_DOUBLES = {"VV", "JJ", "KK", "QQ", "WW", "YY", "ZZ", "XX"}
    for dd in BAD_DOUBLES:
        score -= 1.0 * plain.count(dd)

    # Letter frequency fit (negative chi2)
    score += -0.25 * chi_squared_distance(plain)

    # Vowel/consonant sanity (too few vowels penalized)
    vowels = set('AEIOU')
    letters = [c for c in plain if c in ALPHABET_SET]
    if letters:
        v = sum(1 for c in letters if c in vowels)
        ratio = v / len(letters)
        score -= 10.0 * abs(ratio - 0.40)

    return score

def random_swap(mapping: Dict[str, str]) -> Tuple[str, str]:
    a, b = random.sample(ALPHA, 2)
    return a, b

def swap_in_mapping(mapping: Dict[str, str], a: str, b: str) -> Dict[str, str]:
    new_map = mapping.copy()
    pa, pb = new_map[a], new_map[b]
    new_map[a], new_map[b] = pb, pa
    return new_map

def hill_climb(cipher: str, init_map: Dict[str, str], iters: int, seed: int) -> Tuple[float, Dict[str, str], str]:
    random.seed(seed)
    cur_map = init_map
    cur_plain = apply_mapping(cipher, cur_map)
    cur_score = english_like_score(cur_plain)
    best_score, best_map, best_plain = cur_score, cur_map, cur_plain

    # Annealing schedule
    T0, T_end = 3.0, 0.05
    for step in range(1, iters + 1):
        T = T0 * (T_end / T0) ** (step / iters)
        a, b = random_swap(cur_map)
        new_map = swap_in_mapping(cur_map, a, b)
        new_plain = apply_mapping(cipher, new_map)
        new_score = english_like_score(new_plain)
        delta = new_score - cur_score
        if delta > 0 or random.random() < math.exp(delta / max(1e-6, T)):
            cur_map, cur_plain, cur_score = new_map, new_plain, new_score
            if new_score > best_score:
                best_score, best_map, best_plain = new_score, new_map, new_plain
    return best_score, best_map, best_plain

def refine_with_single_letter_words(cipher: str, mapping: Dict[str, str]) -> Dict[str, str]:
    """If the slice contains one-letter words, bias them to A / I appropriately."""
    one_letter_tokens = re.findall(r"(?:(?<= )|^)([A-Z])(?=(?: )|$)", cipher)
    if not one_letter_tokens:
        return mapping
    counts = Counter(one_letter_tokens)
    most_common_cipher = [c for c, _ in counts.most_common(2)]
    targets = ['A', 'I']
    new_map = mapping.copy()

    def force(assign: Dict[str, str], cipher_letter: str, target: str):
        inv = {v:k for k,v in assign.items()}
        if target in inv and inv[target] != cipher_letter:
            other = inv[target]
            assign[other], assign[cipher_letter] = assign[cipher_letter], assign[other]
        else:
            assign[cipher_letter] = target

    # try both assignments and keep better
    assign1 = new_map.copy()
    if len(most_common_cipher) >= 1:
        force(assign1, most_common_cipher[0], 'A')
    if len(most_common_cipher) >= 2:
        force(assign1, most_common_cipher[1], 'I')

    assign2 = new_map.copy()
    if len(most_common_cipher) >= 1:
        force(assign2, most_common_cipher[0], 'I')
    if len(most_common_cipher) >= 2:
        force(assign2, most_common_cipher[1], 'A')

    p1 = apply_mapping(cipher, assign1)
    p2 = apply_mapping(cipher, assign2)
    return assign1 if english_like_score(p1) >= english_like_score(p2) else assign2

def best_improvement_polish(cipher: str, mapping: Dict[str, str], rounds: int = 40) -> Tuple[float, Dict[str, str], str]:
    """Deterministic local search: keep swapping the *best* pair if it improves score."""
    cur_map = mapping.copy()
    cur_plain = apply_mapping(cipher, cur_map)
    cur_score = english_like_score(cur_plain)
    for _ in range(rounds):
        best_delta = 0.0
        best_pair = None
        best_plain_local = cur_plain
        best_map_local = cur_map
        # evaluate all 325 letter-pair swaps
        for i in range(26):
            a = ALPHA[i]
            for j in range(i+1, 26):
                b = ALPHA[j]
                m2 = swap_in_mapping(cur_map, a, b)
                p2 = apply_mapping(cipher, m2)
                s2 = english_like_score(p2)
                delta = s2 - cur_score
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (a, b)
                    best_plain_local = p2
                    best_map_local = m2
        if best_delta <= 1e-9:
            break
        cur_map, cur_plain, cur_score = best_map_local, best_plain_local, cur_score + best_delta
    return cur_score, cur_map, cur_plain

def crack_slice(cipher_slice: str, tries: int, iters: int, seed: int) -> Tuple[float, Dict[str, str], str]:
    best_overall = (-1e9, None, None)  # score, map, plain

    base_map = frequency_seed_mapping(cipher_slice)
    base_map = refine_with_single_letter_words(cipher_slice, base_map)

    for t in range(tries):
        # randomize a bit from the base map to diversify restarts
        randomized = base_map.copy()
        for _ in range(6):
            a, b = random.sample(ALPHA, 2)
            randomized[a], randomized[b] = randomized[b], randomized[a]
        score, mapping, plain = hill_climb(cipher_slice, randomized, iters, seed + t + 12345)
        # polishing pass
        score, mapping, plain = best_improvement_polish(cipher_slice, mapping, rounds=40)
        if score > best_overall[0]:
            best_overall = (score, mapping, plain)
    assert best_overall[1] is not None and best_overall[2] is not None
    return best_overall  # type: ignore

def first_n_words(text: str, n: int) -> List[str]:
    words = [w for w in text.split(' ') if w]
    return words[:n]

# --- Main ---
def main():
    ap = argparse.ArgumentParser(description="Outer Space Signals cracker (offset mode)")
    ap.add_argument('--signal-file', default='signal.txt', help='Path to signal file')
    ap.add_argument('--msg-len', type=int, default=721, help='Length of hidden message')
    ap.add_argument('--offset', type=int, required=True, help='Start offset in the signal file')
    ap.add_argument('--tries', type=int, default=12, help='Annealing restarts per candidate slice')
    ap.add_argument('--iters', type=int, default=12000, help='Hill-climb iterations per restart')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed (negative for random)')
    ap.add_argument('--print-first', type=int, default=9, help='How many first words to print')
    args = ap.parse_args()

    seed = args.seed if args.seed >= 0 else random.randrange(10**9)
    random.seed(seed)

    raw = load_signal(args.signal_file)
    if args.offset + args.msg_len > len(raw):
        raise ValueError(f"Offset {args.offset} + msg_len {args.msg_len} exceeds file size {len(raw)}")

    slice_text = raw[args.offset:args.offset + args.msg_len]
    print(f"Loaded slice from offset={args.offset}, length={len(slice_text)}")

    score, mapping, plain = crack_slice(slice_text, args.tries, args.iters, seed)

    # Write artifacts
    with open("decoded.txt", "w", encoding="utf-8") as f:
        f.write(plain)
    with open("candidate_slice.txt", "w", encoding="utf-8") as f:
        f.write(slice_text)
    with open("mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    words = first_n_words(plain, args.print_first)
    print("\n===== RESULTS =====")
    print(f"Offset : {args.offset}")
    print(f"First {args.print_first} words : {' '.join(words)}")
    print("Artifacts written: decoded.txt, mapping.json, candidate_slice.txt")


if __name__ == '__main__':
    main()
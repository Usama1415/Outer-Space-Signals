import string
import os
from collections import Counter

MESSAGE_LENGTH = 721
TOP10_ENGLISH = "EATOIRSNHU"  # Expected top letters in decrypted text


def load_signal(filepath: str) -> str:
    """Load the 64KB alien signal text from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def find_message(signal: str, length: int = MESSAGE_LENGTH) -> str:
    """
    Find the 721-character hidden message inside the 64KB text.
    For simplicity we just return the first valid segment of length 721.
    """
    for i in range(len(signal) - length + 1):
        segment = signal[i:i + length]
        if all(ch in string.ascii_uppercase + " " for ch in segment):
            return segment
    raise ValueError("No valid 721-character message found.")


def build_substitution(cipher_text: str) -> dict:
    """
    Build a naive substitution mapping based on frequency analysis.
    The most frequent cipher letters are mapped to the most frequent English letters.
    """
    freq = Counter(ch for ch in cipher_text if ch != " ")
    sorted_cipher = [ch for ch, _ in freq.most_common()]

    mapping = {}
    for i, ch in enumerate(sorted_cipher):
        if i < len(TOP10_ENGLISH):
            mapping[ch] = TOP10_ENGLISH[i]
    return mapping


def apply_substitution(cipher_text: str, mapping: dict) -> str:
    """Apply substitution mapping to cipher text."""
    return "".join(mapping.get(ch, ch) for ch in cipher_text)


def main():
    print("🛸 NASA Signal Decoder - Deciphering Messages from Planet Dyslexia 🛸")

    # Step 1: Load file
    base_dir = os.path.dirname(__file__)
    signal = load_signal(os.path.join(base_dir, "..", "signal.txt"))

    # Step 2: Find hidden message
    cipher_message = find_message(signal, MESSAGE_LENGTH)
    print("\n📡 Found encrypted message (first 100 chars):")
    print(cipher_message[:100])

    # Step 3: Build substitution mapping
    mapping = build_substitution(cipher_message)
    print("\n🔤 Initial substitution mapping (freq-based):")
    print(mapping)

    # Step 4: Decode message
    decoded = apply_substitution(cipher_message, mapping)
    print("\n📝 Draft decoded message (first 200 chars):")
    print(decoded[:200])

    # Step 5: First 9 words
    first_words = " ".join(decoded.split()[:9])
    print("\n🚀 First 9 words of message:")
    print(first_words)


if __name__ == "__main__":
    main()

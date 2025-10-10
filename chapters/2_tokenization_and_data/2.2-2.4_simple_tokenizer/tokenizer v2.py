# chapter2_tokenizer.py
import re
from typing import List, Dict, Tuple

# -----------------------
# 2.2 Tokenizing text
# -----------------------
def tokenize(text: str) -> List[str]:
    """
    Split text into tokens (words, punctuation, etc.) similar to the book's approach.
    """
    # Split words, punctuation, and whitespace into separate tokens
    tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
    # Remove empty strings and stray whitespace tokens
    tokens = [tok.strip() for tok in tokens if tok.strip()]
    return tokens


# -----------------------
# 2.3 Building a vocabulary
# -----------------------
def build_vocab(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Build a bidirectional vocabulary from a token list:
    - str_to_int: token -> id
    - int_to_str: id -> token
    Returns (str_to_int, int_to_str, vocab_size).
    """
    # Unique tokens, sorted alphabetically (as shown in the book)
    unique_tokens = sorted(set(tokens))

    specials = ["<|endoftext|>", "<|unk|>"]
    for s in specials:
        if s not in unique_tokens:
            unique_tokens.append(s)



    # Forward & inverse maps
    str_to_int = {tok: idx for idx, tok in enumerate(unique_tokens)}
    int_to_str = {idx: tok for tok, idx in str_to_int.items()}

    return str_to_int, int_to_str, len(str_to_int)


# -----------------------
# Listing 2.3 – SimpleTokenizerV1
# -----------------------
class SimpleTokenizerV1:
    """
    Minimal tokenizer that uses a provided vocabulary to encode/decode:
      - encode(text) -> List[int]
      - decode(List[int]) -> str
    """
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int: Dict[str, int] = vocab
        self.int_to_str: Dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        # Same preprocessing as §2.2
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # IMPORTANT: V1 assumes *every* token exists in the vocab
        # (unknowns will raise a KeyError). V2 will add <|unk|>.
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Join tokens and then fix spaces before punctuation
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

if __name__ == "__main__":
    sample_text = "Hello, do you like tea? I like tea."

    # Step 1: Tokenize
    tokens = tokenize(sample_text)
    print("Tokens:", tokens)

    # Step 2: Build vocabulary (should include <|endoftext|> and <|unk|>)
    str_to_int, int_to_str, vocab_size = build_vocab(tokens)
    print("Vocab size:", vocab_size)
    print("First 10 vocab entries:", list(str_to_int.items())[:10])

    # -----------------------
    # Self-tests
    # -----------------------

    # 1. Special tokens exist
    assert "<|endoftext|>" in str_to_int, "Missing <|endoftext|> in vocab"
    assert "<|unk|>" in str_to_int, "Missing <|unk|> in vocab"

    # 2. Size bump check
    old_size = len(set(tokens))
    assert vocab_size == old_size + 2, f"Expected {old_size+2}, got {vocab_size}"

    # 3. Round-trip consistency
    unk_id = str_to_int["<|unk|>"]
    eot_id = str_to_int["<|endoftext|>"]
    assert int_to_str[unk_id] == "<|unk|>"
    assert int_to_str[eot_id] == "<|endoftext|>"

    print(" Self-tests passed")

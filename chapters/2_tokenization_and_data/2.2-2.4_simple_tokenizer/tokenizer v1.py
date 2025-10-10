import re

# -----------------------
# 2.2 Tokenizing text
# -----------------------
def tokenize(text: str):
    """
    Splits text into tokens (words, punctuation, etc.).
    Mimics Listing 2.1 in the book.
    """
    # Regex separates words, punctuation, and spaces
    tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
    # Remove empty strings and stray whitespace
    tokens = [tok.strip() for tok in tokens if tok.strip()]
    return tokens


# -----------------------
# 2.3 Building a vocabulary
# -----------------------
def build_vocab(tokens):
    """
    Builds bidirectional vocab dicts from a list of tokens.
    Based on Listing 2.2 in the book.
    """
    # 1. Collect unique tokens, sorted alphabetically
    unique_tokens = sorted(set(tokens))

    # 2. Forward and inverse mapping
    str_to_int = {tok: idx for idx, tok in enumerate(unique_tokens)}
    int_to_str = {idx: tok for tok, idx in str_to_int.items()}

    vocab_size = len(str_to_int)
    return str_to_int, int_to_str, vocab_size


# -----------------------
# Example run
# -----------------------
if __name__ == "__main__":
    sample_text = "Hello, do you like tea? I like tea."
    
    # Step 1: Tokenize
    tokens = tokenize(sample_text)
    print("Tokens:", tokens)

    # Step 2: Build vocabulary
    str_to_int, int_to_str, vocab_size = build_vocab(tokens)
    print("Vocab size:", vocab_size)
    print("First 10 vocab entries:", list(str_to_int.items())[:10])

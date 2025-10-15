# resources/train_check.py
# Chapter 6.4 — Initialize a model with pretrained weights and sanity-check generation

import os
from pathlib import Path
import torch
import tiktoken
import pandas as pd

# ---- local helpers from earlier chapters (your consolidated files)
from resources.loading import download_and_load_gpt2, load_weights_into_gpt
from resources.past_chap import Toilet, generate_text_simple

# ---------- Small utility wrappers (compatible with the book text) ----------
def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """Encode text into a batch of token ids (shape [1, T])."""
    return torch.tensor(
        tokenizer.encode(text, allowed_special={"<|endoftext|>"}),
        dtype=torch.long
    ).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """Decode a batch of token ids (shape [1, T]) back to text."""
    return tokenizer.decode(token_ids.squeeze(0).tolist())

# ---------- Config from the screenshot ----------
CHOOSE_MODEL = "gpt2-small (124M)"   # change to other sizes if you like
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,   # Context length
    "drop_rate": 0.0,         # Dropout rate
    "qkv_bias": True          # Query-key-value bias
}

# Per-size architectural overrides
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# ---------- Check dataset sequence length vs context length ----------
# We reuse the Chapter 6.3 tokenization to estimate max sequence length on train.csv
tokenizer = tiktoken.get_encoding("gpt2")

def longest_tokenized_length(csv_path: str) -> int:
    df = pd.read_csv(csv_path)
    max_len = 0
    for text in df["Text"]:
        L = len(tokenizer.encode(text))
        if L > max_len:
            max_len = L
    return max_len

train_csv = Path("train.csv")  # produced in Chapter 6.2
if not train_csv.exists():
    raise FileNotFoundError("Expected train.csv from Chapter 6.2 in the current directory.")

train_max_len = longest_tokenized_length(str(train_csv))

assert train_max_len <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_max_len} exceeds model's context length "
    f"{BASE_CONFIG['context_length']}. Reinitialize datasets with "
    f"max_length={BASE_CONFIG['context_length']}"
)

print(f"[OK] Max train sequence length: {train_max_len} (<= context {BASE_CONFIG['context_length']})")

# ---------- Download OpenAI GPT-2 weights and build the model ----------
# The loader expects size tokens like "124M", "355M", ...
model_size_token = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size_token, models_dir="gpt2")

model = Toilet(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
print("[OK] Weights loaded into model and set to eval().")

# ---------- Sanity check #1: short continuation ----------
text_1 = INPUT_PROMPT
idx = text_to_token_ids(text_1, tokenizer)
token_ids = generate_text_simple(
    model=model,
    idx=idx,
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"],
)
print("\n--- Sample continuation ---")
print(token_ids_to_text(token_ids, tokenizer))

# ---------- Sanity check #2: can the base model follow the spam instruction? ----------
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
idx2 = text_to_token_ids(text_2, tokenizer)
token_ids2 = generate_text_simple(
    model=model,
    idx=idx2,
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"],
)

print("\n--- Instruction prompt result (before finetuning) ---")
print(token_ids_to_text(token_ids2, tokenizer))

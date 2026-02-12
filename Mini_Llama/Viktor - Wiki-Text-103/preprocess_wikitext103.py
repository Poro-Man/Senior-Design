import os
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

# ---------------- CONFIG ----------------
OUTPUT_DIR = "wikitext103_tokens"
SEQ_LEN = 1024
SPLITS = ["train", "validation"]
TOKENIZER_NAME = "gpt2"
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_text(text):
    text = text.strip()
    return text

def tokenize_split(dataset, tokenizer, split_name):
    all_tokens = []

    print(f"Tokenizing {split_name} split...")
    for example in tqdm(dataset):
        text = normalize_text(example["text"])
        if text == "":
            continue
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.int32)

    # Trim to multiple of SEQ_LEN
    usable_len = (len(all_tokens) // SEQ_LEN) * SEQ_LEN
    all_tokens = all_tokens[:usable_len]

    # Reshape into sequences
    sequences = all_tokens.reshape(-1, SEQ_LEN)

    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.npy")
    np.save(out_path, sequences)
    print(f"Saved {sequences.shape[0]} sequences to {out_path}")

def main():
    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    for split in SPLITS:
        tokenize_split(dataset[split], tokenizer, split)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()

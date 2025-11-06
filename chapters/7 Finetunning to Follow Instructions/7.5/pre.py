import os, json, urllib.request
import torch
import tiktoken

# ---- use YOUR two files instead of book chapters ----
from loading import (
    download_and_load_gpt2,
    load_weights_into_gpt,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)
from past_chap import Toilet


# -------------------------
# 7.2 utilities (inline)
# -------------------------
URL  = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
FILE = "instruction-data.json"

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as r:
            txt = r.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(txt)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def partition_dataset(data):
    n = len(data)
    n_train = int(n * 0.85)
    n_test  = int(n * 0.10)
    train = data[:n_train]
    test  = data[n_train:n_train+n_test]
    val   = data[n_train+n_test:]
    return train, val, test

def format_input(entry: dict) -> str:
    txt = ("Below is an instruction that describes a task. "
           "Write a response that appropriately completes the request."
           f"\n\n### Instruction:\n{entry['instruction']}")
    if entry.get("input"):
        txt += f"\n\n### Input:\n{entry['input']}"
    return txt


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If you're on Apple Silicon and want MPS:
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    print("Device:", device)

    # Tokenizer + EOS id
    tokenizer = tiktoken.get_encoding("gpt2")
    EOS_ID = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    # GPT base config + size presets (aligned to your Toilet class)
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model_configs = {
        "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # Download OpenAI GPT-2 weights and map into YOUR model
    model_size = CHOOSE_MODEL.split(" ")[-1].strip("()")  # "355M"
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = Toilet(BASE_CONFIG).to(device).eval()
    load_weights_into_gpt(model, params)

    # Load data & build prompt from val_data[0]
    torch.manual_seed(123)
    data = download_and_load_file(FILE, URL)
    _, val_data, _ = partition_dataset(data)

    input_text = format_input(val_data[0])
    print("\n--- Prompt (truncated) ---\n", input_text[:300], "...\n")

    # Baseline generation (pre-finetuning)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=EOS_ID,
        top_k=50,
        temperature=1.0,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    # Strip the prompt to show only the model's answer
    response_text = generated_text[len(input_text):].strip()
    print("### Response:\n", response_text)

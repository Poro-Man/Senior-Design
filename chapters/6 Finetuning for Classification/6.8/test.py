# 6.8/inference_with_saved_model.py
# Use the finetuned classifier saved in 6.7 (review_classifier.pth)

import torch
import torch.nn as nn
import tiktoken

# ---- Your local GPT-2 model class (chapter 5) ----
from resources.past_chap import Toilet as GPTModel  # same class used in training

# ----------------------------
# 1) Rebuild the architecture
# ----------------------------
CHOOSE_MODEL = "gpt2-small (124M)"  # must match 6.7
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Build the model skeleton and swap in the 2-class head
model = GPTModel(BASE_CONFIG)
model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=2)

# ---------------------------------------
# 2) Load the finetuned weights from 6.7
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_path = "review_classifier.pth"   # produced at the end of 6.7
state = torch.load(state_path, map_location=device)
model.load_state_dict(state)
model.to(device).eval()
print(f"[OK] Loaded finetuned classifier from: {state_path}")

# ------------------------
# 3) Tokenizer and helper
# ------------------------
tokenizer = tiktoken.get_encoding("gpt2")

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    Returns 'spam' or 'not spam' for a single input string, matching Listing 6.12.
    - Truncates to min(max_length, model context)
    - Pads to that length
    - Uses last-token logits -> argmax
    """
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_ctx = model.pos_emb.weight.shape[1]
    clip_len = supported_ctx if max_length is None else min(max_length, supported_ctx)
    TRAIN_MAX_LENGTH = 120  # or whatever your train_dataset.max_length was


    input_ids = input_ids[:clip_len]                                  # truncate
    input_ids += [pad_token_id] * (clip_len - len(input_ids))         # pad

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]                        # last-token logits
        pred = torch.argmax(logits, dim=-1).item()

    return "spam" if pred == 1 else "not spam"

# ------------------------
# 4) Example predictions
# ------------------------
if __name__ == "__main__":
    # If you know the training max length from 6.3, set it here (e.g., 120).
    # Otherwise, leave as None to use the model's full context window.
    TRAIN_MAX_LENGTH = 120

    text_1 = (
        "You are a winner you have been specially "
        "selected to receive $1000 cash or a $2000 award."
    )
    text_2 = (
        "Hey, just wanted to check if we're still on for dinner tonight? Let me know! We've been seeing together for a while now and I really enjoy your company."
    )

    print("Example 1:", classify_review(text_1, model, tokenizer, device, max_length=TRAIN_MAX_LENGTH))
    print("Example 2:", classify_review(text_2, model, tokenizer, device, max_length=TRAIN_MAX_LENGTH))

    # If you want to re-save to a different filename/location:
    # torch.save(model.state_dict(), "review_classifier_copy.pth")

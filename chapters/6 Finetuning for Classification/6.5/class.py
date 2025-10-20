"""Classification head setup for GPT-2

This script replaces GPT-2's language modeling head with a 2-class
classification head, freezes most model parameters, and leaves the
final transformer block, final layernorm, and the new head trainable.

It performs a quick forward pass using a short example input to
sanity-check shapes and outputs.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import tiktoken

# Reduce TensorFlow logging (if TF is present in the environment)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Local helper functions that load model code and pretrained weights
from resources.loading import download_and_load_gpt2, load_weights_into_gpt
from resources.past_chap import Toilet as GPTModel, generate_text_simple

# Model selection and base configuration
CHOOSE_MODEL = "gpt2-small (124M)"
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

# Tokenizer used for encoding sample text
tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text: str) -> torch.Tensor:
    """Encode text to a tensor of token ids with batch dim 1."""
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

# Download/load pretrained weights and initialize model
model_size_token = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
_, params = download_and_load_gpt2(model_size=model_size_token, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

print(model)

# Replace LM head with a 2-class classification head
torch.manual_seed(123)
num_classes = 2
model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# Freeze all parameters, then selectively unfreeze the last block, final
# normalization layer, and the new classification head for fine-tuning.
for p in model.parameters():
    p.requires_grad = False

for p in model.trf_blocks[-1].parameters():
    p.requires_grad = True
for p in model.final_norm.parameters():
    p.requires_grad = True

for p in model.out_head.parameters():
    p.requires_grad = True

# Print a short summary of trainable parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable params: {trainable:,} / {total:,}")

# Encode a short example input and run a forward pass for shape checks
inputs = text_to_token_ids("Do you have time")
print("\nInputs:", inputs)
print("Inputs dimensions:", inputs.shape)

with torch.no_grad():
    outputs = model(inputs)

print("\nOutputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)

# Take last-token logits and convert to probabilities
last_token_logits = outputs[:, -1, :]
print("\nLast output token:", last_token_logits)

probs = torch.softmax(last_token_logits, dim=-1)
print("Probabilities (ham, spam):", probs)

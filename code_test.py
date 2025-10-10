import sys
import torch
import torch.nn as nn
import tiktoken
from resources.work import Toilet

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

def the_numbers_mason(text: str, tokenizer):
    # tiktoken's encode returns list[int]; allow EOT if you use it
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # (1, T)

def what_do_they_mean(token_ids: torch.Tensor, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)

if __name__ == "__main__":
    torch.manual_seed(123)

    # Build model once
    model = Toilet(GPT_CONFIG_124M)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # ---- Generation demo ----
    start_context = "The numbers, Mason. What do they mean?"
    start_ids = the_numbers_mason(start_context, tokenizer)   # (1, T0)

    # Use Toilet's generate_text class method
    out_ids = Toilet.generate_text(
        model=model,
        idx=start_ids,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    print("Output text:\n", what_do_they_mean(out_ids.cpu(), tokenizer))

    # ---- Forward pass + softmax shape sanity ----
    # Inputs must be LongTensor within vocab range
    inputs = torch.tensor([[16883, 3626, 6100],
                           [   40, 1107,  588]], dtype=torch.long, device=device)  # (B=2, T=3)
    targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
                             [107, 588, 11311]]) # " really like chocolate"]

    with torch.no_grad():
        logits = model(inputs)                  # (2, 3, 50257)
    probas = torch.softmax(logits, dim=-1)     # (2, 3, 50257)
    print("logits shape :", logits.shape)
    print("probas shape :", probas.shape)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)  # Greedily pick the next token
    print("Token IDs:\n", token_ids)
    print(f"Target batch 1: {what_do_they_mean(targets[0], tokenizer)}")
    print(f"Target batch 2: {what_do_they_mean(targets[1], tokenizer)}")

    text_idx = 0 
    target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
    print("Text 1:", target_probas_1)
    text_idx = 1
    target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
    print("Text 2:", target_probas_2)
# -*- coding: utf-8 -*-
"""
bonfire.py — 5.4 save/load demo wired to your repo

Uses:
- resources/past_chap.py  -> Toilet (GPT model)
- resources/data.py       -> spawn_dataloader (book Chapter 2 loader)
Matches the book's Listing 5.3 training loop and §5.4 save/load procedure.

NOTE: forced CPU to avoid RTX 5090 (sm_120) + current PyTorch mismatch.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # force CPU execution

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# --- your repo imports (keep these names) ---
from resources.past_chap import Toilet
from resources.data import spawn_dataloader

# -------------------------
# Config (GPT-2 small-ish)
# -------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   # bump to 1024 later if you like
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# -------------------------------------------------
# §5.1 helpers — text <-> tokens and generation
# -------------------------------------------------
def text_to_token_ids(text: str, tokenizer):
    ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)

@torch.no_grad()
def generate_text_simple(model: nn.Module,
                         idx: torch.Tensor,
                         max_new_tokens: int,
                         context_size: int) -> torch.Tensor:
    model.eval()
    idx = idx.to(next(model.parameters()).device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)              # (B, Tcond, V)
        next_logits = logits[:, -1, :]        # (B, V)
        probs = torch.softmax(next_logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # greedy
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# -------------------------------------------------
# §5.1 loss utilities
# -------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)                        # (B, T, V)
    return F.cross_entropy(logits.flatten(0, 1),       # (B·T, V)
                           target_batch.flatten())     # (B·T,)

@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    limit = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total = 0.0
    for i, (xb, yb) in enumerate(data_loader):
        if i >= limit:
            break
        total += calc_loss_batch(xb, yb, model, device).item()
    return total / limit

# -------------------------------------------------
# §5.2 evaluation + sample printer
# -------------------------------------------------
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context, context_length):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        out_ids = generate_text_simple(model, encoded, 50, context_length)
        decoded = token_ids_to_text(out_ids.cpu(), tokenizer)
        print(decoded.replace("\n", " "))
    model.train()

# -------------------------------------------------
# §5.2 main training loop (Listing 5.3)
# -------------------------------------------------
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context,
                       tokenizer, context_length):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(xb, yb, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += xb.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                tr, va = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(tr)
                val_losses.append(va)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {tr:.3f}, Val loss {va:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context, context_length)

    return train_losses, val_losses, track_tokens_seen

# -------------------------------------------------
# Runner (book §5.4 save & load)
# -------------------------------------------------
def main():
    torch.manual_seed(123)

    # Force CPU to avoid the 5090 kernel-image error with your current build
    device = torch.device("cpu")
    print("Device:", device)

    # Model & tokenizer
    model = Toilet(GPT_CONFIG_124M).to(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Dataset
    corpus_path = os.path.join("resources", "the-verdict.txt")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_text, val_text = text_data[:split_idx], text_data[split_idx:]

    # DataLoaders (use your spawn_dataloader)
    train_loader = spawn_dataloader(
        train_text,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        tokenizer=tokenizer,
    )
    val_loader = spawn_dataloader(
        val_text,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
        tokenizer=tokenizer,
    )

    # Optimizer (book uses AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

    # ----- Train a bit -----
    _ = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=2, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        context_length=GPT_CONFIG_124M["context_length"],
    )

    # ----- §5.4: Save model + optimizer -----
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pth"))
    torch.save(
        {"model_state_dict": model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(ckpt_dir, "model_and_optimizer.pth"),
    )
    print(f"Saved checkpoints to: {ckpt_dir}/")

    # ----- §5.4: Reload into fresh model + optimizer and continue -----
    print("\nReloading and continuing for one more epoch...")
    re_model = Toilet(GPT_CONFIG_124M).to(device)
    re_optimizer = torch.optim.AdamW(re_model.parameters(), lr=4e-4, weight_decay=0.1)

    # load both states (recommended if you want to keep training)
    checkpoint = torch.load(os.path.join(ckpt_dir, "model_and_optimizer.pth"), map_location=device)
    re_model.load_state_dict(checkpoint["model_state_dict"])
    re_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    re_model.train()

    # continue training for 1 epoch (book’s exercise)
    _ = train_model_simple(
        re_model, train_loader, val_loader, re_optimizer, device,
        num_epochs=1, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        context_length=GPT_CONFIG_124M["context_length"],
    )

if __name__ == "__main__":
    main()

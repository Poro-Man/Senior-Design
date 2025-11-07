# tune.py — Chapter 7.6: finetune GPT-2 on instruction data (noisy / debug-friendly)
# - Builds dataset & custom collate (7.2/7.3)
# - Creates DataLoaders (7.4)
# - Loads pretrained GPT-2 into textbook GPTModel (7.5)
# - Computes initial losses, finetunes, plots & saves losses, saves checkpoint

import os
import json
import urllib.request
import time
from functools import partial
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# === textbook helpers bundled in your previous_chapters.py ===
from previous_chapters import (
    GPTModel,
    load_weights_into_gpt,
    calc_loss_loader,
    train_model_simple,
    plot_losses,  # we'll draw with it and then save the figure via matplotlib
)

# === your GPT-2 weights downloader/mapper ===
from loading import download_and_load_gpt2


URL  = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
FILE = "instruction-data.json"


def debug(msg: str):
    print(msg, flush=True)


# ---------------- §7.2 utils ----------------
def download_and_load_file(file_path: str, url: str):
    if not os.path.exists(file_path):
        debug(f"[download] fetching {url}")
        with urllib.request.urlopen(url) as r:
            txt = r.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(txt)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    debug(f"[data] loaded {len(data)} entries from {file_path}")
    return data

def partition_dataset(data):
    n = len(data)
    n_train = int(n * 0.85)
    n_test  = int(n * 0.10)
    train = data[:n_train]
    test  = data[n_train:n_train+n_test]
    val   = data[n_train+n_test:]
    debug(f"[split] train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test

def format_input(entry: dict) -> str:
    txt = ("Below is an instruction that describes a task. "
           "Write a response that appropriately completes the request."
           f"\n\n### Instruction:\n{entry['instruction']}")
    if entry.get("input"):
        txt += f"\n\n### Input:\n{entry['input']}"
    return txt
# -------------------------------------------


# ---------------- §7.3 dataset & collate ----------------
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encoded = []
        for e in data:
            text = format_input(e) + f"\n\n### Response:\n{e['output']}"
            self.encoded.append(tokenizer.encode(text))
    def __len__(self): return len(self.encoded)
    def __getitem__(self, i): return self.encoded[i]

def custom_collate_fn(
    batch,
    pad_token_id: int,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str = "cpu",
):
    # Pad to batch max length (+1), create inputs/targets (targets shifted by 1),
    # mask all but the first pad in targets with ignore_index.
    L = max(len(x) + 1 for x in batch)
    X, Y = [], []
    for item in batch:
        item = item + [pad_token_id]                 # one trailing pad
        padded = item + [pad_token_id] * (L - len(item))
        x = torch.tensor(padded[:-1])
        y = torch.tensor(padded[1:])
        pad_idx = (y == pad_token_id).nonzero().squeeze()
        if pad_idx.numel() > 1:
            y[pad_idx[1:]] = ignore_index
        if allowed_max_length is not None:
            x = x[:allowed_max_length]
            y = y[:allowed_max_length]
        X.append(x); Y.append(y)
    return torch.stack(X).to(device), torch.stack(Y).to(device)
# --------------------------------------------------------


def force_to(model: torch.nn.Module, device: torch.device):
    """Ensure ALL params & buffers sit on `device` (robust after CPU weight loads)."""
    model.to(device)
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.to(device)
            if p.grad is not None:
                p.grad = p.grad.to(device)
        for b in model.buffers():
            if torch.is_tensor(b):
                b.data = b.data.to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--save_plot", type=str, default="losses_7_6.png")
    args = ap.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available(): device = torch.device("mps")
    debug(f"[device] {device}")

    # Tokenizer & special ids
    tokenizer = tiktoken.get_encoding("gpt2")
    PAD_ID = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    EOS_ID = PAD_ID
    debug(f"[tokens] PAD/EOS id = {PAD_ID}")

    # Data
    data = download_and_load_file(FILE, URL)
    train_data, val_data, test_data = partition_dataset(data)

    # Datasets / Loaders
    collate = partial(
        custom_collate_fn,
        device=device,
        pad_token_id=PAD_ID,
        ignore_index=-100,
        allowed_max_length=args.max_len,
    )
    train_ds = InstructionDataset(train_data, tokenizer)
    val_ds   = InstructionDataset(val_data, tokenizer)
    test_ds  = InstructionDataset(test_data, tokenizer)

    torch.manual_seed(123)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=0, collate_fn=collate)
    debug("[loader] built train/val dataloaders")

    # Model config
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": args.max_len,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model_configs = {
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    }
    BASE_CONFIG.update(model_configs["gpt2-medium (355M)"])

    # Weights
    debug("[gpt2] downloading/loading weights (cached if present)")
    _, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    force_to(model, device)
    model.eval()
    debug("[model] loaded and moved to device")

    # Initial loss (quick check)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=5)
    debug(f"[loss:init] train={train_loss:.3f}  val={val_loss:.3f}")

    # Train
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    num_epochs = args.epochs

    debug("[train] starting…")
    start = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )
    mins = (time.time() - start) / 60
    debug(f"[train] done in {mins:.2f} min")

    # Plot & save (matplotlib; plot_losses has no savepath arg)
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    epochs_axis = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_axis, tokens_seen, train_losses, val_losses)
    plt.savefig(args.save_plot, bbox_inches="tight")
    plt.close()
    debug(f"[plot] saved to {args.save_plot}")

    # Save checkpoint
    ckpt_path = "ft_gpt2_ch7_6.pt"
    torch.save(
        {
            "config": BASE_CONFIG,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "pad_id": PAD_ID,
            "eos_id": EOS_ID,
        },
        ckpt_path,
    )
    debug(f"[ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    debug("[entry] tune.py launched")
    main()

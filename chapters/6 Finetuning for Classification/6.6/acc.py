# 6.6/baseline_metrics.py
# Baseline accuracy & loss BEFORE finetuning (builds on 6.5)

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken

# (Optional) quiet TensorFlow logs if it's installed
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ---------- Imports from earlier chapters ----------
from resources.loading import download_and_load_gpt2, load_weights_into_gpt
from resources.past_chap import Toilet as GPTModel  # your GPT-2 implementation

# ---------- Tokenizer ----------
tokenizer = tiktoken.get_encoding("gpt2")

# ---------- Dataset from 6.3 ----------
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.pad_token_id = pad_token_id
        self.encoded_texts = [tokenizer.encode(t) for t in self.data["Text"]]

        if max_length is None:
            self.max_length = max(len(enc) for enc in self.encoded_texts)
        else:
            self.max_length = max_length

        # truncate
        self.encoded_texts = [enc[: self.max_length] for enc in self.encoded_texts]
        # pad
        self.encoded_texts = [
            enc + [pad_token_id] * (self.max_length - len(enc))
            for enc in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        y = torch.tensor(int(self.data.iloc[idx]["Label"]), dtype=torch.long)
        return x, y


def make_loaders(batch_size=8, num_workers=0):
    train_ds_tmp = SpamDataset("train.csv", tokenizer)              # discover max_len on train
    max_len = train_ds_tmp.max_length

    train_ds = SpamDataset("train.csv", tokenizer, max_length=max_len)
    val_ds   = SpamDataset("validation.csv", tokenizer, max_length=max_len)
    test_ds  = SpamDataset("test.csv", tokenizer, max_length=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    return max_len, train_loader, val_loader, test_loader


# ---------- Model setup from 6.5 (replace LM head with classifier) ----------
def build_classifier_model():
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model_cfgs = {
        "gpt2-small (124M)": {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_cfgs[CHOOSE_MODEL])

    # load pretrained GPT-2 weights
    size_token = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    _, params = download_and_load_gpt2(model_size=size_token, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # replace LM head with 2-class classification head
    torch.manual_seed(123)
    model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=2)

    # freeze everything…
    for p in model.parameters():
        p.requires_grad = False
    # …except the last block, final norm, and new head (for later finetuning)
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.out_head.parameters():
        p.requires_grad = True

    return model


# ---------- Metrics (Listings 6.8 & 6.9) ----------
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct, total = 0, 0

    if len(data_loader) == 0:
        return float("nan")

    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (inp, tgt) in enumerate(data_loader):
        if i >= max_batches:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.no_grad():
            logits = model(inp)[:, -1, :]              # logits of last output token
            preds = torch.argmax(logits, dim=-1)
        total += preds.shape[0]
        correct += (preds == tgt).sum().item()

    return correct / total


def calc_loss_batch(inp, tgt, model, device):
    inp, tgt = inp.to(device), tgt.to(device)
    logits = model(inp)[:, -1, :]                      # logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, tgt)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    model.eval()
    total_loss = 0.0
    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (inp, tgt) in enumerate(data_loader):
        if i >= max_batches:
            break
        with torch.no_grad():
            loss = calc_loss_batch(inp, tgt, model, device)
        total_loss += loss.item()
    return total_loss / max_batches


# ---------- Main: compute baseline metrics ----------
if __name__ == "__main__":
    # loaders
    max_len, train_loader, val_loader, test_loader = make_loaders(batch_size=8, num_workers=0)

    # model
    model = build_classifier_model()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Accuracy on a few batches (fast baseline)
    torch.manual_seed(123)
    train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_acc   = calc_accuracy_loader(val_loader,   model, device, num_batches=10)
    test_acc  = calc_accuracy_loader(test_loader,  model, device, num_batches=10)

    print(f"Training accuracy:   {train_acc*100:5.2f}%")
    print(f"Validation accuracy: {val_acc*100:5.2f}%")
    print(f"Test accuracy:       {test_acc*100:5.2f}%")

    # Loss on a few batches (initial loss)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=5)
        test_loss  = calc_loss_loader(test_loader,  model, device, num_batches=5)

    print(f"\nTraining loss:   {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss:       {test_loss:.3f}")

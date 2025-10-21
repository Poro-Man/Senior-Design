
import os, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken
import matplotlib.pyplot as plt

# Optional: quiet TensorFlow chatter if installed
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ---- Helpers from earlier chapters (your repo structure) ----
from resources.loading import download_and_load_gpt2, load_weights_into_gpt
from resources.past_chap import Toilet as GPTModel

# ---------------------------- Data (from 6.3) ----------------------------
tokenizer = tiktoken.get_encoding("gpt2")

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        enc = [tokenizer.encode(t) for t in self.data["Text"]]
        self.max_length = max(len(x) for x in enc) if max_length is None else max_length
        # truncate & pad
        enc = [x[: self.max_length] for x in enc]
        self.encoded = [x + [pad_token_id] * (self.max_length - len(x)) for x in enc]

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        x = torch.tensor(self.encoded[i], dtype=torch.long)
        y = torch.tensor(int(self.data.iloc[i]["Label"]), dtype=torch.long)
        return x, y

def make_loaders(batch_size=8, num_workers=0):
    _tmp = SpamDataset("train.csv", tokenizer)  # discover max len from train
    max_len = _tmp.max_length
    train = SpamDataset("train.csv", tokenizer, max_length=max_len)
    val   = SpamDataset("validation.csv", tokenizer, max_length=max_len)
    test  = SpamDataset("test.csv", tokenizer, max_length=max_len)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    return max_len, train_loader, val_loader, test_loader

# ---------------------------- Model (from 6.5) ----------------------------
def build_classifier_model():
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    cfgs = {
        "gpt2-small (124M)": {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(cfgs[CHOOSE_MODEL])

    size_token = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    _, params = download_and_load_gpt2(model_size=size_token, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # swap LM head -> 2-class classifier
    torch.manual_seed(123)
    model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=2)

    # freeze everything, unfreeze last block + final norm + head
    for p in model.parameters(): p.requires_grad = False
    for p in model.trf_blocks[-1].parameters(): p.requires_grad = True
    for p in model.final_norm.parameters():    p.requires_grad = True
    for p in model.out_head.parameters():      p.requires_grad = True

    return model

# ---------------------------- Metrics (from 6.6) ----------------------------
def calc_accuracy_loader(loader, model, device, num_batches=None):
    model.eval()
    correct, total = 0, 0
    if len(loader) == 0: return float("nan")
    maxb = len(loader) if num_batches is None else min(num_batches, len(loader))
    for i, (inp, tgt) in enumerate(loader):
        if i >= maxb: break
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.no_grad():
            logits = model(inp)[:, -1, :]           # last-token logits
            preds = torch.argmax(logits, dim=-1)
        total += preds.shape[0]
        correct += (preds == tgt).sum().item()
    return correct / total

def calc_loss_batch(inp, tgt, model, device):
    inp, tgt = inp.to(device), tgt.to(device)
    logits = model(inp)[:, -1, :]                   # last-token logits
    return torch.nn.functional.cross_entropy(logits, tgt)

def calc_loss_loader(loader, model, device, num_batches=None):
    if len(loader) == 0: return float("nan")
    model.eval()
    total = 0.0
    maxb = len(loader) if num_batches is None else min(num_batches, len(loader))
    for i, (inp, tgt) in enumerate(loader):
        if i >= maxb: break
        with torch.no_grad():
            total += calc_loss_batch(inp, tgt, model, device).item()
    return total / maxb

# ---------------------------- Training (Listing 6.10) ----------------------------
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter, tokenizer=None):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()                                              # A
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                  # B
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                        # C
            optimizer.step()                                       # D
            examples_seen += input_batch.shape[0]                  # E
            global_step += 1

            if global_step % eval_freq == 0:                       # F
                tl, vl = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(tl);  val_losses.append(vl)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {tl:.3f}, Val loss {vl:.3f}")

        # G: accuracy at epoch end
        tr_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        va_acc = calc_accuracy_loader(val_loader,   model, device, num_batches=eval_iter)
        print(f"Training accuracy: {tr_acc*100:5.2f}% | Validation accuracy: {va_acc*100:5.2f}%")
        train_accs.append(tr_acc); val_accs.append(va_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

# ---------------------------- Plotting (Listing 6.11) ----------------------------
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="--", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twinx()                                   # B: second x-axis alignment trick
    ax2.plot(examples_seen, train_values, alpha=0)      # invisible; aligns ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()                                  # C
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    # Data
    max_len, train_loader, val_loader, test_loader = make_loaders(batch_size=8, num_workers=0)

    # Model + device
    model = build_classifier_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer & training settings
    torch.manual_seed(123)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    eval_freq  = 50
    eval_iter  = 5

    # Train
    start = time.time()
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter, tokenizer=tokenizer
    )
    mins = (time.time() - start) / 60
    print(f"Training completed in {mins:.2f} minutes.")

    # Plots
    epochs_tensor   = torch.linspace(0, num_epochs, len(train_losses))
    examples_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_tensor, train_losses, val_losses, label="loss")

    epochs_tensor   = torch.linspace(0, num_epochs, len(train_accs))
    examples_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_tensor, train_accs, val_accs, label="accuracy")

    # Final full-set accuracies
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy   = calc_accuracy_loader(val_loader,   model, device)
    test_accuracy  = calc_accuracy_loader(test_loader,  model, device)
    print(f"\nTraining accuracy:   {train_accuracy*100:5.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:5.2f}%")
    print(f"Test accuracy:       {test_accuracy*100:5.2f}%")

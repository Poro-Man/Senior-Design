# ============================================================
# Chapter 6.3 — Building PyTorch Datasets & DataLoaders
# ============================================================

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# ------------------------------------------------------------
# Tokenizer: GPT-2 BPE (pad_token_id uses <|endoftext|> = 50256)
# ------------------------------------------------------------
tokenizer = tiktoken.get_encoding("gpt2")
PAD_TOKEN_ID = 50256  # <|endoftext|>
# (Example from the book showing special-token handling)
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# ------------------------------------------------------------
# Listing 6.4 — Dataset class for tokenizing + padding
# ------------------------------------------------------------
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=PAD_TOKEN_ID):
        self.data = pd.read_csv(csv_file)  # A: load the CSV with columns ["Label", "Text"]
        self.pad_token_id = pad_token_id

        # Encode each text to token IDs
        self.encoded_texts = [tokenizer.encode(t) for t in self.data["Text"]]

        # Determine max length if not provided (uses longest sequence in this split)
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length  # B

        # Truncate to max_length
        self.encoded_texts = [
            enc[: self.max_length] for enc in self.encoded_texts
        ]

        # Pad to max_length
        self.encoded_texts = [
            enc + [pad_token_id] * (self.max_length - len(enc))
            for enc in self.encoded_texts
        ]  # C

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = int(self.data.iloc[index]["Label"])
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def _longest_encoded_length(self):
        max_len = 0
        for enc in self.encoded_texts:
            if len(enc) > max_len:
                max_len = len(enc)
        return max_len


# ------------------------------------------------------------
# Build datasets
#   Note: we compute max_length on the TRAIN split and reuse
#   it for val/test to keep consistent shapes.
# ------------------------------------------------------------
train_dataset = SpamDataset(csv_file="train.csv", tokenizer=tokenizer, max_length=None)
print("Longest sequence length (train split):", train_dataset.max_length)

val_dataset = SpamDataset(
    csv_file="validation.csv",
    tokenizer=tokenizer,
    max_length=train_dataset.max_length,
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    tokenizer=tokenizer,
    max_length=train_dataset.max_length,
)

# ------------------------------------------------------------
# Listing 6.5 — DataLoaders
# ------------------------------------------------------------
num_workers = 0  # A: keeps compatibility across machines
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,
)

# ------------------------------------------------------------
# Quick sanity checks
# ------------------------------------------------------------
for input_batch, label_batch in train_loader:
    pass  # just iterate once through to the last batch

print("Input batch dimensions:", input_batch.shape)   # e.g., torch.Size([8, max_len])
print("Label batch dimensions:", label_batch.shape)    # e.g., torch.Size([8])

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

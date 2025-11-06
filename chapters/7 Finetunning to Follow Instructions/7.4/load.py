# load.py — §7.4 end-to-end (load JSON -> split -> datasets -> loaders)

import os, json, urllib.request
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If you're on Apple Silicon, you can enable MPS:
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
print("Device:", device)


# ------------------ §7.2: load & split ------------------
URL  = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
FILE = "instruction-data.json"   # saved next to this script

def download_and_load_file(file_path: str, url: str):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as r:
            txt = r.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(txt)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def partition_dataset(data):
    n = len(data)
    n_train = int(n * 0.85)      # 85% train
    n_test  = int(n * 0.10)      # 10% test
    train_data = data[:n_train]
    test_data  = data[n_train:n_train + n_test]
    val_data   = data[n_train + n_test:]  # remaining 5%
    return train_data, val_data, test_data


# ------------------ §7.2: prompt formatting ------------------
def format_input(entry: dict) -> str:
    txt = ("Below is an instruction that describes a task. "
           "Write a response that appropriately completes the request."
           f"\n\n### Instruction:\n{entry['instruction']}")
    if entry.get("input"):
        txt += f"\n\n### Input:\n{entry['input']}"
    return txt


# ------------------ §7.3: dataset & collate ------------------
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encoded = []
        for e in data:
            full = format_input(e) + f"\n\n### Response:\n{e['output']}"
            self.encoded.append(tokenizer.encode(full))
    def __len__(self): return len(self.encoded)
    def __getitem__(self, i): return self.encoded[i]

def custom_collate_fn(
    batch,
    pad_token_id: int,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str = "cpu",
):
    L = max(len(x) + 1 for x in batch)
    X, Y = [], []
    for item in batch:
        item = item + [pad_token_id]                    # ensure one pad
        padded = item + [pad_token_id] * (L - len(item))
        x = torch.tensor(padded[:-1])                   # inputs
        y = torch.tensor(padded[1:])                    # targets (shifted)

        # mask all but first pad in targets
        pad_idx = (y == pad_token_id).nonzero().squeeze()
        if pad_idx.numel() > 1:
            y[pad_idx[1:]] = ignore_index

        if allowed_max_length is not None:
            x = x[:allowed_max_length]
            y = y[:allowed_max_length]

        X.append(x); Y.append(y)

    return torch.stack(X).to(device), torch.stack(Y).to(device)


# ------------------ main ------------------
if __name__ == "__main__":
    # 1) Load JSON and split -> defines train_data / val_data / test_data
    data = download_and_load_file(FILE, URL)
    train_data, val_data, test_data = partition_dataset(data)
    print(f"Splits -> train:{len(train_data)}  val:{len(val_data)}  test:{len(test_data)}")

    # 2) Tokenizer & pad id
    tokenizer = tiktoken.get_encoding("gpt2")
    PAD_ID = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    # 3) Datasets
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset   = InstructionDataset(val_data, tokenizer)
    test_dataset  = InstructionDataset(test_data, tokenizer)

    # 4) Collate bound to our device
    collate = partial(
        custom_collate_fn,
        device=device,
        pad_token_id=PAD_ID,
        ignore_index=-100,
        allowed_max_length=1024,   # set None to disable truncation
    )

    # 5) DataLoaders
    torch.manual_seed(123)
    num_workers = 0
    batch_size  = 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=collate, shuffle=True, drop_last=True,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              collate_fn=collate, shuffle=False, drop_last=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              collate_fn=collate, shuffle=False, drop_last=False,
                              num_workers=num_workers)

    # Quick shape check (first few batches)
    print("Train loader:")
    for i, (inp, tgt) in enumerate(train_loader):
        print(inp.shape, tgt.shape)
        if i >= 5:
            break

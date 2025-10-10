from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

# We’ll use tiktoken (GPT-2 BPE) to encode text into token IDs.
# pip install tiktoken
import tiktoken


# =============== Dataset ===============
class AIDatasetV1(Dataset):
    """
    Next-token prediction dataset.

    Given a long list of token IDs (e.g., from tiktoken.encode),
    produce many (input_ids, target_ids) pairs using a sliding window:

      X = ids[i : i+max_length]
      Y = ids[i+1 : i+max_length+1]

    - max_length: context length / block size
    - stride: how far we move the window each step
              (stride < max_length => overlapping windows)
    """

    def __init__(self, text: str, tokenizer, max_length: int, stride: int = 128):
        # 1) Encode text -> token IDs
        #    Allow <|endoftext|> if you include that symbol in your data.
        self.ids: List[int] = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # 2) Precompute window start indices.
        #    We need at least (max_length + 1) tokens to form one (X, Y) pair.
        self.max_length = int(max_length)
        self.stride = int(stride)
        limit = len(self.ids) - (self.max_length + 1) + 1
        if limit < 0:
            limit = 0
        self.starts: List[int] = list(range(0, limit, self.stride))

    def __len__(self) -> int:
        # Number of windows we can extract
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract one (X, Y) pair
        i = self.starts[idx]
        x = self.ids[i : i + self.max_length]
        y = self.ids[i + 1 : i + self.max_length + 1]

        # Convert to Long tensors (indices for nn.Embedding later)
        x_t = torch.tensor(x, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.long)
        return x_t, y_t


# =============== DataLoader factory ===============
def spawn_dataloader(
    text: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader over AIDatasetV1.

    - shuffle=True for training; set False for validation
    - drop_last=True to keep all batches the same size (training convenience)
    """
    dataset = AIDatasetV1(text, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


# =============== Demo / quick test ===============
if __name__ == "__main__":
    # 1) Initialize tokenizer (GPT-2 BPE)
    enc = tiktoken.get_encoding("gpt2")

    # 2) Load text (replace with your path). Keep it small for a quick test.
    #    Example fallback string if the file isn't present.
    try:
        with open("the-verdict.txt", "r", encoding="utf-8") as f:
            raw_txt = f.read()
    except FileNotFoundError:
        raw_txt = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

    # 3) Create DataLoader
    #    For a tiny demo, use small max_length/stride so we get several windows.
    dl = spawn_dataloader(
        text=raw_txt,
        tokenizer=enc,
        batch_size=2,
        max_length=8,
        stride=4,
        shuffle=False,   # set True for training
        drop_last=False  # keep last small batch in this demo
    )

    # 4) Inspect first batch and verify shift relationship (Y is X shifted by 1)
    it = iter(dl)
    try:
        xb, yb = next(it)
    except StopIteration:
        raise RuntimeError(
            "No batches produced. Your text may be too short for the chosen max_length."
        )

    print("Batch X shape:", xb.shape)  # [batch, seq_len]
    print("Batch Y shape:", yb.shape)  # [batch, seq_len]
    print("First X row:", xb[0].tolist())
    print("First Y row:", yb[0].tolist())

    # Sanity check: for each row, X[:, 1:] == Y[:, :-1]
    assert torch.equal(xb[:, 1:], yb[:, :-1]), "Shift check failed"
    print(" Shift check passed")

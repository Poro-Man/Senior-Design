# chapter_7_3.py
# Section 7.3 — Instruction Dataset + Custom Collate

import torch
from torch.utils.data import Dataset
import tiktoken


# ---------- from 7.2 (replace with your actual implementation) ----------
def format_input(entry: dict) -> str:
    """
    Alpaca-style prompt builder. Replace with your 7.2 version if different.
    """
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_part = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_part
# --------------------------------------------------------------------------


# =========================
# Listing 7.4 — Dataset
# =========================
class InstructionDataset(Dataset):
    """
    Pre-tokenizes each entry into one text:
        [formatted instruction+input] + "\n\n### Response:\n" + [output]
    Stores encoded token IDs for fast collation.
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)         # A
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def get_pad_token_id_gpt2() -> int:
    """
    Use <|endoftext|> as padding for GPT-2 BPE.
    """
    tok = tiktoken.get_encoding("gpt2")
    return tok.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


# ===========================================
# Draft 1 — pad to same length, return inputs
# ===========================================
def custom_collate_draft_1(
    batch,
    pad_token_id: int = 50256,
    device: str = "cpu",
):
    # A: Find the longest sequence (plus one extra for shifting later)
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst = []
    for item in batch:
        # B: Pad and prepare inputs
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # C: Remove extra padded token earlier (use all but last)
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # D: Stack and move to device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


# ==================================================
# Draft 2 — build inputs and targets with 1-token shift
# ==================================================
def custom_collate_draft_2(
    batch,
    pad_token_id: int = 50256,
    device: str = "cpu",
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # A: inputs = all but last
        inputs = torch.tensor(padded[:-1])
        # B: targets = all but first (right shift)
        targets = torch.tensor(padded[1:])

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# ==================================
# Listing 7.5 — final collate fn
# ==================================
def custom_collate_fn(
    batch,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str = "cpu",
):
    """
    Returns (inputs, targets) with:
      - padding to batch max length (plus 1),
      - 1-token shift (teacher forcing),
      - pad regions in targets replaced by ignore_index (except the first pad),
      - optional truncation to allowed_max_length,
      - tensors on `device`.
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # Pad sequences to max length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # Truncate last token for inputs / shift +1 for targets
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # A: Replace all but first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # B: Optional truncation to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# ======================
# Self-test / examples
# ======================
if __name__ == "__main__":
    # Show the pad id we’ll use
    pad_id = get_pad_token_id_gpt2()
    print("GPT-2 <|endoftext|> id:", pad_id)

    # Example small batch to mirror book figures
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)

    # Draft 1
    print("\nDraft 1 (inputs only):")
    print(custom_collate_draft_1(batch, pad_token_id=pad_id))

    # Draft 2
    print("\nDraft 2 (inputs, targets):")
    x2, y2 = custom_collate_draft_2(batch, pad_token_id=pad_id)
    print(x2)
    print(y2)

    # Final
    print("\nFinal collate (inputs, targets with ignore_index on extra pads):")
    X, Y = custom_collate_fn(batch, pad_token_id=pad_id)
    print(X)
    print(Y)

    # Cross-entropy example (book parity)
    logits_1 = torch.tensor(
        [[-1.0, 1.0],  # predictions for 1st token
         [-0.5, 1.5]]  # predictions for 2nd token
    )
    targets_1 = torch.tensor([0, 1])  # correct indices
    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print("\nloss_1:", loss_1)

    logits_2 = torch.tensor(
        [[-1.0, 1.0],
         [-0.5, 1.5],
         [-0.5, 1.5]]   # A: new 3rd token prediction
    )
    targets_2 = torch.tensor([0, 1, 1])
    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print("loss_2:", loss_2)

    # Replace the 3rd target with -100: loss matches loss_1
    targets_3 = torch.tensor([0, 1, -100])
    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print("loss_3:", loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)

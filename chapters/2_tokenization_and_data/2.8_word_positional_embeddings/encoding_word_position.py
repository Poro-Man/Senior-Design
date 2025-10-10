# encoding_word_position.py
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken


class AIDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.trgt_ids = []

        # Tokenize the text
        tkn_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Sliding window approach to create input-target pairs
        for i in range(0, len(tkn_ids) - max_length, stride):
            in_chunks = tkn_ids[i: i + max_length]
            trgt_chunks = tkn_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(in_chunks, dtype=torch.long))
            self.trgt_ids.append(torch.tensor(trgt_chunks, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.trgt_ids[idx]


def spawn_dataloader(
    text,
    tokenizer,
    batch_size=4,
    max_length=512,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    dataset = AIDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


# --------------------------
# quick test (now shows §2.8)
# --------------------------
if __name__ == "__main__":
    # Init tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load text
    try:
        with open("the-verdict.txt", "r", encoding="utf-8") as file:
            raw_txt = file.read()
    except FileNotFoundError:
        raw_txt = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

    # Spawn dataloader
    max_length = 8  # small so we can see shapes clearly
    dataloader = spawn_dataloader(
        raw_txt,
        tokenizer,
        batch_size=4,
        max_length=max_length,
        stride=max_length,   # non-overlapping for the demo
        shuffle=False,
        drop_last=False,
    )

    # Inspect first batch
    data_iter = iter(dataloader)
    x_ids, y_ids = next(data_iter)   # x_ids, y_ids: [batch, seq_len] integer IDs
    print("IDs X shape:", x_ids.shape)
    print("IDs Y shape:", y_ids.shape)

    # ===== §2.8: Word + Positional Embeddings =====
    vocab_size = tokenizer.n_vocab           # size of BPE vocabulary
    emb_dim = 128                            # embedding dimension (example size)

    # Word/token embedding: maps token IDs -> dense vectors
    token_embedding = torch.nn.Embedding(vocab_size, emb_dim)

    # Positional embedding (learned): one vector per position [0..seq_len-1]
    pos_embedding = torch.nn.Embedding(max_length, emb_dim)

    # Build position indices for this batch: [batch, seq_len] = [[0,1,2,...], ...]
    seq_len = x_ids.size(1)
    positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(x_ids.size(0), -1)

    # Look up embeddings
    x_tok = token_embedding(x_ids)           # [batch, seq_len, emb_dim]
    x_pos = pos_embedding(positions)         # [batch, seq_len, emb_dim]

    # Final input representation to the model = token + position
    x_repr = x_tok + x_pos                   # [batch, seq_len, emb_dim]

    print("Token emb shape:", x_tok.shape)
    print("Pos emb shape:  ", x_pos.shape)
    print("Sum shape:       ", x_repr.shape)

import torch
import torch.nn as nn
import tiktoken


class toiletGPT(nn.Module):
    def __init__(self, cfg):
        """
        cfg keys expected (as in your snippet):
          "vocab_size", "context_length", "emb_dim",
          "n_heads", "n_layers", "drop_rate", "qkv_bias"
        """
        super().__init__()
        # Embedding layers
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # token embedding
        self.pos_emb = nn.Parameter(
            torch.zeros(1, cfg["context_length"], cfg["emb_dim"])
        )  # learned position embedding (slice in forward)

        # Dropout after adding token + position embeddings
        self.drop = nn.Dropout(cfg["drop_rate"])

        # Stack of placeholder transformer blocks
        self.trf_blocks = nn.Sequential(
            *[ToiletTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final norm + output (LM) head
        self.final_norm = ToiletLayerNorm(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Nice init for positional embeddings (optional)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, in_idx: torch.LongTensor):
        """
        in_idx: (batch_size, seq_length) token IDs
        returns: logits (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = in_idx.shape

        # (B, T, C)
        tok_embds = self.tok_emb(in_idx)

        # Slice the learned positional embeddings to current T (do NOT call as a function)
        # (1, T, C) broadcasts across batch
        pos_embds = self.pos_emb[:, :seq_length, :]

        x = tok_embds + pos_embds                  # (B, T, C)
        x = self.drop(x)                            # (B, T, C)
        x = self.trf_blocks(x)                      # (B, T, C)  — identity blocks for §4.1
        x = self.final_norm(x)                      # (B, T, C)
        logits = self.out_head(x)                   # (B, T, V)
        return logits


class ToiletTransformerBlock(nn.Module):
    """
    Placeholder transformer block so §4.1 runs.
    Later you’ll replace with: LN -> (masked MHA) -> residual, LN -> MLP -> residual.
    """
    def __init__(self, cfg):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)


class ToiletLayerNorm(nn.Module):
    """
    Thin wrapper around nn.LayerNorm to keep your naming.
    """
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        return self.norm(x)


if __name__ == "__main__":
    # Build the example batch exactly like in your snippet
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    # Config (your keys & values)
    torch.manual_seed(123)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,       # Vocabulary size
        "context_length": 1024,    # Context length (max sequence length)
        "emb_dim": 768,            # Embedding/channel dimension
        "n_heads": 12,             # Number of attention heads (unused here)
        "n_layers": 12,            # Number of layers (identity blocks for now)
        "drop_rate": 0.1,          # Dropout rate
        "qkv_bias": False          # Query-Key-Value bias (unused here)
    }

    model = toiletGPT(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)  # expected: (2, T, 50257)
    print(logits)

import re
from torch.nn.modules import LayerNorm
from resources.Multihead import MultiHeadAttn
from resources.Glue import ConveyorBelt
from resources.optimus import AutoBlocks
import torch
import torch.nn as nn


class Toilet(nn.Module):
    """
    A GPT-style transformer model for text generation.
    Implements a decoder-only transformer architecture similar to GPT-2.
    """
    def __init__(self, cfg):
        """
        Initialize the model with given configuration.
        Args:
            cfg (dict): Configuration dictionary containing model hyperparameters
        """
        super().__init__()
        # Token embedding layer: converts token IDs to vectors
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Positional embedding layer: adds position information
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Dropout layer to prevent overfitting
        self.drop_emb = nn.Dropout(cfg["drop_rate"])


        self.trf_blocks = nn.Sequential(
            *[AutoBlocks(cfg) for _ in range(cfg["n_layers"])])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        # Output projection layer to vocabulary size
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Forward pass of the model.
        Args:
            in_idx (torch.Tensor): Input tensor of token indices [batch_size, seq_len]
        Returns:
            torch.Tensor: Output logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = in_idx.size()
        # Convert tokens to embeddings
        tok_embeds = self.tok_emb(in_idx)

        # Generate and add positional embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        # Apply embedding dropout
        x = self.drop_emb(x)
        # Pass through transformer blocks
        x = self.trf_blocks(x)
        # Apply final normalization
        x = self.final_norm(x)
        # Project to vocabulary size
        logits = self.out_head(x)
        return logits


if __name__ == "__main__":
    # Configuration for a GPT-2 small model (124M parameters)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Number of unique tokens in vocabulary
        "context_length": 1024,     # Maximum sequence length (tokens per input)
        "emb_dim": 768,             # Embedding dimension (model width)
        "n_heads": 12,              # Number of attention heads
        "n_layers": 12,             # Number of transformer blocks
        "drop_rate": 0.1,           # Dropout probability
        "qkv_bias": False           # Whether to use bias in Q/K/V projections
    }

    # Set random seed for reproducibility
    torch.manual_seed(123)
    # Initialize model with configuration
    model = Toilet(GPT_CONFIG_124M)
    # Create random input batch for testing
    batch = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2, 4))  # (batch_size=2, seq_len=4)

    # Forward pass
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    # Calculate and display model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    # Calculate parameters with weight tying consideration
    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    # Calculate model size in MB
    total_size_bytes = total_params * 4  # 4 bytes per parameter (float32)
    total_size_mb = total_size_bytes / (1024 * 1024) 
    print(f"Total size of the model: {total_size_mb:.2f} MB")
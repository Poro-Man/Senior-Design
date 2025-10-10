from torch.nn.modules import LayerNorm
from resources.Multihead import MultiHeadAttn
from resources.Glue import ConveyorBelt
import torch
import torch.nn as nn

class AutoBlocks(nn.Module):
    """
    Implements a single Transformer block as used in GPT-style architectures.
    This block consists of:
      - Multi-head self-attention (with causal masking, if implemented in MultiHeadAttn)
      - Layer normalization (pre-norm)
      - Position-wise feed-forward network (MLP)
      - Dropout for regularization
      - Residual (shortcut) connections after each sub-layer

    Args:
        cfg (dict): Model configuration dictionary. Must contain:
            - "emb_dim": Embedding dimension (input/output size of block)
            - "context_length": Maximum sequence length
            - "drop_rate": Dropout probability
            - "n_heads": Number of attention heads
            - "qkv_bias": Whether to use bias in Q/K/V projections
    """
    def __init__(self, cfg):
        super().__init__()
        # Multi-head self-attention module.
        # Allows each token to attend to all previous tokens (if causal mask is used).
        # Output shape: (batch, seq_len, emb_dim)
        self.attn = MultiHeadAttn(
            d_in=cfg["emb_dim"],            # Input embedding dimension
            d_out=cfg["emb_dim"],           # Output embedding dimension (must match for residual)
            context_length=cfg["context_length"],  # Max sequence length for mask
            dropout=cfg["drop_rate"],       # Dropout on attention weights
            num_heads=cfg["n_heads"],       # Number of parallel attention heads
            qkv_bias=cfg["qkv_bias"]        # Whether to use bias in Q/K/V projections
        )

        # Position-wise feed-forward network (MLP).
        # Expands embedding dimension (typically 4x), applies GELU, then projects back.
        self.ff = ConveyorBelt(cfg)

        # Layer normalization before attention and MLP (pre-norm).
        # Helps stabilize training and gradient flow.
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout applied after attention and MLP, before adding residual.
        self.shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass for a single Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, emb_dim)

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # --- Multi-head Attention Sub-layer ---
        # Save input for residual connection
        short = x
        # Apply layer normalization (pre-norm)
        x = self.norm1(x)
        # Compute multi-head self-attention
        x = self.attn(x)
        # Apply dropout to attention output
        x = self.shortcut(x)
        # Add residual connection (original input)
        x = x + short

        # --- Feed-Forward Sub-layer ---
        # Save output for next residual connection
        short = x
        # Apply layer normalization (pre-norm)
        x = self.norm2(x)
        # Pass through position-wise feed-forward network (MLP)
        x = self.ff(x)
        # Apply dropout to MLP output
        x = self.shortcut(x)
        # Add residual connection
        x = x + short

        return x

if __name__ == "__main__":
    # Example configuration for a GPT-2 124M-style block
    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Number of unique tokens in vocabulary
        "context_length": 1024,     # Maximum sequence length (tokens per input)
        "emb_dim": 768,             # Embedding dimension (model width)
        "n_heads": 12,              # Number of attention heads
        "n_layers": 12,             # Number of transformer blocks (not used here)
        "drop_rate": 0.1,           # Dropout probability
        "qkv_bias": False           # Whether to use bias in Q/K/V projections
    }

    torch.manual_seed(123)
    # Create a random input tensor: batch size 2, sequence length 4, embedding dim 768
    x = torch.rand(2, 4, 768)
    # Instantiate a single transformer block
    block = AutoBlocks(GPT_CONFIG_124M)
    # Forward pass through the block
    output = block(x)
    # Print input and output shapes to verify shape preservation
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
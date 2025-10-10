import re
from torch.nn.modules import LayerNorm
from resources.Multihead import MultiHeadAttn
from resources.Glue import ConveyorBelt
from resources.optimus import AutoBlocks
import torch
import torch.nn as nn
import tiktoken


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

    @staticmethod
    def generate_text(model, idx, max_new_tokens, context_size):
        """
        Generate text using the model.
        Args:
            model: The Toilet model instance
            idx (torch.Tensor): Input token indices
            max_new_tokens (int): Number of new tokens to generate
            context_size (int): Maximum context size to use
        Returns:
            torch.Tensor: Generated token indices
        """
        model.eval()  # Set to evaluation mode
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]  # Crop context if needed
            with torch.no_grad():
                logits = model(idx_cond)
            
            logits = logits[:, -1, :]  # Focus on last token
            probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # Greedily pick the next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence
        return idx


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # Use tiktoken for GPT-2
    enc = tiktoken.get_encoding("gpt2")

    model = Toilet(GPT_CONFIG_124M)

    start_context = "Hello, I am"
    encoded = enc.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = model.generate_text(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = enc.decode(out.squeeze(0).tolist())
    print(decoded_text)
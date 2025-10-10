import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)                                    # #A

        # Upper-triangular mask (True above the diagonal ⇒ future positions)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1) # #B
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, num_tokens, d_in)
        b, num_tokens, d_in = x.shape                                         # #C

        keys    = self.W_key(x)        # (b, T, d_out)
        queries = self.W_query(x)      # (b, T, d_out)
        values  = self.W_value(x)      # (b, T, d_out)

        # Scaled dot-product logits
        attn_scores = queries @ keys.transpose(1, 2)                          # #C  (b, T, T)

        # Causal mask: block attending to future tokens
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],                       # #D
            -torch.inf
        )

        # Softmax → attention weights, then dropout
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vectors
        context_vec = attn_weights @ values    # (b, T, d_out)
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)

    T, d_in, d_out = 6, 3, 2
    inputs = torch.randn(T, d_in)                  # (T, d_in)
    batch = torch.stack((inputs, inputs), dim=0)   # (batch=2, T, d_in)

    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
    context_vecs = ca(batch)

    print("context_vecs.shape:", context_vecs.shape)

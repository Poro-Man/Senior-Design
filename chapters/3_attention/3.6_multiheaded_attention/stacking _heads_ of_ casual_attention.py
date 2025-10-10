import torch
import torch.nn as nn
from casual_attention import CausalAttention 

class MultiHeadedAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Each head is a CausalAttention block producing (B, T, d_out)
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )
        self.num_heads = num_heads
        self.d_out = d_out

    def forward(self, x):
        # x: (B, T, d_in)
        # Concatenate head outputs along feature dim -> (B, T, num_heads * d_out)
        return torch.cat([h(x) for h in self.heads], dim=-1)


if __name__ == "__main__":
    torch.manual_seed(123)


    T, d_in, d_out = 6, 3, 2
    inputs = torch.randn(T, d_in)                  # (T, d_in)
    batch = torch.stack((inputs, inputs), dim=0)   # (B=2, T, d_in)

    context_length = batch.shape[1]                # number of tokens T
    mha = MultiHeadedAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)

    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)  # -> torch.Size([2, 6, 4]) since 2 heads * d_out(2)

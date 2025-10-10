# self_attention_v2.py
import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    """
    Scaled dot-product self-attention using nn.Linear for Wq/Wk/Wv.
    Input:  x of shape [T, d_in]  (T = seq len)
    Output: (context_vec, attn_weights)
        context_vec   [T, d_out]
        attn_weights  [T, T]
    """
    def __init__(self, d_in, d_out, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor):
        # x: [T, d_in]
        keys    = self.W_key(x)      # [T, d_out]
        queries = self.W_query(x)    # [T, d_out]
        values  = self.W_value(x)    # [T, d_out]

        # scores: [T, T]
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / (self.d_out ** 0.5), dim=-1)

        # context: [T, d_out]
        context_vec = attn_weights @ values
        return context_vec, attn_weights


if __name__ == "__main__":
    torch.manual_seed(0)
    T, d_in, d_out = 6, 3, 2
    x = torch.randn(T, d_in)                   # pretend embedded tokens
    attn = SelfAttention_v2(d_in, d_out)
    ctx, w = attn(x)
    print("x:", x.shape)                       # [6, 3]
    print("ctx:", ctx.shape)                   # [6, 2]
    print("w:", w.shape)                       # [6, 6]
    print("row0 sums to:", w[0].sum().item())  # ~1.0

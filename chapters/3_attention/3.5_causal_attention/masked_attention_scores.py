import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec  = attn_weights @ values
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(789)

    T, d_in, d_out = 6, 3, 2
    inputs = torch.randn(T, d_in)

    sa_v2 = SelfAttention_v2(d_in, d_out)

    # Reuse query/key projections from sa_v2
    queries = sa_v2.W_query(inputs)
    keys    = sa_v2.W_key(inputs)

    attn_scores  = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

    print("attention weights:\n", attn_weights)

    #implemntation of causal mask
    context_length = attn_scores.shape[0]
    causal_mask = torch.tril(torch.ones(context_length, context_length))
    print(causal_mask)

    causal_mask= attn_weights*causal_mask
    print(causal_mask)

    row_sums = causal_mask.sum(dim=1, keepdim=True)
    masked_simple_norm = causal_mask / row_sums
    print(masked_simple_norm)
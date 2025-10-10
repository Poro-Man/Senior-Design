# sel_attn.py
import torch.nn as nn
import torch

class SelfAttenion_v1(nn.Module):
    """
    Single-head self-attention with trainable W_q, W_k, W_v (section 3.4).
    Expects a 2D input: [T, d_in], where T = sequence length.
    Returns:
      context_vecs: [T, d_out] context vectors
      attn_weights: [T, T] attention matrix (row i attends over all j)
    """
    def __init__(self, d_in, d_out):
        super(SelfAttenion_v1, self).__init__()
        # Trainable projection matrices for queries, keys, values (3.4)
        # Small init helps keep scores in a reasonable range at start.
        self.W_q = nn.Parameter(torch.randn(d_in, d_out) * 0.02)
        self.W_k = nn.Parameter(torch.randn(d_in, d_out) * 0.02)
        self.W_v = nn.Parameter(torch.randn(d_in, d_out) * 0.02)

    def forward(self, inputs):
        """
        inputs: [T, d_in]
        """
        # 1) Project to Q, K, V (trainable linear projections; sec. 3.4)
        keys    = inputs @ self.W_k          # [T, d_out]
        queries = inputs @ self.W_q          # [T, d_out]
        values  = inputs @ self.W_v          # [T, d_out]

        # 2) Raw attention scores via dot product QK^T (3.4.1)
        attn_scores = queries @ keys.T       # [T, T]

        # 3) Scale by sqrt(d_k) before softmax (scaled dot-product)
        scale = keys.shape[-1] ** 0.5        # sqrt(d_out)
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)  # [T, T]

        # 4) Weighted sum over values to get context vectors
        context_vecs = attn_weights @ values  # [T, d_out]
        return context_vecs, attn_weights


if __name__ == "__main__":
    # Quick sanity check mirroring the book’s flow in 3.4
    torch.manual_seed(123)

    # Toy example: 6 tokens (T=6), 3-dim input embeddings, project to 2-dim
    T, d_in, d_out = 6, 3, 2
    # Random stand-in for embedded tokens (in Ch. 2 you created embeddings already)
    inputs = torch.randn(T, d_in)

    sa_1v = SelfAttenion_v1(d_in, d_out)
    context, weights = sa_1v(inputs)

    print("inputs.shape:", inputs.shape)         # [6, 3]
    print("context.shape:", context.shape)       # [6, 2]
    print("attn_weights.shape:", weights.shape)  # [6, 6]
    print("\nFirst row of attention weights (token 0 attends over all tokens):")
    print(weights[0])

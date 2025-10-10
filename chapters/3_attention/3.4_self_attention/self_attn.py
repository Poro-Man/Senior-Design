import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

x_2 = inputs[1] # The 2nd input element
d_in = inputs.shape[1] # Input embedding dimension
d_out = 2   # Output embedding dimension

torch.manual_seed(123)
W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_q 

key_2 = x_2 @ W_k

value_2 = x_2 @ W_v

print(query_2)


keys = inputs @ W_k
values = inputs @ W_v
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

keys.shape: torch.Size([6, 2])
values.shape: torch.Size([6, 2])

keys_2 = keys[1] 
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T 
print(attn_scores_2)

d_k = keys.shape[-1]  # d_k is the dimension of the keys
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)   # Scale dot-product
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values # Weighted sum of the values
print(context_vec_2)
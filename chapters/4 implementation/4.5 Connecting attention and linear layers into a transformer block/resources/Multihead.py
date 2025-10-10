import torch
import torch.nn as nn


class MultiHeadAttn(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize the Multi-Head Attention module.
        
        Args:
            d_in (int): Input dimension (embedding size)
            d_out (int): Output dimension (should be divisible by num_heads)
            context_length (int): Maximum sequence length for causal masking
            dropout (float): Dropout probability for attention weights
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to include bias in query, key, value projections
        """
        super().__init__()

        # Ensure output dimension is divisible by number of heads
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        # Store dimensions
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension per attention head
        
        # Linear projections for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Output projection to combine all heads
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (upper triangular matrix with 1s above diagonal)
        # This prevents tokens from attending to future tokens
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        Forward pass of the Multi-Head Attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, d_out)
        """
        # Get input dimensions
        b, num_tokens, d_in = x.shape
        
        # Generate keys, queries, and values for all heads simultaneously
        # Shape: (batch_size, num_tokens, d_out) -> (batch_size, num_heads, num_tokens, head_dim)
        keys    = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores using scaled dot-product attention
        # Shape: (batch_size, num_heads, num_tokens, num_tokens)
        attn_scores  = (queries @ keys.transpose(2,3))
        
        # Extract the relevant portion of the causal mask for current sequence length
        mask_boolean = self.mask[:num_tokens, :num_tokens]

        # Apply causal mask by setting future positions to negative infinity
        # This ensures softmax gives 0 probability to future tokens
        attn_scores.masked_fill_(mask_boolean.bool(), -torch.inf)

        # Apply softmax to get attention weights, scaled by sqrt(head_dim) for stability
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # Apply dropout to attention weights for regularization
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum of values using attention weights
        # Shape: (batch_size, num_heads, num_tokens, head_dim) -> (batch_size, num_tokens, d_out)
        context_vecs = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        
        # Apply final output projection to combine information from all heads
        context_vecs = self.out_proj(context_vecs)

        return context_vecs
    
    
if __name__ == "__main__":
    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                        [0.8993, 0.0390, 0.9268, 0.7388],
                        [0.7179, 0.7058, 0.9156, 0.4340]],
                        [[0.0772, 0.3565, 0.1479, 0.5331],
                        [0.4066, 0.2318, 0.4545, 0.9737],
                        [0.4606, 0.5159, 0.4220, 0.5786]]]])
    print(a @ a.transpose(2, 3))

    first_head = a[0, 0, :, :]
    first_res = first_head @ first_head.T
    print("First head:\n", first_res)
    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print("\nSecond head:\n", second_res)


    T, d_in, d_out = 6, 3, 2
    inputs = torch.randn(T, d_in)                  # (T, d_in)
    batch = torch.stack((inputs, inputs), dim=0)   # (batch=2, T, d_in)

    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttn(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
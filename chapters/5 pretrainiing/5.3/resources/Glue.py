import torch
import torch.nn as nn

class Elmers(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) activation function implementation.
    
    GELU is a smooth, non-monotonic activation function that combines 
    properties of ReLU and dropout. It's commonly used in transformer 
    architectures like GPT and BERT.
    
    Mathematical formula:
    GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    This is an approximation of the exact GELU function:
    GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
    """
    
    def __init__(self):
        """Initialize the GELU activation function."""
        super().__init__()

    def forward(self, x):
        """
        Apply GELU activation to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of any shape
            
        Returns:
            torch.Tensor: Output tensor with GELU activation applied element-wise
            
        The formula breakdown:
        - 0.5 * x: Scale the input by half
        - √(2/π) ≈ 0.7978: Scaling constant for the tanh approximation
        - 0.044715 * x³: Cubic term that makes GELU different from standard tanh
        - tanh(...): Hyperbolic tangent for smooth transitions
        - (1 + tanh(...)): Shift range from [-1,1] to [0,2]
        - Final multiplication gives the GELU output
        """
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))
    

class ConveyorBelt(nn.Module):
    """
    Feed-Forward Network (FFN) layer used in transformer architectures.
    
    This implements the position-wise feed-forward network that appears
    in each transformer block. It consists of:
    1. Linear expansion layer (embedding_dim -> 4 * embedding_dim)
    2. GELU activation function
    3. Linear projection layer (4 * embedding_dim -> embedding_dim)
    
    The 4x expansion is a common pattern in transformers (GPT, BERT, etc.)
    that allows the model to learn complex non-linear transformations.
    """
    
    def __init__(self, cfg):
        """
        Initialize the Feed-Forward Network.
        
        Args:
            cfg (dict): Configuration dictionary containing:
                - "emb_dim": Embedding dimension (input/output size)
                
        The network structure:
        - First linear layer expands dimensionality by 4x
        - GELU activation provides non-linearity
        - Second linear layer projects back to original dimension
        """
        super().__init__()
        
        # Sequential container for the three-layer FFN
        self.layers = nn.Sequential(
            # Expansion layer: project from embedding_dim to 4 * embedding_dim
            # This creates a higher-dimensional representation for complex transformations
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            
            # GELU activation function for smooth, differentiable non-linearity
            # GELU is preferred over ReLU in many transformer implementations
            Elmers(),
            
            # Projection layer: compress back from 4 * embedding_dim to embedding_dim
            # This maintains the residual connection compatibility
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim)
                             Typically (batch_size, seq_len, emb_dim)
                             
        Returns:
            torch.Tensor: Output tensor of same shape as input (..., emb_dim)
                         
        The computation flow:
        1. x -> Linear(emb_dim, 4*emb_dim) -> expanded representation
        2. expanded -> GELU() -> activated representation  
        3. activated -> Linear(4*emb_dim, emb_dim) -> final output
        
        This allows each position to be processed independently (position-wise)
        while learning complex non-linear transformations.
        """
        return self.layers(x)
    
if __name__ == "__main__":
    """
    Test script to demonstrate the Feed-Forward Network functionality.
    
    This section tests the ConveyorBelt (FFN) with GPT-124M configuration
    to ensure proper input/output shape preservation.
    """
    
    # Configuration dictionary matching GPT-124M architecture
    GPT_CONFIG_124M = {
        "vocab_size": 50257,       # Vocabulary size (number of unique tokens)
        "context_length": 1024,    # Context length (max sequence length)
        "emb_dim": 768,            # Embedding/channel dimension (model width)
        "n_heads": 12,             # Number of attention heads (unused in FFN)
        "n_layers": 12,            # Number of transformer layers (unused here)
        "drop_rate": 0.1,          # Dropout rate for regularization (unused here)
        "qkv_bias": False          # Query-Key-Value bias flag (unused in FFN)
    }
    
    # Create an instance of the Feed-Forward Network
    ffn = ConveyorBelt(GPT_CONFIG_124M)
    
    # Create test input tensor
    # Shape: (batch_size=2, sequence_length=3, embedding_dim=768)
    # This simulates a batch of 2 sequences, each with 3 tokens,
    # where each token is represented by a 768-dimensional embedding
    x = torch.rand(2, 3, 768)
    
    # Forward pass through the FFN
    output = ffn(x)
    
    # Verify output shape matches input shape (crucial for residual connections)
    print(f"Input shape:  {x.shape}")      # Expected: torch.Size([2, 3, 768])
    print(f"Output shape: {output.shape}") # Expected: torch.Size([2, 3, 768])
    
    # Additional verification
    assert output.shape == x.shape, f"Shape mismatch! Input: {x.shape}, Output: {output.shape}"
    print("✓ Shape preservation test passed!")
    
    # Test GELU activation separately
    gelu = Elmers()
    test_vals = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    gelu_output = gelu(test_vals)
    print(f"GELU test values: {test_vals}")
    print(f"GELU outputs:     {gelu_output}")
    print("✓ GELU activation test completed!")

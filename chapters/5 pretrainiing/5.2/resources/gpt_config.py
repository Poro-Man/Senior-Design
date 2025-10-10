 GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Number of unique tokens in vocabulary
        "context_length": 1024,     # Maximum sequence length (tokens per input)
        "emb_dim": 768,             # Embedding dimension (model width)
        "n_heads": 12,              # Number of attention heads
        "n_layers": 12,             # Number of transformer blocks (not used here)
        "drop_rate": 0.1,           # Dropout probability
        "qkv_bias": False           # Whether to use bias in Q/K/V projections
    }
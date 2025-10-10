import torch
from resources.toilet import Toilet
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, 
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, 
    "qkv_bias": False
}
model = Toilet(GPT_CONFIG_124M)

torch.manual_seed(123)
model = Toilet(GPT_CONFIG_124M)
model.eval()

def the_numbers_mason(text, tokenizer):
    # Fix: encode instead of encoded
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def what_do_they_mean(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

if __name__ == "__main__":
    start_context = "The numbers, Mason. What do they mean?"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Use model's generate_text method directly
    token_ids = model.generate_text(
        model=model,
        idx=the_numbers_mason(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"])
    print("Output text:\n", what_do_they_mean(token_ids, tokenizer))
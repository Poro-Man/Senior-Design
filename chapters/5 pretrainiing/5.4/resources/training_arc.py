import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from resources.work import Toilet
from resources.data import spawn_dataloader

# -------------------------
# Config 
# -------------------------
# Configuration dictionary defining architecture parameters for a smaller GPT-2 variant
# Matches OpenAI's 124M parameter model specifications
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Size of the token vocabulary
    "context_length": 256,    # Maximum sequence length for input
    "emb_dim": 768,          # Dimension of embeddings and hidden states
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer layers
    "drop_rate": 0.1,        # Dropout probability
    "qkv_bias": False,       # Whether to use bias in query, key, value projections
}

# -------------------------------------------------
#  helpers — text <-> tokens and generation
# ------------------------------------------------
def get_device():
    """
    Determines the available computation device (CUDA GPU or CPU).
    Returns:
        torch.device: The device to be used for model computations
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Running on CPU (no CUDA detected).")
    return device
        
device = get_device()

def text_to_token_ids(text: str, tokenizer):
    """
    Converts input text to token IDs using the provided tokenizer.
    Args:
        text (str): Input text to tokenize
        tokenizer: GPT-2 tokenizer instance
    Returns:
        torch.Tensor: Tensor of token IDs with batch dimension [1, sequence_length]
    """
    ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer):
    """
    Converts token IDs back to text.
    Args:
        token_ids (torch.Tensor): Tensor of token IDs [1, sequence_length]
        tokenizer: GPT-2 tokenizer instance
    Returns:
        str: Decoded text
    """
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)

@torch.no_grad()
def generate_text_simple(model: nn.Module, idx: torch.Tensor,
                         max_new_tokens: int, context_size: int):
    """
    Generates text using greedy sampling (taking most likely token each time).
    Args:
        model (nn.Module): The transformer model
        idx (torch.Tensor): Initial context tokens
        max_new_tokens (int): Number of tokens to generate
        context_size (int): Maximum context window size
    Returns:
        torch.Tensor: Generated token sequence including initial context
    """
    model.eval()
    idx = idx.to(next(model.parameters()).device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

# -------------------------------------------------
# 5.1 loss utilities
# -------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculates cross-entropy loss for a single batch.
    Args:
        input_batch: Batch of input sequences
        target_batch: Batch of target sequences
        model: The transformer model
        device: Computation device (CPU/GPU)
    Returns:
        torch.Tensor: Scalar loss value
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculates average loss across multiple batches from a data loader.
    Args:
        data_loader: DataLoader instance
        model: The transformer model
        device: Computation device (CPU/GPU)
        num_batches: Optional limit on number of batches to evaluate
    Returns:
        float: Average loss across batches
    """
    if len(data_loader) == 0:
        return float("nan")
    limit = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total = 0.0
    for i, (xb, yb) in enumerate(data_loader):
        if i >= limit:
            break
        total += calc_loss_batch(xb, yb, model, device).item()
    return total / limit

# -------------------------------------------------
# 5.2 evaluation + sample printer
# -------------------------------------------------
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context, context_length):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        out_ids = generate_text_simple(model, encoded, 50, context_length)
        decoded = token_ids_to_text(out_ids.cpu(), tokenizer)
        print(decoded.replace("\n", " "))
    model.train()

# -------------------------------------------------
# 5.2 main training loop (Listing 5.3)
# -------------------------------------------------
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context,
                       tokenizer, context_length):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                tr, va = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(tr)
                val_losses.append(va)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {tr:.3f}, Val loss {va:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context, context_length)

    return train_losses, val_losses, track_tokens_seen

# -------------------------------------------------
# Runner 
# -------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(123)

    # Model & tokenizer
    model = Toilet(GPT_CONFIG_124M).to(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Dataset
    corpus_path = os.path.join("resources", "the-verdict.txt")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_text, val_text = text_data[:split_idx], text_data[split_idx:]

    # DataLoaders
    train_loader = spawn_dataloader(
        train_text, batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True, drop_last=True, num_workers=0,
        tokenizer=tokenizer,
    )
    val_loader = spawn_dataloader(
        val_text, batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=False, drop_last=False, num_workers=0,
        tokenizer=tokenizer,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

    # Train
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=10, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        context_length=GPT_CONFIG_124M["context_length"],
    )


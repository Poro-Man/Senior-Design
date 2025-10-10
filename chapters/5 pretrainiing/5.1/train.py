import os
import torch
import torch.nn as nn
import tiktoken

from resources.work import Toilet
from resources.data import spawn_dataloader

# Model Configuration
# ------------------
# Using GPT-2 small architecture parameters with reduced context length for efficiency
# Total params: ~124M parameters
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # GPT-2 standard vocabulary size
    "context_length": 256,   # Reduced from 1024 for faster training
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of transformer layers
    "drop_rate": 0.1,       # Dropout probability
    "qkv_bias": False,      # No bias in attention projections
}

# Set random seed for reproducibility and determine compute device
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move to appropriate device
model = Toilet(GPT_CONFIG_124M).to(device).eval()

# Text Processing Utilities
# ------------------------
def text_to_token_ids(text: str, tokenizer):
    """
    Convert input text to token IDs using the specified tokenizer.
    
    Args:
        text (str): Input text to tokenize
        tokenizer: GPT-2 tokenizer instance
    
    Returns:
        torch.Tensor: Token IDs with shape (1, sequence_length)
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer):
    """
    Convert token IDs back to text.
    
    Args:
        token_ids (torch.Tensor): Tensor of token IDs
        tokenizer: GPT-2 tokenizer instance
    
    Returns:
        str: Decoded text
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# Text Generation
# --------------
@torch.no_grad()
def generate_text_simple(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int):
    """
    Generate text using greedy sampling (selecting highest probability token).
    
    Args:
        model (nn.Module): The language model
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
        logits = model(idx_cond)           # Shape: (batch_size, seq_len, vocab_size)
        next_logits = logits[:, -1, :]     # Get predictions for next token
        probs = torch.softmax(next_logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # Greedy selection
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Initialize tokenizer and run generation test
tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Every effort moves you"
start_ids = text_to_token_ids(start_context, tokenizer).to(device)

# Test generation with a small sample
gen_ids = generate_text_simple(
    model=model,
    idx=start_ids,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("Output text:\n", token_ids_to_text(gen_ids.cpu(), tokenizer))

# Probability Distribution Analysis
# -------------------------------
# Demonstration of how the model converts logits to probabilities and token predictions
inputs = torch.tensor([[16883,  3626,  6100],
                      [   40,  1107,   588]], dtype=torch.long, device=device)
targets = torch.tensor([[ 3626, 6100,   345],
                       [  107,  588, 11311]], dtype=torch.long, device=device)

with torch.no_grad():
    logits = model(inputs)                     # Shape: (2, 3, vocab_size)
probas = torch.softmax(logits, dim=-1)         # Convert to probabilities
print("probas.shape:", probas.shape)

token_ids_pred = torch.argmax(probas, dim=-1, keepdim=True)   # Select most likely tokens
print("Token IDs (argmax):\n", token_ids_pred.cpu())

# Compare predicted vs target sequences
print(f"Targets batch 1: {token_ids_to_text(targets[0].unsqueeze(0).cpu(), tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids_pred[0].squeeze(-1).unsqueeze(0).cpu(), tokenizer)}")

# Probability Analysis for Target Tokens
# -----------------------------------
b0, b1 = 0, 1  # Batch indices
target_probas_1 = probas[b0, torch.arange(targets.size(1), device=device), targets[b0]]
target_probas_2 = probas[b1, torch.arange(targets.size(1), device=device), targets[b1]]
print("Text 1 target probas:", target_probas_1.detach().cpu())
print("Text 2 target probas:", target_probas_2.detach().cpu())

# Calculate log probabilities and their statistics
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("log_probas (concatenated):", log_probas.detach().cpu())
avg_log_probas = torch.mean(log_probas)
print("avg_log_probas:", avg_log_probas.detach().cpu())
neg_avg_log_probas = -avg_log_probas
print("neg_avg_log_probas:", neg_avg_log_probas.detach().cpu())

# Cross-Entropy Loss Calculation
# ----------------------------
print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

# Reshape tensors for loss calculation
logits_flat = logits.flatten(0, 1)     # Shape: (batch_size * seq_len, vocab_size)
targets_flat = targets.flatten()        # Shape: (batch_size * seq_len)
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

ce_loss = nn.functional.cross_entropy(logits_flat, targets_flat)
print("Cross-entropy loss:", ce_loss.detach().cpu())

# Data Loading and Preprocessing
# ---------------------------
# Load and split dataset
file_path = os.path.join("resources", "the-verdict.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

# Calculate dataset statistics
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# Split data into train/validation sets
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# DataLoader Configuration
# ----------------------
torch.manual_seed(123)

# Initialize training dataloader
train_loader = spawn_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=0,            # Set to 0 for Windows compatibility
    tokenizer=tokenizer,
)

# Initialize validation dataloader
val_loader = spawn_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,
    drop_last=False,
    num_workers=0,
    tokenizer=tokenizer,
)

# Display dataloader shapes for verification
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# Loss Calculation Utilities
# ------------------------
def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor,
                   model: nn.Module, device: torch.device):
    """
    Calculate loss for a single batch.
    
    Args:
        input_batch (torch.Tensor): Input token IDs
        target_batch (torch.Tensor): Target token IDs
        model (nn.Module): The language model
        device (torch.device): Computation device
    
    Returns:
        torch.Tensor: Cross-entropy loss
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model: nn.Module, device: torch.device, num_batches=None):
    """
    Calculate average loss across all batches in a dataloader.
    
    Args:
        data_loader: PyTorch DataLoader instance
        model (nn.Module): The language model
        device (torch.device): Computation device
        num_batches (int, optional): Number of batches to process
    
    Returns:
        float: Average loss across batches
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (inp, tgt) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(inp, tgt, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# Initial Loss Evaluation
# ---------------------
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("\nTraining loss:", train_loss)
print("Validation loss:", val_loss)

import os
import torch
import torch.nn as nn
import tiktoken

from resources.work import Toilet
from resources.data import spawn_dataloader
from resources.train import train_loss, val_loss, calc_loss_batch, calc_loss_loader

def trainning_arc(model, train_loader, val_loader, optimizer, device, num_epochs, 
                 eval_freq, eval_iter, start_context, tokenizer):
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
                resources.train.train_losss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(resources.train.train_loss)
                val_losses.append(resources.train.val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch+1} (Step {global_step:06d}): "
                    f"Train Loss {resources.train.train_loss:.4f}, "
                    f"Val Loss {resources.train.val_loss:.4f}"
                )

        generate_and_print_text(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_of_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_text(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size 
        )
        decoded = tokenizer.decode(token_ids, tokenizer)
        print(decoded._text.repalce("\n", " "))
    model.train()

# Main execution
torch.manual_seed(123)
GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
model = Toilet(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1
)
num_epochs = 10

train_losses, val_losses, track_tokens_seen = trainning_arc(
    model=model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you.",
    tokenizer= tokenizer)

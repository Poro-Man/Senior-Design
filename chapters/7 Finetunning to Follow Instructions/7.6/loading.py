import argparse
import json
import numpy as np
import os
import requests
import tensorflow as tf
import tiktoken
import torch
from tqdm import tqdm


from past_chap import Toilet


# ---------------------------
# Tokenization helpers
# ---------------------------
def text_to_token_ids(text, tokenizer):
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # (T,)
    return tokenizer.decode(flat.tolist())


# ---------------------------
# GPT-2 download & load 
# ---------------------------
def download_and_load_gpt2(model_size, models_dir):
    """Download GPT-2 checkpoint files and return (settings, params) from TF ckpt."""
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        url = os.path.join(base_url, model_size, filename)
        dest = os.path.join(model_dir, filename)
        download_file(url, dest)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def download_file(url, destination):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    remote_size = int(r.headers.get("Content-Length", 0))

    if os.path.exists(destination):
        local_size = os.path.getsize(destination)
        if remote_size and remote_size == local_size:
            print(f"File already exists and is up-to-date: {destination}")
            return

    block = 1024
    with tqdm(total=remote_size, unit="iB", unit_scale=True, desc=os.path.basename(url)) as bar:
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=block):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Read TF checkpoint into a nested Python dict `params` (book style)."""
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(ckpt_path):
        arr = np.squeeze(tf.train.load_variable(ckpt_path, name))
        parts = name.split("/")[1:]  # drop "model/"
        target = params
        if parts[0].startswith("h"):
            layer_idx = int(parts[0][1:])
            target = params["blocks"][layer_idx]
        for key in parts[1:-1]:
            target = target.setdefault(key, {})
        target[parts[-1]] = arr
    return params


# ---------------------------
# Weight assignment utilities
# ---------------------------
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """Map OpenAI GPT-2 weights from `params` into YOUR model (`Toilet`)."""

    # Positional and token embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])   # A
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])   # A

    # Transformer blocks
    for b in range(len(params["blocks"])):                            # B
        # ---- Attention Q,K,V (weights) ----
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)  # C
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight   = assign(gpt.trf_blocks[b].att.W_key.weight,   k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # ---- Attention Q,K,V (biases) ----
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias   = assign(gpt.trf_blocks[b].att.W_key.bias,   k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # ---- Attention output projection ----
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        # ---- MLP (feed-forward) ----
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias   = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,   params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias   = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,   params["blocks"][b]["mlp"]["c_proj"]["b"])

        # ---- LayerNorms (pre-norm) ----
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    # Final norm + weight tying (OpenAI tied output to token embeddings)      # D
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight  = assign(gpt.out_head.weight,  params["wte"])


# ---------------------------
# Generation (book 5.4/5.5)
# ---------------------------
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float("-inf")).to(logits.device),
                                 logits)

        if temperature > 0.0:
            logits = logits / temperature
            # numerical stability (optional)
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ---------------------------
# Main
# ---------------------------
def main(gpt_config, input_prompt, model_size, device):
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # Build YOUR model class (Toilet) with the config
    gpt = Toilet(gpt_config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device).eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a pretrained GPT-2 model.")
    parser.add_argument("--prompt", default="Every effort moves you",
                        help="Prompt text used to seed the generation.")
    parser.add_argument("--device", default="cpu",
                        help="Device for running inference, e.g., cpu, cuda, mps.")
    args = parser.parse_args()

    torch.manual_seed(123)

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = args.prompt
    DEVICE = torch.device(args.device)

    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)":   {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size, DEVICE)

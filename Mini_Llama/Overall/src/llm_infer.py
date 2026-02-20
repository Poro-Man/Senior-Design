# src/llm_infer.py
import os
import math
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

import tiktoken

from src.model import ModelArgs, LlamaForCausalLM


def resolve_device(device: str) -> torch.device:
    device = (device or "").lower().strip()
    if device in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device in ("cpu", ""):
        return torch.device("cpu")
    # fallback
    return torch.device(device)


def _extract_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Supports multiple checkpoint formats:
      A) raw state_dict: {"tok_embeddings.weight": ..., ...}
      B) { "model": state_dict, "model_args": {...}, ... }
      C) { "model_state_dict": state_dict, ... }
      D) { "model_state": state_dict, ... } (older variants)
    Returns: (state_dict, model_args_dict_or_empty)
    """
    model_args = {}

    # A) raw state dict
    if isinstance(ckpt, dict) and "tok_embeddings.weight" in ckpt:
        return ckpt, model_args

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict or a raw state_dict.")

    # pull model args if present
    if "model_args" in ckpt and isinstance(ckpt["model_args"], dict):
        model_args = ckpt["model_args"]

    # B/C/D) wrapped state dict
    for key in ("model", "model_state_dict", "model_state"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key], model_args

    # sometimes trainers store it under "state_dict"
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"], model_args

    raise KeyError("Could not find model weights in checkpoint (tried: model, model_state_dict, model_state, state_dict).")


class LLMInfer:
    def __init__(self, ckpt_path: str, device: str = "cpu", tokenizer_name: str = "cl100k_base"):
        self.device = resolve_device(device)

        # tokenizer
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.enc.n_vocab

        # load ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state, model_args = _extract_state_dict(ckpt)

        # build model args
        # prefer ckpt model_args if available; else fall back to reasonable defaults
        margs = ModelArgs(
            dim=int(model_args.get("dim", 1024)),
            n_layers=int(model_args.get("n_layers", 12)),
            n_heads=int(model_args.get("n_heads", 8)),
            n_kv_heads=int(model_args.get("n_kv_heads", 4)),
            vocab_size=int(model_args.get("vocab_size", self.vocab_size)),
            max_seq_len=int(model_args.get("max_seq_len", 512)),
            dropout=float(model_args.get("dropout", 0.0)),
        )

        self.model = LlamaForCausalLM(margs).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # keep these for convenience
        self.model_args = margs
        self.ckpt_path = ckpt_path

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        # encode prompt -> tensor
        ids = self.enc.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            out = self.model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out  # (B,T,V)
            next_logits = logits[:, -1, :]  # (B,V)

            if temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)

                # top-p nucleus sampling
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    keep = cum <= top_p
                    keep[..., 0] = True  # always keep at least 1 token

                    filtered = torch.zeros_like(probs)
                    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
                    probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                next_id = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_id], dim=1)

            # keep context window
            if x.size(1) > self.model_args.max_seq_len:
                x = x[:, -self.model_args.max_seq_len :]

        out_text = self.enc.decode(x[0].tolist())
        return out_text

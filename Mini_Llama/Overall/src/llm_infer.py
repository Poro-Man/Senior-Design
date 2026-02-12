# llm_infer.py
import __main__
from typing import Any, Dict

import torch
import tiktoken

from .model import ModelArgs, LlamaForCausalLM


def _extract_model_kwargs(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to reconstruct model hyperparams from common checkpoint fields.
    Falls back to the known local training settings if not found.
    """
    for key in ("config", "args", "model_args"):
        if key in ckpt:
            cfg = ckpt[key]
            if hasattr(cfg, "__dict__"):
                cfg = vars(cfg)
            if isinstance(cfg, dict):
                out: Dict[str, Any] = {}
                for k in ("dim", "n_layers", "n_heads", "n_kv_heads", "max_seq_len", "multiple_of", "norm_eps"):
                    if k in cfg and cfg[k] is not None:
                        out[k] = cfg[k]
                # Make sure integer-y fields are ints
                for k in ("dim", "n_layers", "n_heads", "n_kv_heads", "max_seq_len", "multiple_of"):
                    if k in out:
                        out[k] = int(out[k])
                return out

    # Fallback: what you trained with in your quick command
    return {
        "dim": 1024,
        "n_layers": 12,
        "n_heads": 8,
        "n_kv_heads": 4,
        "max_seq_len": 512,
    }


class LLMInfer:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path

        # Must match training tokenizer
        self.enc = tiktoken.get_encoding("cl100k_base")
        vocab_size = self.enc.n_vocab

        # Fix: checkpoint may reference __main__.ModelArgs (saved from training script)
        setattr(__main__, "ModelArgs", ModelArgs)

        # Load ckpt first so we can build the model with matching sizes
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_kwargs = _extract_model_kwargs(ckpt)

        args = ModelArgs(vocab_size=vocab_size, **model_kwargs)
        self.model = LlamaForCausalLM(args).to(self.device)
        self.model.eval()

        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=True)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        # Encode prompt
        prompt_ids = self.enc.encode(prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)[None, :]

        # Autoregressive decode
        for _ in range(int(max_new_tokens)):
            logits, _ = self.model(x, targets=None)  # (1, T, V)
            logits = logits[:, -1, :]  # (1, V)

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
            else:
                probs = torch.softmax(logits / float(temperature), dim=-1)

                # Top-p (nucleus) sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)

                cutoff = cum > float(top_p)
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                sampled = torch.multinomial(sorted_probs, num_samples=1)  # (1, 1)
                next_id = sorted_idx.gather(-1, sampled)  # (1, 1)

            x = torch.cat([x, next_id], dim=1)

        # IMPORTANT: return only newly generated tokens (not the prompt)
        new_tokens = x[0].tolist()[len(prompt_ids):]
        return self.enc.decode(new_tokens)

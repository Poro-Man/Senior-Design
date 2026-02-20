#!/usr/bin/env python3
"""
benchmark.py  —  Comprehensive benchmarking suite for Mini LLaMA.

Measures inference throughput, latency, memory, perplexity,
SDPA-vs-fallback speedup, and generates sample outputs.

Usage:
    python -m scripts.benchmark --ckpt ./checkpoints/pretrain_owt_1b/checkpoint_step_61036.pt --device cuda
    python -m scripts.benchmark --ckpt <path> --device cpu --num_runs 2 --max_new_tokens 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import statistics
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# ── project imports ──────────────────────────────────────────────────────
# Ensures the project root is on sys.path for `from src.*` imports
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model import ModelArgs, LlamaForCausalLM
from src.llm_infer import LLMInfer, resolve_device


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

_DIVIDER = "=" * 72
_SECTION = "-" * 72

DEFAULT_PROMPT = (
    "In the beginning of the twenty-first century, advances in artificial "
    "intelligence began to reshape every aspect of human life. Researchers "
    "discovered that large language models could"
)

# A longer passage used for perplexity evaluation.  ~500 tokens of
# public-domain text keeps the benchmark self-contained (no data deps).
_PERPLEXITY_TEXT = """
Language modeling is the task of predicting the next word in a sequence given
the words that precede it. It is one of the oldest and most fundamental
problems in natural language processing. The quality of a language model is
typically measured by perplexity, which is the exponentiated average
negative log-likelihood of the test set. A lower perplexity indicates that
the model assigns higher probability to the held-out text, and therefore
captures the statistical regularities of the language more effectively.

Modern language models are based on the Transformer architecture, which was
introduced in 2017. The Transformer replaces recurrence with self-attention,
allowing every position in the sequence to attend to every other position
directly. This design enables efficient parallelization during training and
has proven to scale remarkably well. When trained on large corpora of text,
Transformer-based models learn rich representations of language that transfer
across a wide variety of downstream tasks.

The decoder-only variant of the Transformer, popularized by the GPT family
of models, generates text autoregressively: at each step, the model outputs
a probability distribution over the vocabulary conditioned on all preceding
tokens. During inference, tokens are sampled from this distribution one at
a time, and the generated token is appended to the context for the next step.
Techniques such as top-p (nucleus) sampling and temperature scaling allow
users to control the trade-off between diversity and coherence in the
generated text.
""".strip()


def _fmt_time(seconds: float) -> str:
    """Human-readable time string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def _fmt_mem(nbytes: int) -> str:
    """Human-readable memory string."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.2f} MB"
    return f"{nbytes / (1 << 10):.2f} KB"


def _print_header(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def _print_kv(key: str, value: Any, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key:<36s} {value}")


# ═════════════════════════════════════════════════════════════════════════
# 1. Model Info
# ═════════════════════════════════════════════════════════════════════════

def bench_model_info(llm: LLMInfer) -> Dict[str, Any]:
    """Report parameter counts, architecture config, and device info."""
    _print_header("MODEL INFO")

    model = llm.model
    params = model.get_num_params(non_embedding=False)
    params_no_emb = model.get_num_params(non_embedding=True)
    emb_params = params - params_no_emb

    _print_kv("Checkpoint", llm.ckpt_path)
    _print_kv("Device", str(llm.device))
    _print_kv("PyTorch version", torch.__version__)
    _print_kv("CUDA available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        _print_kv("GPU", torch.cuda.get_device_name(llm.device))
        _print_kv("CUDA version", torch.version.cuda or "N/A")
    _print_kv("SDPA available", str(hasattr(F, "scaled_dot_product_attention")))

    print(f"\n  {'Parameter Counts':<36s}")
    print(f"  {_SECTION[:36]}")
    _print_kv("Total parameters", f"{params:,}")
    _print_kv("Non-embedding parameters", f"{params_no_emb:,}")
    _print_kv("Embedding parameters", f"{emb_params:,}")

    args_dict = asdict(llm.model_args)
    print(f"\n  {'Architecture Config':<36s}")
    print(f"  {_SECTION[:36]}")
    for k, v in args_dict.items():
        _print_kv(k, str(v))

    return {
        "total_params": params,
        "non_embedding_params": params_no_emb,
        "embedding_params": emb_params,
        "device": str(llm.device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sdpa_available": hasattr(F, "scaled_dot_product_attention"),
        "model_args": args_dict,
    }


# ═════════════════════════════════════════════════════════════════════════
# 2. Throughput
# ═════════════════════════════════════════════════════════════════════════

def bench_throughput(
    llm: LLMInfer,
    prompt: str,
    max_new_tokens: int,
    num_runs: int,
) -> Dict[str, Any]:
    """Measure tokens-per-second over several runs."""
    _print_header("INFERENCE THROUGHPUT")

    # Warmup run (excluded from timing)
    _ = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)

    times: List[float] = []
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize(llm.device)
        t0 = time.perf_counter()
        _ = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)
        if torch.cuda.is_available():
            torch.cuda.synchronize(llm.device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    toks_per_sec = [max_new_tokens / t for t in times]

    mean_tps = statistics.mean(toks_per_sec)
    std_tps = statistics.stdev(toks_per_sec) if len(toks_per_sec) > 1 else 0.0
    min_tps = min(toks_per_sec)
    max_tps = max(toks_per_sec)
    mean_time = statistics.mean(times)

    _print_kv("Generated tokens per run", max_new_tokens)
    _print_kv("Number of runs", num_runs)
    _print_kv("Mean generation time", _fmt_time(mean_time))
    _print_kv("Mean tokens/sec", f"{mean_tps:.2f}")
    _print_kv("Std  tokens/sec", f"{std_tps:.2f}")
    _print_kv("Min  tokens/sec", f"{min_tps:.2f}")
    _print_kv("Max  tokens/sec", f"{max_tps:.2f}")

    return {
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs,
        "times_sec": times,
        "tokens_per_sec_mean": mean_tps,
        "tokens_per_sec_std": std_tps,
        "tokens_per_sec_min": min_tps,
        "tokens_per_sec_max": max_tps,
        "mean_generation_time_sec": mean_time,
    }


# ═════════════════════════════════════════════════════════════════════════
# 3. Latency (time-to-first-token)
# ═════════════════════════════════════════════════════════════════════════

def bench_latency(
    llm: LLMInfer,
    prompt: str,
    max_new_tokens: int,
    num_runs: int,
) -> Dict[str, Any]:
    """Measure time-to-first-token vs total generation time."""
    _print_header("LATENCY (TIME-TO-FIRST-TOKEN)")

    ttft_times: List[float] = []
    total_times: List[float] = []

    for _ in range(num_runs):
        # --- Time to first token (generate just 1 token) ---
        if torch.cuda.is_available():
            torch.cuda.synchronize(llm.device)
        t0 = time.perf_counter()
        _ = llm.generate(prompt, max_new_tokens=1, temperature=0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize(llm.device)
        t1 = time.perf_counter()
        ttft_times.append(t1 - t0)

        # --- Total generation time ---
        if torch.cuda.is_available():
            torch.cuda.synchronize(llm.device)
        t0 = time.perf_counter()
        _ = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)
        if torch.cuda.is_available():
            torch.cuda.synchronize(llm.device)
        t1 = time.perf_counter()
        total_times.append(t1 - t0)

    mean_ttft = statistics.mean(ttft_times)
    mean_total = statistics.mean(total_times)
    mean_decode = mean_total - mean_ttft  # approximate decode time (tokens 2..N)

    _print_kv("Mean time-to-first-token", _fmt_time(mean_ttft))
    _print_kv("Mean total generation time", _fmt_time(mean_total))
    _print_kv("Mean decode time (tokens 2..N)", _fmt_time(mean_decode))
    _print_kv(f"Decode tokens/sec (approx)", f"{(max_new_tokens - 1) / mean_decode:.2f}" if mean_decode > 0 else "N/A")

    return {
        "ttft_times_sec": ttft_times,
        "total_times_sec": total_times,
        "mean_ttft_sec": mean_ttft,
        "mean_total_sec": mean_total,
        "mean_decode_sec": mean_decode,
    }


# ═════════════════════════════════════════════════════════════════════════
# 4. Sequence Length Scaling
# ═════════════════════════════════════════════════════════════════════════

def bench_sequence_scaling(
    llm: LLMInfer,
    prompt: str,
    max_new_tokens: int,
    num_runs: int,
) -> Dict[str, Any]:
    """Test throughput at different input prompt lengths."""
    _print_header("SEQUENCE LENGTH SCALING")

    enc = llm.enc
    # Encode the prompt and tile it to create different lengths
    base_ids = enc.encode(prompt)
    max_seq = llm.model_args.max_seq_len

    # Target prompt lengths (in tokens). We stay under max_seq_len - max_new_tokens
    target_lengths = [32, 128, 256]
    # Only include if the model's max_seq_len supports it
    target_lengths = [l for l in target_lengths if l + max_new_tokens <= max_seq]
    if not target_lengths:
        target_lengths = [min(32, max(1, max_seq - max_new_tokens))]

    results_by_length: Dict[int, Dict[str, float]] = {}

    print(f"  {'Prompt Len':<14s} {'Tok/s (mean)':<16s} {'Time (mean)':<16s}")
    print(f"  {_SECTION[:46]}")

    for tgt_len in target_lengths:
        # Build a prompt of the target token length by repeating the base tokens
        repeated_ids = (base_ids * ((tgt_len // len(base_ids)) + 1))[:tgt_len]
        test_prompt = enc.decode(repeated_ids)

        times: List[float] = []
        gen_tokens = min(max_new_tokens, max_seq - tgt_len)
        if gen_tokens <= 0:
            gen_tokens = 1

        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize(llm.device)
            t0 = time.perf_counter()
            _ = llm.generate(test_prompt, max_new_tokens=gen_tokens, temperature=0.8)
            if torch.cuda.is_available():
                torch.cuda.synchronize(llm.device)
            times.append(time.perf_counter() - t0)

        mean_t = statistics.mean(times)
        tps = gen_tokens / mean_t

        print(f"  {tgt_len:<14d} {tps:<16.2f} {_fmt_time(mean_t)}")
        results_by_length[tgt_len] = {
            "gen_tokens": gen_tokens,
            "mean_time_sec": mean_t,
            "tokens_per_sec": tps,
        }

    return {"scaling": results_by_length}


# ═════════════════════════════════════════════════════════════════════════
# 5. Memory Usage
# ═════════════════════════════════════════════════════════════════════════

def bench_memory(
    llm: LLMInfer,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Report GPU/CPU memory usage."""
    _print_header("MEMORY USAGE")

    use_cuda = llm.device.type == "cuda"

    # Pre-inference memory (model loaded, no generation yet)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(llm.device)
        pre_alloc = torch.cuda.memory_allocated(llm.device)

    # Run a generation to measure peak
    _ = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)

    result: Dict[str, Any] = {}

    if use_cuda:
        torch.cuda.synchronize(llm.device)
        peak = torch.cuda.max_memory_allocated(llm.device)
        current = torch.cuda.memory_allocated(llm.device)
        reserved = torch.cuda.memory_reserved(llm.device)

        _print_kv("Model memory (pre-inference)", _fmt_mem(pre_alloc))
        _print_kv("Peak memory (during inference)", _fmt_mem(peak))
        _print_kv("Current memory (post-gen)", _fmt_mem(current))
        _print_kv("Reserved by allocator", _fmt_mem(reserved))
        _print_kv("Inference overhead (peak - pre)", _fmt_mem(peak - pre_alloc))

        result = {
            "device": "cuda",
            "pre_inference_bytes": pre_alloc,
            "peak_bytes": peak,
            "current_bytes": current,
            "reserved_bytes": reserved,
            "inference_overhead_bytes": peak - pre_alloc,
        }
    else:
        # CPU fallback — report process RSS via psutil if available
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss
            _print_kv("Process RSS (approx)", _fmt_mem(rss))
            result = {"device": "cpu", "process_rss_bytes": rss}
        except ImportError:
            # Rough estimate from model parameters
            param_bytes = sum(p.nelement() * p.element_size() for p in llm.model.parameters())
            _print_kv("Estimated model weight memory", _fmt_mem(param_bytes))
            _print_kv("(install psutil for accurate CPU memory)", "")
            result = {"device": "cpu", "estimated_param_bytes": param_bytes}

    return result


# ═════════════════════════════════════════════════════════════════════════
# 6. Perplexity
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def bench_perplexity(llm: LLMInfer) -> Dict[str, Any]:
    """Compute perplexity on a built-in sample text."""
    _print_header("PERPLEXITY (LANGUAGE MODEL QUALITY)")

    enc = llm.enc
    model = llm.model
    device = llm.device
    max_seq = llm.model_args.max_seq_len

    ids = enc.encode(_PERPLEXITY_TEXT)
    total_tokens = len(ids)

    if total_tokens < 2:
        print("  WARNING: perplexity text too short to evaluate.")
        return {"perplexity": float("nan"), "total_tokens": total_tokens}

    # Evaluate in chunks of max_seq_len (non-overlapping)
    total_loss = 0.0
    total_count = 0

    for start in range(0, total_tokens - 1, max_seq):
        end = min(start + max_seq + 1, total_tokens)  # +1 for targets
        chunk = ids[start:end]
        if len(chunk) < 2:
            break

        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_count += y.numel()

    avg_loss = total_loss / total_count
    ppl = math.exp(avg_loss)

    _print_kv("Evaluation tokens", total_count)
    _print_kv("Average cross-entropy loss", f"{avg_loss:.4f}")
    _print_kv("Perplexity", f"{ppl:.2f}")

    return {
        "perplexity": ppl,
        "avg_loss": avg_loss,
        "eval_tokens": total_count,
    }


# ═════════════════════════════════════════════════════════════════════════
# 7. SDPA vs Fallback Comparison
# ═════════════════════════════════════════════════════════════════════════

def bench_sdpa_vs_fallback(
    llm: LLMInfer,
    prompt: str,
    max_new_tokens: int,
    num_runs: int,
) -> Dict[str, Any]:
    """Compare throughput with SDPA enabled vs disabled."""
    _print_header("SDPA vs FALLBACK ATTENTION")

    if not hasattr(F, "scaled_dot_product_attention"):
        print("  SDPA not available in this PyTorch build — skipping comparison.")
        return {"skipped": True, "reason": "SDPA not available"}

    # ---- SDPA enabled (normal path) ----
    warmup = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)
    sdpa_times: List[float] = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize(llm.device)
        t0 = time.perf_counter()
        _ = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)
        if torch.cuda.is_available():
            torch.cuda.synchronize(llm.device)
        sdpa_times.append(time.perf_counter() - t0)

    # ---- Disable SDPA by temporarily removing the function ----
    original_sdpa = F.scaled_dot_product_attention
    try:
        delattr(F, "scaled_dot_product_attention")

        # The model's forward checks hasattr(F, "scaled_dot_product_attention")
        # each call, so deleting it forces the fallback path.
        warmup = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)
        fallback_times: List[float] = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize(llm.device)
            t0 = time.perf_counter()
            _ = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)
            if torch.cuda.is_available():
                torch.cuda.synchronize(llm.device)
            fallback_times.append(time.perf_counter() - t0)
    finally:
        # Restore SDPA
        F.scaled_dot_product_attention = original_sdpa

    sdpa_tps = max_new_tokens / statistics.mean(sdpa_times)
    fallback_tps = max_new_tokens / statistics.mean(fallback_times)
    speedup = sdpa_tps / fallback_tps if fallback_tps > 0 else float("inf")

    _print_kv("SDPA tokens/sec", f"{sdpa_tps:.2f}")
    _print_kv("Fallback tokens/sec", f"{fallback_tps:.2f}")
    _print_kv("SDPA speedup", f"{speedup:.2f}x")

    return {
        "sdpa_tokens_per_sec": sdpa_tps,
        "fallback_tokens_per_sec": fallback_tps,
        "speedup": speedup,
        "sdpa_times_sec": sdpa_times,
        "fallback_times_sec": fallback_times,
    }


# ═════════════════════════════════════════════════════════════════════════
# 8. Sample Outputs
# ═════════════════════════════════════════════════════════════════════════

def bench_sample_outputs(
    llm: LLMInfer,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Generate text at several temperature/top_p settings."""
    _print_header("SAMPLE OUTPUTS")

    configs = [
        {"temperature": 0.2, "top_p": 0.90, "label": "Low temp (focused)"},
        {"temperature": 0.8, "top_p": 0.95, "label": "Medium temp (balanced)"},
        {"temperature": 1.2, "top_p": 0.98, "label": "High temp (creative)"},
    ]

    samples: List[Dict[str, Any]] = []
    for cfg in configs:
        print(f"\n  ── {cfg['label']} (temp={cfg['temperature']}, top_p={cfg['top_p']}) ──")
        text = llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
        # Print just the generated continuation (after the prompt)
        generated = text[len(prompt):] if text.startswith(prompt) else text
        # Wrap long lines for readability
        for line in generated.split("\n"):
            print(f"  {line}")
        samples.append({
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "generated_text": generated.strip(),
        })

    return {"samples": samples}


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark suite for Mini LLaMA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", type=str, required=True,
                     help="Path to model checkpoint")
    ap.add_argument("--device", type=str, default="cuda",
                     help="Device to run on (cuda, cpu, mps)")
    ap.add_argument("--max_new_tokens", type=int, default=128,
                     help="Tokens to generate per benchmark run")
    ap.add_argument("--num_runs", type=int, default=5,
                     help="Number of timed runs (higher = less noise)")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                     help="Prompt text for generation benchmarks")
    ap.add_argument("--output", type=str, default=None,
                     help="Optional path to save results as JSON")
    args = ap.parse_args()

    print(_DIVIDER)
    print("  MINI LLAMA BENCHMARK SUITE")
    print(_DIVIDER)
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Device     : {args.device}")
    print(f"  Tokens/run : {args.max_new_tokens}")
    print(f"  Num runs   : {args.num_runs}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n  Loading model...")
    t_load_start = time.perf_counter()
    llm = LLMInfer(args.ckpt, device=args.device)
    t_load = time.perf_counter() - t_load_start
    print(f"  Model loaded in {_fmt_time(t_load)}")

    all_results: Dict[str, Any] = {"load_time_sec": t_load}

    # ── Run benchmarks ───────────────────────────────────────────────────
    all_results["model_info"] = bench_model_info(llm)
    all_results["throughput"] = bench_throughput(llm, args.prompt, args.max_new_tokens, args.num_runs)
    all_results["latency"] = bench_latency(llm, args.prompt, args.max_new_tokens, args.num_runs)
    all_results["sequence_scaling"] = bench_sequence_scaling(llm, args.prompt, args.max_new_tokens, args.num_runs)
    all_results["memory"] = bench_memory(llm, args.prompt, args.max_new_tokens)
    all_results["perplexity"] = bench_perplexity(llm)
    all_results["sdpa_vs_fallback"] = bench_sdpa_vs_fallback(llm, args.prompt, args.max_new_tokens, args.num_runs)
    all_results["sample_outputs"] = bench_sample_outputs(llm, args.prompt, args.max_new_tokens)

    # ── Summary ──────────────────────────────────────────────────────────
    _print_header("BENCHMARK SUMMARY")
    _print_kv("Total params", f"{all_results['model_info']['total_params']:,}")
    _print_kv("Throughput (mean)", f"{all_results['throughput']['tokens_per_sec_mean']:.2f} tok/s")
    _print_kv("Time-to-first-token (mean)", _fmt_time(all_results['latency']['mean_ttft_sec']))
    _print_kv("Perplexity", f"{all_results['perplexity']['perplexity']:.2f}")

    sdpa_res = all_results["sdpa_vs_fallback"]
    if not sdpa_res.get("skipped"):
        _print_kv("SDPA speedup", f"{sdpa_res['speedup']:.2f}x")

    if "peak_bytes" in all_results["memory"]:
        _print_kv("Peak GPU memory", _fmt_mem(all_results['memory']['peak_bytes']))

    # ── Save JSON ────────────────────────────────────────────────────────
    if args.output:
        # Convert any non-serializable values
        def _clean(obj):
            if isinstance(obj, float):
                if math.isinf(obj) or math.isnan(obj):
                    return str(obj)
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj

        with open(args.output, "w") as f:
            json.dump(_clean(all_results), f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    print(f"\n{_DIVIDER}")
    print("  BENCHMARKING COMPLETE")
    print(_DIVIDER)


if __name__ == "__main__":
    main()

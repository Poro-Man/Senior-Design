#!/usr/bin/env python3
"""Quick evaluation of model output quality across diverse prompts."""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.llm_infer import LLMInfer

CKPT = "./checkpoints/sft_ultrachat_200m_v2/checkpoint_step_15000.pt"

def main():
    llm = LLMInfer(CKPT, device="cuda")
    prompts = [
        ("Chat: Capital of France",
         "User: What is the capital of France?\nAssistant:"),
        ("Chat: Neural network explanation",
         "User: Explain what a neural network is in simple terms.\nAssistant:"),
        ("Chat: Python coding",
         "User: Write a Python function to reverse a string.\nAssistant:"),
        ("Plain completion",
         "The quick brown fox"),
        ("Factual completion",
         "Water boils at"),
        ("Chat: List instruction",
         "User: List three benefits of exercise.\nAssistant:"),
    ]
    results = []
    for label, p in prompts:
        text = llm.generate(p, max_new_tokens=100, temperature=0.5, top_p=0.9)
        generated = text[len(p):] if text.startswith(p) else text
        results.append({"label": label, "prompt": p, "output": generated.strip()})
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done")

if __name__ == "__main__":
    main()

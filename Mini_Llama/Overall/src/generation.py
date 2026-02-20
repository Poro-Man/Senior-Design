# src/generation.py
import argparse
from src.llm_infer import LLMInfer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    llm = LLMInfer(args.ckpt, device=args.device)

    text = llm.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()

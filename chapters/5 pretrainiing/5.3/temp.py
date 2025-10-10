import os
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt


from resources.work import Toilet


vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}


next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])


def demo_argmax_vs_multinomial():
    probas = torch.softmax(next_token_logits, dim=0)

    next_token_id = torch.argmax(probas).item()
    print("Greedy (argmax):", inverse_vocab[next_token_id])

    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print("Sampled once (multinomial):", inverse_vocab[next_token_id])

    def print_sampled_tokens(p):
        torch.manual_seed(123)
        sample = [torch.multinomial(p, num_samples=1).item() for _ in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(p))
        for i, freq in enumerate(sampled_ids.tolist()):
            print(f"{freq} x {inverse_vocab[i]}")

    print("\nSampling histogram (1,000 draws):")
    print_sampled_tokens(probas)



def demo_topk_k_equals_3():
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("\nTop logits:", top_logits)
    print("Top positions:", top_pos)

    # where: set logits below the minimum top-k value to -inf
    new_logits = torch.where(
        next_token_logits < top_logits[-1],           # A
        torch.tensor(float("-inf")),                  # B
        next_token_logits                             # C
    )
    print("\nTop-k filtered logits:", new_logits)
    print("Softmax after top-k:", torch.softmax(new_logits, dim=0))


# -------------------------------------------------
# 3b) Temperature scaling visualization (book §5.4)
# -------------------------------------------------
import matplotlib.pyplot as plt

def softmax_with_temperature(logits, temperature):
    """Apply temperature scaling to logits and return probabilities."""
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

def demo_temperature_scaling():
    """
    Illustrate how temperature affects the softness of the probability distribution.
    T > 1  → more uniform (less confident)
    T < 1  → sharper (more confident)
    """
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(6, 3))

    for i, T in enumerate(temperatures):
        rects = ax.bar(
            x + i * bar_width,
            scaled_probas[i],
            bar_width,
            label=f'Temperature = {T}'
        )

    ax.set_ylabel('Probability')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size,  
             temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):                                  
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]

        if top_k is not None:                                        
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        if temperature > 0.0:                                        
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:                                                         
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None:                                       
            if torch.all(idx_next.squeeze(-1) == eos_id):
                idx = torch.cat((idx, idx_next), dim=1)
                break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx



def text_to_token_ids(text: str, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())



GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}




if __name__ == "__main__":

    print("=== Argmax vs Multinomial (Toy) ===")
    demo_argmax_vs_multinomial()

    print("\n=== Top-k Filtering (k=3) ===")
    demo_topk_k_equals_3()

    print("\n=== Temperature Scaling Demo ===")
    demo_temperature_scaling()



    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Toilet(GPT_CONFIG_124M).to(device).eval()
    tokenizer = tiktoken.get_encoding("gpt2")

    start = "Every effort moves you"
    start_ids = text_to_token_ids(start, tokenizer).to(device)

    token_ids = generate(
        model=model,
        idx=start_ids,
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4,
        eos_id=None,
    )

    print("\n=== Output text (Model) ===")
    print(token_ids_to_text(token_ids.cpu(), tokenizer))



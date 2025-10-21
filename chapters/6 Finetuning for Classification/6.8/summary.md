# 🧩 Chapter 6.8 — Evaluating and Using the Finetuned Model

## Overview
Chapter **6.8** of *Build a Large Language Model (From Scratch)* concludes the finetuning process by demonstrating how to **save, reload, and use** the trained GPT-2 spam classifier for inference on new text inputs.  

After training in Chapter 6.7, the model is transformed from a generative language model into a practical classifier — capable of distinguishing between spam and non-spam messages using its final two-class output head.

---

## 1️⃣ Saving the Finetuned Model

Once training completes, the model’s learned weights are saved for later reuse.  
This is essential so the model doesn’t need to be retrained every time you want to classify messages.

```
save_path = "review_classifier.pth"
torch.save(model.state_dict(), save_path)
print(f"[OK] Finetuned model saved to: {save_path}")
```

This creates a file called **`review_classifier.pth`**, containing all trainable parameters — including the finetuned last transformer block, final normalization layer, and new classification head.

---

## 2️⃣ Rebuilding the Architecture for Inference

To load the model later, the same GPT-2 architecture from earlier chapters must be reconstructed.  
This ensures the saved weight shapes match the model layers.

```
from resources.past_chap import Toilet as GPTModel
import torch.nn as nn

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model = GPTModel(BASE_CONFIG)
model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=2)
```

This sets up the same **GPT-2-small** model structure used for finetuning.

---

## 3️⃣ Loading the Finetuned Weights

The saved checkpoint is then reloaded into the model, restoring the classifier to its finetuned state.

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("review_classifier.pth", map_location=device))
model.to(device).eval()
print("[OK] Loaded finetuned classifier from: review_classifier.pth")
```

`model.eval()` ensures dropout and other training-only behaviors are disabled, allowing deterministic inference.

---

## 4️⃣ Classifying New Messages

The function `classify_review()` runs the model on new text samples.  
It handles **tokenization**, **truncation**, and **padding**, then uses the **last token’s logits** to decide between “spam” and “not spam.”

```
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_ctx = model.pos_emb.weight.shape[1]

    clip_len = supported_ctx if max_length is None else min(max_length, supported_ctx)
    input_ids = input_ids[:clip_len]
    input_ids += [pad_token_id] * (clip_len - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        pred = torch.argmax(logits, dim=-1).item()

    return "spam" if pred == 1 else "not spam"
```

---

## 5️⃣ Testing the Classifier

Example messages (as in the book):

```
text_1 = (
    "You are a winner you have been specially "
    "selected to receive $1000 cash or a $2000 award."
)
text_2 = (
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
)

print("Example 1:", classify_review(text_1, model, tokenizer, device, max_length=120))
print("Example 2:", classify_review(text_2, model, tokenizer, device, max_length=120))
```

**Expected Output**
```
[OK] Loaded finetuned classifier from: review_classifier.pth
Example 1: spam
Example 2: not spam
```

If both return “spam,” it typically means the wrong `max_length` was used — during training the model likely saw sequences around **120 tokens**, so you should pass that same `max_length` for proper behavior.

---

## 🧠 Key Takeaways

| Step | Description |
|------|--------------|
| **Save model** | Use `torch.save(model.state_dict())` after finetuning to preserve learned weights. |
| **Reload** | Rebuild GPT-2’s architecture, then load the `.pth` file into the model. |
| **Inference** | Use the last-token logits to classify inputs into two categories. |
| **Consistency** | The same tokenizer, context length, and padding must be used for reliable results. |

---

## 🧩 Chapter Context

Chapter 6.8 completes the **classification pipeline** that began in Chapter 6.1.  
After pretraining, finetuning, and evaluation, the model is now ready for deployment — it can take in arbitrary text and output accurate spam/not-spam predictions.

![Figure 6.18 — End-to-End Spam Classifier Pipeline](images/Screenshot%202025-10-21%20104213.png)

---

**Summary:**  
> Chapter 6.8 finalizes the GPT-2 spam classification project by teaching how to save and reload the trained model for real-world inference.  
> The `classify_review()` function encapsulates preprocessing, model prediction, and label decoding — turning the once-generative GPT-2 into a practical, deployable binary text classifier.

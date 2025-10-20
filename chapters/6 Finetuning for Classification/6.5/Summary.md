# 🧩 Chapter 6.5 — Replacing the Language Model Head with a Classification Head

## Overview
Chapter **6.5** of *Build a Large Language Model (From Scratch)* marks the beginning of **fine-tuning GPT-2 for classification**.  
In this section, the model’s original **language modeling head** (which predicts next tokens) is replaced with a **two-class classifier head** for the spam detection task.  

This step transforms GPT-2 from a generative model into a discriminative model capable of labeling text messages as either **“ham”** or **“spam.”**

---

## 1️⃣ Loading the Pretrained GPT-2 Model

The pretrained GPT-2 model (downloaded and loaded in Chapter 6.4) is initialized with the same configuration dictionary and its weights.  
This ensures the classifier starts with rich linguistic knowledge from pretraining.

```
from resources.loading import download_and_load_gpt2, load_weights_into_gpt
from resources.past_chap import Toilet as GPTModel

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size_token = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
_, params = download_and_load_gpt2(model_size=model_size_token, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

At this point, GPT-2 can generate text but not perform classification.  
The next step adds a custom output head for binary classification.

---

## 2️⃣ Adding a Classification Head

The GPT-2 model’s final **linear projection layer (`out_head`)** originally maps hidden states to vocabulary logits.  
This layer is replaced with a **two-output linear layer**, representing the classes:
- **0:** Ham (not spam)  
- **1:** Spam  

```
torch.manual_seed(123)
num_classes = 2
model.out_head = nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
```

This conversion enables the model to produce logits for class probabilities instead of next-token predictions.

---

## 3️⃣ Freezing and Unfreezing Parameters

To avoid retraining the entire GPT-2 (which would be computationally expensive), most parameters are **frozen**, while select layers are left trainable:
- **Unfrozen layers:**  
  - The **last transformer block** (`trf_blocks[-1]`)  
  - The **final layer normalization layer** (`final_norm`)  
  - The **new classifier head (`out_head`)**

```
for p in model.parameters():
    p.requires_grad = False

for p in model.trf_blocks[-1].parameters():
    p.requires_grad = True
for p in model.final_norm.parameters():
    p.requires_grad = True
for p in model.out_head.parameters():
    p.requires_grad = True
```

This selective unfreezing allows GPT-2 to **adapt its final representations** to the new classification task without forgetting the general language knowledge from pretraining.

![Figure 6.12 — Selective Fine-Tuning Strategy](images/Screenshot%202025-10-20%20094244.png)

---

## 4️⃣ Forward Pass and Output Verification

To verify that the new classification head works, a simple forward pass is run using an example input text:

```
inputs = text_to_token_ids("Do you have time")
print("Inputs shape:", inputs.shape)

with torch.no_grad():
    outputs = model(inputs)
print("Outputs shape:", outputs.shape)
```

The output tensor has shape **`[batch_size, num_tokens, num_classes]`** —  
for example, `[1, 4, 2]`, meaning:
- **1 sample**,  
- **4 tokens**,  
- **2 output logits** (ham, spam).

The final token’s output is extracted for classification.

```
last_token_logits = outputs[:, -1, :]
print("Last token logits:", last_token_logits)

probs = torch.softmax(last_token_logits, dim=-1)
print("Probabilities (ham, spam):", probs)
```

This produces class probabilities like:
Probabilities (ham, spam): tensor([[0.53, 0.47]])



---

## 🧠 Key Takeaways

| Step | Purpose |
|------|----------|
| **Model Loading** | Initializes GPT-2 with pretrained weights. |
| **Classifier Head** | Replaces the vocabulary projection layer with a binary classifier. |
| **Freezing Strategy** | Freezes most layers to retain pretrained knowledge. |
| **Selective Unfreezing** | Allows only the final layers and head to learn classification features. |
| **Forward Test** | Confirms correct tensor shapes and output probabilities. |

---

## 🧩 Chapter Context
This chapter transitions GPT-2 from a **language generator** into a **text classifier**.  
By modifying only the final layers, we efficiently adapt GPT-2 to the SMS spam dataset prepared in earlier chapters while preserving its linguistic understanding.

![Figure 6.13 — Modified GPT-2 with Classification Head](images/Screenshot%202025-10-20%20094409.png)

---

**Summary:**  
> Chapter 6.5 introduces the classification head on top of GPT-2, freezes most of the model to preserve pretrained knowledge, and validates that the modified architecture outputs correct shapes and probabilities.  
> This step prepares the model for the upcoming fine-tuning process using the spam dataset.

# 🧠 Building a Large Language Model (From Scratch)

> **Senior Design 2025 – University of Connecticut**  
> A step-by-step implementation and study of modern large language model architectures inspired by *Andrej Karpathy’s “Build a Large Language Model (From Scratch)”* and adapted for our Senior Design capstone.

---

## 1. 📘 Project Overview

This project serves as a full-stack exploration of how large language models (LLMs) are designed, trained, and optimized from the ground up.  
Our goal is to **demystify the internals of GPT-style models** by building each component—from tokenization to transformer layers—entirely from first principles.

Unlike high-level frameworks that abstract away internals, this implementation focuses on **transparent, modular construction**.  
Each chapter builds incrementally toward a functional generative model capable of learning from text data and producing coherent output.

---

## 2. 🎯 Objectives & Motivation

The primary objectives of this project are to:

- Understand and reproduce the **core architecture of transformer-based models**.
- Build a **training pipeline** capable of scaling from toy datasets to meaningful samples.
- Develop custom utilities for:
  - Dataset preparation and tokenization.
  - Forward and backward passes.
  - Model checkpointing and continuation.
- Bridge **theoretical understanding** (e.g., self-attention, positional encoding) with **practical engineering** (e.g., GPU acceleration, optimization loops).

> **Motivation:**  
> As LLMs increasingly define the landscape of AI and software automation, hands-on understanding of their construction provides both academic insight and industry relevance.

---

## 3. 🏗️ System Architecture

The project follows a modular architecture divided into distinct components:
```
Senior-Design/
├── chapters/
│ ├── 4 embeddings/
│ ├── 5 pretraining/
│ ├── 6 finetuning/
│ └── ...
├── resources/
│ ├── data.py # Dataloader and tokenization helpers
│ ├── work.py # Core model (Toilet / GPT implementation)
│ └── ...
├── checkpoints/ # Saved model states
├── notebooks/ # Experimental runs and visualizations
├── bonfire.py # Main training script
├── training_arc.py # Training architecture definition
└── README.md
```


Each folder corresponds to a **chapter of development**—reflecting both the learning curve and incremental build process of the model.

---

## 4. 🔬 Implementation Breakdown (By Chapter)

Below is a summary of major milestones:

| Chapter | Focus | Core Concepts |
|:--------:|:------|:--------------|
| 1–3 | Fundamentals | Tokenization, encoding, and data prep |
| 4 | Embedding layers | Token + positional embeddings |
| 5 | Pretraining pipeline | Model initialization, loss tracking, checkpoints |
| 6 | Finetuning | Continued training on domain-specific data |
| 7+ | Extensions | Evaluation, generation, and efficiency improvements |

Each chapter’s code block includes inline explanations and training outputs for clarity.

**Example snippet placeholder:**
```
# Example: positional embeddings from Chapter 4
pos = torch.arange(context_length)
pos_emb = nn.Parameter(torch.zeros(1, context_length, emb_dim))
```

---

## 5. 🧩 Dataset & Tokenization

The project uses both **open text corpora** and custom samples for experimentation.  
Datasets are preprocessed through a tokenizer (initially GPT-2 BPE) using `tiktoken`, with flexibility to integrate other tokenizers.

Pipeline overview:

1. Load raw text → normalize whitespace  
2. Encode tokens via tokenizer  
3. Create training and validation splits  
4. Generate context windows and batches  

**Sample code placeholder:**
```
from resources.data import spawn_dataloader

train_loader, val_loader = spawn_dataloader(text_path="data/input.txt", batch_size=32)
```

---

## 6. ⚙️ Model Configuration & Training

Model configurations are defined as Python dictionaries (e.g., `GPT_CONFIG_124M`) specifying layer counts, embedding size, and dropout rates.

Training leverages modular scripts such as `bonfire.py` and `training_arc.py` which handle:
- Forward/backward propagation
- Loss computation
- Model saving/resuming
- Logging and validation

**Training loop placeholder:**
```
for epoch in range(num_epochs):
    for x, y in train_loader:
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Checkpoints:**  
Intermediate states are automatically saved to `/checkpoints` and can be reloaded for continued training.

---



## 7. 📦 Current Stack

- **Languages:** Python (extensible to other languages for experimentation)
- **Frameworks:** Flexible deep learning backend (currently PyTorch-based)
- **Libraries:** tiktoken, numpy, tqdm
- **Tools:** VS Code, GitHub, local or cloud training environments
- *(Future expansion: TensorFlow, quantization toolkits, efficiency analysis, visualization modules)*

---

### 🧩 Notes

This README is designed as a **living document**—to be updated as training progresses, new chapters are completed, and additional results are obtained.  
All ````` placeholders and `TODO` sections mark expansion points for ongoing development.

# 🚀 Chapter 6.7 — Finetuning the Model to Classify Spam

## Overview
In **Chapter 6.7**, the GPT-2 model is *finetuned* for spam detection.  
After evaluating its untrained baseline in Chapter 6.6, we now optimize its last transformer block, final normalization layer, and new 2-class output head using **PyTorch’s training loop**.

The goal:  
Train the model to correctly classify whether a given text message is **spam (1)** or **not spam (0)** — while monitoring **training loss**, **validation loss**, and **accuracy** across epochs.

---

## 🧩 The Finetuning Process
Figure 6.15 (below) outlines the **typical PyTorch training loop** — each epoch passes over the full training dataset and adjusts model weights to minimize loss.

![Figure 6.15 – The PyTorch Training Loop](images/Screenshot%202025-10-21%20094703.png)

Steps:
1. **For each epoch**, loop through batches in the training set.  
2. **Zero gradients** from the previous iteration.  
3. **Compute loss** on the current batch.  
4. **Perform backpropagation** to compute gradients.  
5. **Update weights** with `optimizer.step()`.  
6. **Periodically print** training and validation loss.  
7. **Compute classification accuracy** per epoch.

---

## 🧠 Training Function: `train_classifier_simple()`

```
def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter, tokenizer):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()  # A: enable gradient updates

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # B: clear gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()        # C: backpropagate
            optimizer.step()       # D: update weights

            examples_seen += input_batch.shape[0]  # E
            global_step += 1

            if global_step % eval_freq == 0:       # F: log periodically
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # G: calculate epoch accuracy
        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc   = calc_accuracy_loader(val_loader,   model, device, num_batches=eval_iter)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Training accuracy: {train_acc*100:.2f}% | Validation accuracy: {val_acc*100:.2f}%")

    return train_losses, val_losses, train_accs, val_accs, examples_seen
```

---

## ⚙️ Model Evaluation Function

```
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
```

This evaluates both training and validation loss without affecting gradients.

---

## 🧮 Finetuning Configuration
The optimizer and loop setup:

```
import time

start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5, tokenizer=tokenizer
)

print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")
```

During training, loss decreases sharply and accuracy rises steadily.  
Example console output:
```
Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
Ep 1 (Step 000050): Train loss 0.617, Val loss 0.637
Ep 2 (Step 000200): Train loss 0.419, Val loss 0.397
Ep 3 (Step 000350): Train loss 0.340, Val loss 0.306
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 5.65 minutes.
```

---

## 📊 Visualizing Loss and Accuracy

Loss and accuracy are plotted using the helper below (Listing 6.11):

```
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="--", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twinx()  # Add second x-axis for examples seen
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
```

When plotted, **training and validation loss** both drop rapidly before flattening, showing convergence:

![Loss Plot (zoomed)](images/Screenshot%202025-10-21%20104213.png)

And the **accuracy curves** reveal near-perfect performance by epoch 5:

![Figure 6.17 — Training vs Validation Accuracy](images/Screenshot%202025-10-21%20095225.png)

---

## ✅ Final Results
After finetuning, the model achieves outstanding accuracy:

```
Training accuracy: 97.21%
Validation accuracy: 97.32%
Test accuracy: 95.67%
```


The close alignment of the training and validation curves indicates **minimal overfitting** —  
the model generalized well to unseen data.

---

## 🧠 Key Takeaways

| Concept | Description |
|----------|-------------|
| **Finetuning** | Adjusting pre-trained GPT-2 weights for spam classification. |
| **Backpropagation** | Computes gradients of loss with respect to model parameters. |
| **Loss & Accuracy Tracking** | Periodic evaluation keeps training stable and interpretable. |
| **AdamW Optimizer** | Improves generalization by applying decoupled weight decay. |
| **Results** | Near-perfect accuracy after just 5 epochs, with strong generalization. |

---

## 🧩 Chapter Summary
Chapter 6.7 completes the **spam classifier finetuning**.  
Through a standard PyTorch loop, the model learns to separate spam and ham texts by retraining only the uppermost layers of GPT-2.  
The low loss and high accuracy curves show that transfer learning is extremely efficient — even with limited training data.

**End result:**  
A GPT-2–based classifier capable of detecting spam messages with over **95% test accuracy**, visualized via both **loss** and **accuracy** plots for transparent performance monitoring.

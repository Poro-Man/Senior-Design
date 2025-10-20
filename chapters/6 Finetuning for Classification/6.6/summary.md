# ⚙️ Chapter 6.6 — Evaluating the Model Before Finetuning

## Overview
Chapter **6.6** of *Build a Large Language Model (From Scratch)* evaluates the GPT-2 classifier **before finetuning**.  
The goal is to establish a **baseline performance** for accuracy and loss using the model from Chapter 6.5.  
At this stage, the model’s output head has been replaced for spam detection, but the weights haven’t been trained yet—so results will be near random.

---

## 1️⃣ Interpreting the Model’s Output
The model outputs logits for each token, but **only the last token’s output** is used for classification.  
That output is converted to **probabilities** with softmax and then reduced to a class label via **argmax**.

```
print("Last output token:", outputs[:, -1, :])

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())
```

Here,  
- **index 0** → *not spam (ham)*  
- **index 1** → *spam*

Since the model isn’t finetuned yet, predictions will often be **incorrect**.

![Figure 6.14 — Interpreting the Model’s Logits as Class Predictions](images/Screenshot%202025-10-20%20102057.png)

---

## 2️⃣ Calculating Classification Accuracy
To quantify baseline performance, the chapter defines a function to compute **accuracy** over a given number of batches.

```
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Only last token
                predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples
```

Then it is used to calculate accuracy for the **training**, **validation**, and **test** sets:

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_acc   = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_acc  = calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Training accuracy: {train_acc*100:.2f}%")
print(f"Validation accuracy: {val_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")
```

Typical untrained results:
```
Training accuracy: 46.25%
Validation accuracy: 45.00%
Test accuracy: 48.75%
```

As expected, the model performs **around 50%**, equivalent to random guessing.

---

## 3️⃣ Defining the Classification Loss
Since accuracy is not differentiable, the model uses **cross-entropy loss** as a proxy objective.  
This function compares the predicted logits of the last token with the target labels.

```
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

---

## 4️⃣ Averaging Loss Across Batches
To obtain average loss values for the training, validation, and test sets, the following loader function iterates through multiple batches:

```
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches
```

**Example output:**
```
Training loss: 3.095
Validation loss: 2.583
Test loss: 2.322
```

These relatively high loss values indicate that the classifier is **not yet optimized**.

---

## 🧠 Key Takeaways

| Step | Purpose |
|------|----------|
| **Softmax + Argmax** | Converts logits into predicted class labels. |
| **Accuracy Function** | Measures correct predictions across batches. |
| **Cross-Entropy Loss** | Differentiable objective used for training. |
| **Baseline Results** | Random-level accuracy (~50%) and high loss. |

---

## 🧩 Chapter Context
This section completes the **pre-finetuning diagnostic** phase.  
It establishes that the model currently performs at random and must be trained using a cross-entropy objective to learn meaningful spam classification.

---

**Summary:**  
> Chapter 6.6 establishes the GPT-2 classifier’s **baseline accuracy (~50%)** and **initial loss values** before fine-tuning.  
> By analyzing logits, probabilities, and classification outputs, it highlights that the model can process inputs but has not yet learned to distinguish spam — motivating the need for the upcoming finetuning stage.

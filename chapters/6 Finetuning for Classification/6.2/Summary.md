# 🧩 Chapter 6.2 — Preparing the SMS Spam Dataset

## Overview
Chapter **6.2** of *Build a Large Language Model (From Scratch)* walks through the **data preprocessing pipeline** for fine-tuning a model on a **spam classification task**.  
It demonstrates how to **download**, **clean**, **balance**, and **split** the SMS Spam dataset before training.  
This preparation ensures that the model learns effectively and avoids bias toward the more common “ham” (non-spam) class.

---

## 1️⃣ Downloading and Extracting the Dataset

The dataset used in this chapter is the **UCI SMS Spam Collection**, which contains thousands of text messages labeled as either *“ham”* or *“spam.”*

The following code downloads the dataset ZIP file, extracts it, and renames the data file for easier access:

```
import urllib.request, zipfile, os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    os.rename(Path(extracted_path) / "SMSSpamCollection", data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
```

Once extracted, the data is loaded into a **Pandas DataFrame**:

```
import pandas as pd
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
df.head()
```

---

## 2️⃣ Balancing the Dataset

Because the dataset contains more *ham* messages than *spam*, balancing is necessary to prevent model bias.  
A subset of *ham* messages equal to the number of *spam* messages is randomly sampled.

```
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
```

After balancing, the labels are converted to numeric form for training:

```
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
```

---

## 3️⃣ Splitting the Dataset

The balanced dataset is split into **training (70%)**, **validation (10%)**, and **test (20%)** sets.  
This ensures that the model can learn, tune hyperparameters, and evaluate performance fairly.

```
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
```

Finally, each split is saved as a separate **CSV file** for future use:

```
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
```

---

## 🧠 Key Takeaways
- **Dataset:** The *SMS Spam Collection* from UCI ML Repository.  
- **Goal:** Prepare clean, balanced, and structured data for spam classification fine-tuning.  
- **Steps:**  
  1. Download and extract dataset.  
  2. Load into a DataFrame.  
  3. Balance *ham* and *spam* messages.  
  4. Encode labels numerically.  
  5. Split data into training, validation, and testing sets.  

---

## 💾 Output Files
| File Name | Purpose |
|------------|----------|
| `train.csv` | Used for model training |
| `validation.csv` | Used for model tuning |
| `test.csv` | Used for model evaluation |

---

**Summary:**  
> Chapter 6.2 provides the foundation for preparing text datasets for LLM finetuning.  
> It emphasizes balanced sampling, reproducible random splits, and modular data handling — crucial for ensuring fair model evaluation and reproducible experiments.

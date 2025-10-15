# ============================================================
# Chapter 6.2 - Preparing the SMS Spam Dataset for Finetuning
# ============================================================

import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

# ---------------------------
# Listing 6.1: Downloading and Unzipping the Dataset
# ---------------------------

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """Download and extract SMS spam dataset from UCI repository"""
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # A. Download dataset
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # B. Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # C. Rename the extracted file
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

# ---------------------------
# Reading the dataset into Pandas
# ---------------------------

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print("\n📊 Dataset Loaded:")
print(df.head(), "\n")

# ---------------------------
# Listing 6.2: Creating a Balanced Dataset
# ---------------------------

def create_balanced_dataset(df):
    """Balance the dataset by sampling equal numbers of spam and ham messages"""
    num_spam = df[df["Label"] == "spam"].shape[0]  # A
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)  # B
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])  # C
    return balanced_df


balanced_df = create_balanced_dataset(df)
print("Balanced dataset class counts:")
print(balanced_df["Label"].value_counts(), "\n")

# Encode labels: ham -> 0, spam -> 1
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# ---------------------------
# Listing 6.3: Splitting the Dataset
# ---------------------------

def random_split(df, train_frac, validation_frac):
    """Randomly split dataset into train, validation, and test sets"""
    # A. Shuffle
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # B. Compute split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # C. Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
print("Train/Validation/Test sizes:", len(train_df), len(validation_df), len(test_df))

# ---------------------------
# Save datasets as CSV files
# ---------------------------

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

print("\n✅ Data preparation complete. Files saved as:")
print(" - train.csv")
print(" - validation.csv")
print(" - test.csv")

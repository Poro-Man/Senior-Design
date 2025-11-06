
import os
import json
import urllib.request



def download_and_load_file(file_path: str, url: str):
    """
    Downloads the instruction dataset if not already available locally.
    Then reads and returns the parsed JSON list.
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)

    # File already exists — just read it
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def format_input(entry: dict) -> str:
    """
    Converts a dataset entry {instruction, input, output}
    into the Alpaca-style prompt format used for supervised finetuning.

    Example:
    Below is an instruction that describes a task.
    Write a response that appropriately completes the request.

    ### Instruction:
    Identify the correct spelling of the following word.

    ### Input:
    Occassion
    """
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # Only include input block if present
    input_part = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""

    return instruction_text + input_part


def partition_dataset(data):
    """
    Splits the dataset into train, validation, and test subsets.
    Returns (train_data, val_data, test_data).
    """
    n = len(data)
    train_portion = int(n * 0.85)
    test_portion = int(n * 0.10)
    val_portion = n - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    return train_data, val_data, test_data


if __name__ == "__main__":
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    # View two example entries
    print("\nExample entry:\n", data[50])
    print("\nAnother example entry:\n", data[999])

    # Format examples into Alpaca-style text
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print("\n---- Formatted (index 50) ----")
    print(model_input + desired_response)

    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"
    print("\n---- Formatted (index 999) ----")
    print(model_input + desired_response)

    # Split dataset and display lengths
    train_data, val_data, test_data = partition_dataset(data)
    print("\nTraining set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

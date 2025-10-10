import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken


class AIDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.trgt_ids = []

        # Tokenize the text
        tkn_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Sliding window approach to create input-target pairs
        for i in range(0, len(tkn_ids) - max_length, stride):
            in_chunks = tkn_ids[i: i + max_length]
            trgt_chunks = tkn_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(in_chunks))
            self.trgt_ids.append(torch.tensor(trgt_chunks))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.trgt_ids[idx]


def spawn_dataloader(
    text,
    tokenizer,
    batch_size=4,
    max_length=512,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    dataset = AIDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


# --------------------------
#quick test
# --------------------------
if __name__ == "__main__":
    # Init tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load text
    try:
        with open("the-verdict.txt", "r", encoding="utf-8") as file:
            raw_txt = file.read()
    except FileNotFoundError:
        raw_txt = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

    # Spawn dataloader
    dataloader = spawn_dataloader(
        raw_txt,
        tokenizer,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False,
        drop_last=False,
    )

    # Inspect first batch
    data_iter = iter(dataloader)
    first_batch = next(data_iter)

    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)

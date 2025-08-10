import tiktoken

import torch
from torch.utils.data import Dataset

import pandas as pd

from torch.utils.data import DataLoader

###################################################
#
#
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length-len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


###################################################
#
#
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    train_dataset = SpamDataset(
        csv_file="train.csv",
        tokenizer=tokenizer,
        max_length=None
    )
    print(train_dataset.max_length)

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        tokenizer=tokenizer,
        max_length=None
    )
    print(val_dataset.max_length)

    test_dataset = SpamDataset(
        csv_file="test.csv",
        tokenizer=tokenizer,
        max_length=None
    )
    print(test_dataset.max_length)    

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False        
    )    
    
    for input_batch, target_batch in train_loader:
        pass

    print("input batch dimensions: ", input_batch.shape)
    print("Label batch dimensions: ", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

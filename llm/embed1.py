import os
import re
import importlib
import tiktoken
from torch.utils.data import DataLoader
from gptdatasetv1 import GPTDatasetV1

import torch

########################################################3
#
#
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

########################################################3
#
#
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("------------inputs shape-------------------")
print(inputs)
print(inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("------------token embeddings shape-------------------")
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("-----------position embeddings shape--------------------")
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("-----------token + position embeddings shape--------------------")
print(input_embeddings.shape)




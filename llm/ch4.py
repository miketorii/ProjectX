import torch
import torch.nn as nn
import tiktoken

#from importlib.metadata import version
#print("torch:", version("torch"))
#print("matplotlib:", version("matplotlib"))
#print("tiktoken:", version("tiktoken"))

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

#################################################
#
#
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        print("---DummyGPTModel---")

    def forward(self, in_idx):
        print("---forward---")
        
#################################################
#
#
tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append( torch.tensor(tokenizer.encode(txt1)) )
batch.append( torch.tensor(tokenizer.encode(txt2)) )
batch = torch.stack(batch, dim=0)

print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

#logits = model(batch)

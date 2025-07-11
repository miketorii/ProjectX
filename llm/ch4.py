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
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"]) ]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear( cfg["emb_dim"], cfg["vocab_size"], bias=False )

    def forward(self, in_idx):
        print("---forward---")
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

#################################################
#
#
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        print("---DummyTransformerBlock---")
        super().__init__()

    def forward(self, x):
        return x

#################################################
#
#
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        print("---LayerNorm---")
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var*self.eps)
        return self.scale * norm_x + self.shift
        
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

logits = model(batch)
print("Output shape: ", logits.shape)
print(logits)

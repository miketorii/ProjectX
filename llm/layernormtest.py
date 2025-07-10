import torch
import torch.nn as nn
import tiktoken


torch.manual_seed(123)

batch_example = torch.randn(2,5)

layer = nn.Sequential( nn.Linear(5,6), nn.ReLU() )
out = layer(batch_example)

print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean: ", mean)
print("Variance: ", var)

out_norm = (out-mean) / torch.sqrt(var)
print("Normalized layer outputs: ", out_norm)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

print("Mean: ", mean)
print("Variance: ", var)

torch.set_printoptions(sci_mode=False)
print("Mean: ", mean)
print("Variance: ", var)


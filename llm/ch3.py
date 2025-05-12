from importlib.metadata import version
import torch

print(version("torch"))

inputs = torch.tensor(
    [[0.43,0.15,0.89],
     [0.55,0.87,0.66],
     [0.57,0.85,0.64],
     [0.22,0.58,0.33],
     [0.77,0.25,0.10],
     [0.05,0.80,0.55]]
)

query = inputs[1]

print(query)

attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    print(x_i)
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)



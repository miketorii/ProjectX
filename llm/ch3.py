from importlib.metadata import version
import torch

print("----Start---")

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

#########################################
res = 0

for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)
print(torch.dot(inputs[0],query))
#########################################

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

#########################################

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

#########################################

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights", attn_weights_2)
print("Sum:", attn_weights_2.sum())

#########################################

query = inputs[1]

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print("context vector")
print(context_vec_2)


##########3.3.2###########################

print("---------------------------------")

attn_scores = torch.empty(6,6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
        
print(attn_scores)

#########################################

attn_scores2 = inputs @ inputs.T
print(attn_scores2)

#########################################

attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)

print("All row sums:", attn_weights.sum(dim=1))

#########################################

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

#########################################

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
print("x_2=", x_2, "d_in=", d_in)

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print("W_query=", W_query)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("query_2=",query_2)

keys = inputs @ W_key
values = inputs @ W_value

print("keys=", keys, keys.shape)
print("values=", values, values.shape)

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print("attn_scores_2=", attn_scores_2)

d_k = keys.shape[1]
print("d_k=", d_k)
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attn_weights_2=", attn_weights_2)

print("attn_weights_2.shape=", attn_weights_2.shape)
print("values.shape=", values.shape)
context_vec_2 = attn_weights_2 @ values
print("context_vec_2=", context_vec_2)

print("----End---")

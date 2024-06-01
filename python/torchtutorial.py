import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

x_ones = torch.ones_like(x_data)
print(x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

shape = (2,3,)
R = torch.rand(shape)
O = torch.ones(shape)
Z = torch.zeros(shape)
print(R)
print(O)
print(Z)

tr = torch.rand(3,4)
print(tr.shape, tr.dtype, tr.device)

tr = torch.randint(4, (3,3))
print(tr)
print(tr[0], tr[:,0], tr[:,-1], tr[...,-1])
tr[:,1] = 4
tr[1,:] = 5
print(tr)

tr1 = torch.zeros(3)
tr2 = torch.ones(4)
print(tr1, tr2)
tr = torch.cat([tr1, tr2])
print(tr)

tr = torch.tensor([[1,2],[1,3]])
print(tr)
tr1 = tr @ tr.T
print(tr1)
tr2 = tr.matmul(tr.T)
print(tr2)
tr3 = tr * tr
print(tr3)
tr4 = tr.mul(tr)
print(tr4)

tr = torch.tensor([[1,2],[1,3]])
agg = tr.sum()
print(agg, type(agg))
agg_item = agg.item()
print(agg_item, type(agg_item))

tr = torch.ones((4,4))
tr[:,1] = 0
print(tr)
tr.add_(5)
print(tr)

import numpy as np
import pandas as pd

from itertools import product
from pulp import LpVariable, lpSum, value
from ortoolpy import model_min, addvars, addvals

df_tc = pd.read_csv('./data/trans_cost.csv', index_col="工場")
print( df_tc.head() )
print( len(df_tc) )
print('----------------------------------------')

df_demand = pd.read_csv('./data/demand.csv')
print( df_demand.head() )
print( len(df_demand) )
print('----------------------------------------')

df_supply = pd.read_csv('./data/supply.csv')
print( df_supply.head() )
print( len(df_supply) )
print('----------------------------------------')

np.random.seed(1)
nw = len(df_tc.index)
nf = len(df_tc.columns)
print(nw)
print(nf)
pr = list( product(range(nw), range(nf)) )
print(pr)

m1 = model_min()
v1 = { (i,j):LpVariable('v%d_%d'%(i,j), lowBound=0) for i,j in pr}

m1 += lpSum( df_tc.iloc[i].iloc[j]*v1[i,j] for i,j in pr)
for i in range(nw):
    m1 += lpSum( v1[i,j] for j in range(nf)) <= df_supply.iloc[0].iloc[i]
for j in range(nf):
    m1 += lpSum( v1[i,j] for i in range(nw)) <= df_demand.iloc[0].iloc[j]    
m1.solve()

df_tr_sol = df_tc.copy()
total_cost = 0
for k,x in v1.items():
    i,j = k[0],k[1]
    df_tr_sol.iloc[i, j] = value(x)
    total_cost += df_tc.iloc[i].iloc[j]*value(x)

print(df_tr_sol)
print(total_cost)

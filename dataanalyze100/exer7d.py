import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

print('---------------product-------------------')
df_material = pd.read_csv('./data/product_plan_material.csv', index_col="製品")
print( df_material.head() )
print( len(df_material) )

print('--------------profit------')
df_profit = pd.read_csv("./data/product_plan_profit.csv")
print( df_profit.head() )
print( len(df_profit) )

print('--------------stock--------------------')
df_stock = pd.read_csv('./data/product_plan_stock.csv')
print( df_stock.head() )
print( len(df_stock) )

print('---------------plan-------------------')
df_plan = pd.read_csv('./data/product_plan.csv')
print( df_plan.head() )
print( len(df_plan) )

print('---------------------------------------------')
print('---------------------------------------------')

import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

print('---------------product-------------------')
df_material = pd.read_csv('./data/product_plan_material.csv', index_col="製品")
print( df_material.head() )
print( len(df_material) )

print('--------------profit------')
df_profit = pd.read_csv("./data/product_plan_profit.csv", index_col='製品')
print( df_profit.head() )
print( len(df_profit) )

print('--------------stock--------------------')
df_stock = pd.read_csv('./data/product_plan_stock.csv', index_col='項目')
print( df_stock.head() )
print( len(df_stock) )

print('---------------plan-------------------')
df_plan = pd.read_csv('./data/product_plan.csv', index_col='製品')
print( df_plan.head() )
print( len(df_plan) )

print('---------------------------------------------')

def product_plan(df_profit, df_plan):
    profit = 0
    for i in range( len(df_profit.index) ):
        for j in range( len(df_plan.columns) ):
            a = df_profit.iloc[i].iloc[j]
            b = df_plan.iloc[i].iloc[j]
            profit += a*b
    return profit

print( product_plan(df_profit, df_plan) )


print('---------------------------------------------')

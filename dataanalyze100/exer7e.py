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
df_replan = pd.read_csv('./data/product_replan.csv', index_col='製品')
print( df_replan.head() )
print( len(df_replan) )

print('---------------------------------------------')

def condition_stock(df_plan, df_material,df_stock):
    flag = np.zeros( len(df_material.columns) )
    for i in range( len(df_material.columns) ):
        temp_sum = 0
        for j in range( len(df_material.index) ):
            temp_sum = temp_sum + df_material.iloc[j].iloc[i]*float( df_plan.iloc[j])
        if( temp_sum <= df_stock.iloc[0].iloc[i] ):
            flag[i] = 1
        print( df_material.columns[i] )
        print( temp_sum )
        print( df_stock.iloc[0].iloc[i] )
    return flag

print( condition_stock(df_replan, df_material, df_stock) )

print('---------------------------------------------')

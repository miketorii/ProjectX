import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

print('-----------df_tc_new trans_cost_new.csv------')
df_tr = pd.read_csv('./data/trans_cost_min.csv', index_col="工場")
print( df_tr.head() )
print( len(df_tr) )

print('--------------df_pos trans_rout_pos.csv------')
df_pos = pd.read_csv("./data/trans_route_pos.csv")
print( df_pos.head() )
print( len(df_pos) )


print('--------------demand--------------------')
df_demand = pd.read_csv('./data/demand.csv')
print( df_demand.head() )
print( len(df_demand) )

print('---------------dupply-------------------')
df_supply = pd.read_csv('./data/supply.csv')
print( df_supply.head() )
print( len(df_supply) )

print('---------------------------------------------')

def condition_demand(df_tr, df_demand):
    print("---condition_demand----")
    flag = np.zeros( len(df_demand.columns) )
    for i in range( len(df_demand.columns) ):
        temp_sum = sum( df_tr[df_demand.columns[i]] )
        if( temp_sum >= df_demand.iloc[0].iloc[i] ):
            flag[i] = 1
    return flag
    
def condition_supply(df_tr, df_supply):
    print("---condition_supply----")
    flag = np.zeros( len(df_supply.columns) )
    for i in range( len(df_supply.columns) ):
        temp_sum = sum( df_tr.loc[df_supply.columns[i]] )
        if( temp_sum <= df_supply.iloc[0].iloc[i] ):
            flag[i] = 1
    return flag
    
print("---new demand---")    
print( condition_demand(df_tr, df_demand) )
print("---new supply---")
print( condition_supply(df_tr, df_supply) )


print('---------------------------------------------')

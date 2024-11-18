import pandas as pd

df_tr = pd.read_csv("./data/trans_route.csv", index_col="工場")
print( df_tr.head() )
df_tc = pd.read_csv("./data/trans_cost.csv", index_col="工場")
print( df_tc.head() )

def trans_cost(df_tr, df_tc):
    cost = 0
    for i in range(len(df_tc.index)):
            for j in range(len(df_tr.columns)):
                cost += df_tr.iloc[i].iloc[j]*df_tc.iloc[i].iloc[j]
    return cost

print( trans_cost(df_tr, df_tc) )

        
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")





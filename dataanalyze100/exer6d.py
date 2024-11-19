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

df_demand = pd.read_csv("./data/demand.csv")
df_supply = pd.read_csv("./data/supply.csv")
print( df_demand.head() )
print( df_supply.head() )

print("---demand---")
for i in range( len(df_demand.columns) ):
    temp_sum = sum( df_tr[ df_demand.columns[i] ] )

    print("Volume to "+str(df_demand.columns[i])+" is "+str(temp_sum))    
    print("Supply volume "+str(df_demand.iloc[0].iloc[i]))
    
    if temp_sum >= df_demand.iloc[0].iloc[i]:
        print("Under supply")
    else:
        print("Over. Need to recalculate")

print("---supply---")        
for i in range( len(df_supply.columns) ):
    temp_sum = sum( df_tr.loc[ df_supply.columns[i] ] )

    print("Volume from "+str(df_supply.columns[i])+" is "+str(temp_sum))    
    print("Max supply volume "+str(df_supply.iloc[0].iloc[i]))
    
    if temp_sum <= df_supply.iloc[0].iloc[i]:
        print("Under supply")
    else:
        print("Over. Need to recalculate")        



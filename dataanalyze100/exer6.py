import pandas as pd

factories = pd.read_csv("./data/tbl_factory.csv", index_col=0)
print( factories.head() )

warehouses = pd.read_csv("./data/tbl_warehouse.csv", index_col=0 )
print( warehouses.head() )

cost = pd.read_csv("./data/rel_cost.csv", index_col=0 )
print( cost.head() )

trans = pd.read_csv("./data/tbl_transaction.csv", index_col=0 )
print( trans.head() )

join_data = pd.merge( trans, cost, left_on=["ToFC","FromWH"], right_on=["FCID","WHID"], how="left" )
print( join_data.head() )
print( len(join_data) )

join_data = pd.merge( join_data, factories, left_on=["ToFC"], right_on=["FCID"], how="left" )
print( join_data.head() )

join_data = pd.merge( join_data, warehouses, left_on=["FromWH"], right_on=["WHID"], how="left" )
join_data = join_data[ ["TransactionDate","Quantity","Cost","ToFC","FCName","FCDemand","FromWH","WHName","WHSupply","WHRegion"] ]
print( join_data.head() )

kanto = join_data.loc[ join_data["WHRegion"]=="関東" ]
print( kanto.head() )

tohoku = join_data.loc[ join_data["WHRegion"]=="東北" ]
print( tohoku.head() )

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

print( "関東 Cost = " , kanto["Cost"].sum() )
print( "東北 Cost = ", tohoku["Cost"].sum() )

print( "関東 Quantity = " , kanto["Quantity"].sum() )
print( "東北 Quantity = ", tohoku["Quantity"].sum() )

kanto_per_q = kanto["Cost"].sum() / kanto["Quantity"].sum() * 10000
print( kanto_per_q )
tohoku_per_q = tohoku["Cost"].sum() / tohoku["Quantity"].sum() * 10000
print( tohoku_per_q )

cost_chk = pd.merge( cost, factories, on="FCID", how="left" )
kanto_mean = cost_chk["Cost"].loc[ cost_chk["FCRegion"]=="関東" ].mean()
print( kanto_mean )
tohoku_mean = cost_chk["Cost"].loc[ cost_chk["FCRegion"]=="東北" ].mean()
print( tohoku_mean )


print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")





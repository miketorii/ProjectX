import pandas as pd

factories = pd.read_csv("./data/tbl_factory.csv", index_col=0)
print( factories.head() )

warehouses = pd.read_csv("./data/tbl_warehouse.csv", index_col=0 )
print( warehouses.head() )

cost = pd.read_csv("./data/rel_cost.csv", index_col=0 )
print( cost.head() )

trans = pd.read_csv("./data/tbl_transaction.csv", index_col=0 )
print( trans.head() )

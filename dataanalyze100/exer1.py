import pandas as pd

trans1 = pd.read_csv('./data/transaction_1.csv')
print( trans1.head() )

trans2 = pd.read_csv('./data/transaction_2.csv')
print( trans2.head() )

trans = pd.concat([trans1, trans2], ignore_index=True)
print( trans.head() )

print( len(trans1) )
print( len(trans2) )
print( len(trans) )

print("===============================================")
trans1d = pd.read_csv('./data/transaction_detail_1.csv')
print( trans1d.head() )

trans2d = pd.read_csv('./data/transaction_detail_2.csv')
print( trans2d.head() )

transd = pd.concat([trans1d, trans2d], ignore_index=True)
print( transd.head() )

print( len(trans1d) )
print( len(trans2d) )
print( len(transd) )

print("===============================================")
joindata = pd.merge(transd,trans[ ["transaction_id", "payment_date", "customer_id"] ], on="transaction_id", how="left")
print( joindata.head() )

print( len(joindata) )

print("===============================================")
customer = pd.read_csv('./data/customer_master.csv')
print( customer.head() )

item = pd.read_csv('./data/item_master.csv')
print( item.head() )

joindata = pd.merge(joindata, customer, on="customer_id", how="left")
joindata = pd.merge(joindata, item, on="item_id", how="left")
print( joindata.head() )

print( len(joindata) )

print("===============================================")
joindata["price"] = joindata["quantity"] * joindata["item_price"]
print( joindata[["quantity", "item_price", "price"]].head() )

print("===============================================")
print( joindata["price"].sum() )
print( trans["price"].sum() )

print("===============================================")
print( joindata.isnull().sum() )
print( joindata.describe() )

print("===============================================")
print( joindata.dtypes )

joindata["payment_date"] = pd.to_datetime( joindata["payment_date"] )
joindata["payment_month"] = joindata["payment_date"].dt.strftime("%Y%m")
print(joindata[["payment_date","payment_month"]].head())

#print( joindata.dtypes )

print( joindata.groupby("payment_month").sum(numeric_only=True)["price"] )

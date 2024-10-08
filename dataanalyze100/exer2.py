import pandas as pd
#import matplotlib.pyplot as plt

uriagedata = pd.read_csv('./data/uriage.csv')
print( uriagedata.head() )
print( len(uriagedata) )

print("===============================================")
kokyakudata = pd.read_excel('./data/kokyaku_daicho.xlsx')
print( kokyakudata.head() )
print( len(kokyakudata) )

print("===============================================")
print( uriagedata["item_name"].head() )
print( uriagedata["item_price"].head() )

print("===============================================")



print("===============================================")
uriagedata["purchase_date"] = pd.to_datetime( uriagedata["purchase_date"] )
uriagedata["purchase_month"] = uriagedata["purchase_date"].dt.strftime("%Y%m")
print( uriagedata[["purchase_date","purchase_month"]].head() )

print( pd.pivot_table( uriagedata, index="purchase_month", columns="item_name", aggfunc="size", fill_value=0) )
print( pd.pivot_table( uriagedata, index="purchase_month", columns="item_name", values="item_price", aggfunc="sum", fill_value=0) )

print("===============================================")
print( len(pd.unique(uriagedata["item_name"])) )

uriagedata["item_name"] = uriagedata["item_name"].str.upper()
uriagedata["item_name"] = uriagedata["item_name"].str.replace("　","")
uriagedata["item_name"] = uriagedata["item_name"].str.replace(" ","")
print( uriagedata.sort_values(by=["item_name"], ascending=True) )

print( pd.unique(uriagedata["item_name"]) )
print( len(pd.unique(uriagedata["item_name"])) )

print("===============================================")
print( uriagedata.isnull().any(axis=0) )
fig_is_null = uriagedata["item_price"].isnull()
#print(fig_is_null)
print(uriagedata.head())
for trg in list( uriagedata.loc[fig_is_null, "item_name"].unique() ):
    price = uriagedata.loc[ (~fig_is_null) & (uriagedata["item_name"]==trg), "item_price" ].max()
    uriagedata.loc[(fig_is_null) & (uriagedata["item_name"]==trg), "item_price"] = price

print("---confirm---")
print(uriagedata.head())
print( uriagedata.isnull().any(axis=0) )
for trg in list( uriagedata.loc[fig_is_null, "item_name"].unique() ):
    print( trg + " MAX " + str(uriagedata.loc[ uriagedata["item_name"]==trg ]["item_price"].max() ) )
    print( trg + " MIN " + str(uriagedata.loc[ uriagedata["item_name"]==trg ]["item_price"].min(skipna=False) ) )    













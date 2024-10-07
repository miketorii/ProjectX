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

import numpy as np
import pandas as pd

print('--------------df_tc_new trans_cost_new.csv------')
df_tr = pd.read_csv('./data/trans_cost_min.csv', index_col="工場")
print( df_tr.head() )
print( len(df_tr) )


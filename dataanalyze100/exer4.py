import pandas as pd
import matplotlib.pyplot as plt

print("==========================================")

uselogdata = pd.read_csv("./data/use_log.csv")
print( uselogdata.head() )
print( uselogdata.isnull().sum() )

customerdata = pd.read_csv("./customer_join.csv")
print( customerdata.head() )
print( customerdata.isnull().sum() )

print("==========================================")



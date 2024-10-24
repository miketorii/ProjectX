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

customercluster = customerdata.max(numeric_only=True)
print( customercluster.head() )

customercluster = customerdata.max(numeric_only=True)["membership_period"]
print(customercluster)



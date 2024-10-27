import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("==========================================")

uselogdata = pd.read_csv("./data/use_log.csv")
print( uselogdata.head() )
print( uselogdata.isnull().sum() )

customerdata = pd.read_csv("./customer_join2.csv")
print( customerdata.head() )
print( customerdata.isnull().sum() )

print("==========================================")

#customercluster = customerdata.max(numeric_only=True)
#print( customercluster.head() )

#customercluster = customerdata.max(numeric_only=True)["membership_period"]
#print(customercluster)

customercluster = customerdata[["mean","median","max","min","membership_period"]]
print(customercluster.head())

sc = StandardScaler()
customer_clustering_sc = sc.fit_transform( customercluster )

kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit( customer_clustering_sc )
customercluster = customercluster.assign(cluster = clusters.labels_)

print(customercluster["cluster"].unique())
print(customercluster.head())
print(customercluster.tail())

print("==========================================")

print( customercluster.groupby("cluster").count() )

print("------------------------------------------")
print( customercluster.groupby("cluster").mean() )

print("==========================================")






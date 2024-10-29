import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
X = customercluster
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df["cluster"] = customercluster["cluster"]

for i in customercluster["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"]==i]
    plt.scatter(tmp[0], tmp[1])

plt.savefig("exer4.png")

print("==========================================")

customercluster = pd.concat([customercluster, customerdata], axis=1 )
print(customercluster.head())
print( customercluster.groupby(["cluster","is_deleted"],as_index=False).count()[["cluster","is_deleted","customer_id"]] )
print( customercluster.groupby(["cluster","routine_flg"],as_index=False).count()[["cluster","routine_flg","customer_id"]] )

print("==========================================")

uselogdata["usedate"] = pd.to_datetime( uselogdata["usedate"] )
uselogdata["年月"] = uselogdata["usedate"].dt.strftime("%Y%m")
uselog_months = uselogdata.groupby(["年月","customer_id"], as_index=False).count()
uselog_months.rename(columns={"log_id":"count"}, inplace=True)
del uselog_months["usedate"]
print( uselog_months.head() )

year_months = list( uselog_months["年月"].unique() )
predict_data = pd.DataFrame()

for i in range(6, len(year_months) ):
    tmp = uselog_months.loc[uselog_months["年月"]==year_months[i]].copy()
    tmp.rename(columns={"count":"count_pred"}, inplace=True)
    for j in range(1,7):
        tmp_before = uselog_months.loc[uselog_months["年月"]==year_months[i-j]].copy()
        del tmp_before["年月"]
        tmp_before.rename(columns={"count":"count_{}".format(j-1) }, inplace=True)
        tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    predict_data = pd.concat([predict_data, tmp], ignore_index=True)

print( predict_data.head() )

predict_data = predict_data.dropna()
predict_data = predict_data.reset_index(drop=True)

print( predict_data.head() )











import pandas as pd

customer = pd.read_csv('./data/customer_join2.csv')
print(customer.head())

uselog_months = pd.read_csv('./data/use_log_months.csv')
print(uselog_months.head())

year_months = list(uselog_months["年月"].unique())
uselog = pd.DataFrame()

for i in range(1, len(year_months)):
    tmp = uselog_months.loc[ uselog_months["年月"]==year_months[i] ].copy()
    tmp.rename(columns={"count":"count_0"}, inplace=True)
    tmp_before = uselog_months.loc[uselog_months["年月"]==year_months[i-1]].copy()
    del tmp_before["年月"]
    tmp_before.rename(columns={"count":"count_1"}, inplace=True)
    tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    uselog = pd.concat([uselog, tmp], ignore_index=True)

print(uselog.head())

               
    

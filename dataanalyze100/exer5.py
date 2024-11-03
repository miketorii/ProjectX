import pandas as pd

from dateutil.relativedelta import relativedelta


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

print( len(uselog) )
print(uselog.head())

print("=====================================================")

exit_customer = customer.loc[ customer["is_deleted"]==1 ].copy()
exit_customer["exit_date"] = None

exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])

for i in exit_customer.index:
    exit_customer.loc[i, "exit_date"] = exit_customer.loc[i, "end_date"] - relativedelta(months=1)

exit_customer["exit_date"] = pd.to_datetime(exit_customer["exit_date"])
exit_customer["年月"] = exit_customer["exit_date"].dt.strftime("%Y%m")
uselog["年月"] = uselog["年月"].astype(str)

print(exit_customer.head())

exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id","年月"], how="left")

print( "uselog length=" , len(uselog) )
print( "exit_uselog length=" , len(exit_uselog) )

print( exit_uselog.head() )

exit_uselog = exit_uselog.dropna(subset=["name"])

print( "exit_uselog length=" , len(exit_uselog) )
print( exit_uselog["customer_id"].unique() )

print( exit_uselog.head() )

print("=====================================================")


    

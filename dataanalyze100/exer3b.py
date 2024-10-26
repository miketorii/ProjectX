import pandas as pd
import matplotlib.pyplot as plt

print("====================================================")

uselogdata = pd.read_csv("./data/use_log.csv")
print( uselogdata.head() )
print( len(uselogdata) )

print("---------------------------------------------------")

customerdata = pd.read_csv("./data/customer_master.csv")
print( customerdata.head() )
print( len(customerdata) )

print("---------------------------------------------------")

campaignmasterdata = pd.read_csv("./data/campaign_master.csv")
print( campaignmasterdata.head() )
print( len(campaignmasterdata) )

print("---------------------------------------------------")

classmasterdata = pd.read_csv("./data/class_master.csv")
print( classmasterdata.head() )
print( len(classmasterdata) )

print("---------------------------------------------------")

print("====================================================")

customerjoindata = pd.merge( customerdata, classmasterdata, on="class", how="left")
customerjoindata = pd.merge( customerjoindata, campaignmasterdata, on="campaign_id", how="left")
print(customerjoindata.head())
print( len(customerjoindata) )

print( len(customerdata) )
print( customerjoindata.isnull().sum() )

print("====================================================")

calcval = customerjoindata.groupby("class_name").count()["customer_id"]
print(calcval)

calcval = customerjoindata.groupby("campaign_name").count()["customer_id"]
print(calcval)

calcval = customerjoindata.groupby("gender").count()["customer_id"]
print(calcval)

calcval = customerjoindata.groupby("is_deleted").count()["customer_id"]
print(calcval)

customerjoindata["start_date"] = pd.to_datetime( customerjoindata["start_date"] )
print(customerjoindata.head())

print("---------------------------------------------------")

customerstart = customerjoindata.loc[customerjoindata["start_date"]>pd.to_datetime("20180401")]
print(customerstart.head())
print("---------------------------------------------------")
print(customerstart.tail())

print("====================================================")

customerjoindata["end_date"] = pd.to_datetime( customerjoindata["end_date"] )
customernewer = customerjoindata.loc[(customerjoindata["end_date"]>=pd.to_datetime("20190331"))|(customerjoindata["end_date"].isna())]
print(customernewer.head())
print( len(customernewer) )

calcval = customernewer.groupby("class_name").count()["customer_id"]
print(calcval)

calcval = customernewer.groupby("campaign_name").count()["customer_id"]
print(calcval)

calcval = customernewer.groupby("gender").count()["customer_id"]
print(calcval)

print("====================================================")
print("====================================================")

uselogdata["usedate"] = pd.to_datetime( uselogdata["usedate"] )
uselogdata["年月"] = uselogdata["usedate"].dt.strftime("%Y%m")
uselogmonth = uselogdata.groupby(["年月","customer_id"], as_index=False).count()
uselogmonth.rename( columns={"log_id":"count"}, inplace=True )
del uselogmonth["usedate"]
print(uselogmonth.head())

uselogcustomer = uselogmonth.groupby("customer_id").agg(max=('count','max'), min=('count','min'), mean=('count','mean'), median=('count', 'median'))
uselogcustomer = uselogcustomer.reset_index(drop=False)
print(uselogcustomer.head())

#uselogcustomer = uselogmonth.groupby("customer_id").mean(numeric_only=True)["count"]
#uselogcustomer = uselogmonth.groupby("customer_id").agg( "mean", numeric_only=True )["count"]
#print(uselogcustomer.head())

#uselogcustomer = uselogmonth.groupby("customer_id").agg( "median", numeric_only=True )["count"]
#print(uselogcustomer.head())

#uselogcustomer = uselogmonth.groupby("customer_id").agg( "max", numeric_only=True )["count"]
#print(uselogcustomer.head())

#uselogcustomer = uselogmonth.groupby("customer_id").agg( "min", numeric_only=True )["count"]
#print(uselogcustomer.head())

print("====================================================")

print( uselogdata.head() )
uselogdata["weekday"] = uselogdata["usedate"].dt.weekday
uselogweekday = uselogdata.groupby(["customer_id","年月","weekday"], as_index=False).count()[["customer_id","年月","weekday","log_id"]]
uselogweekday.rename( columns={"log_id":"count"}, inplace=True )
print( uselogweekday.head() )

uselogweekday = uselogweekday.groupby("customer_id", as_index=False).max(  numeric_only=True )[["customer_id","count"]]
uselogweekday["routine_flg"] = 0
uselogweekday["routine_flg"] = uselogweekday["routine_flg"].where( uselogweekday["count"]<4, 1)
print( uselogweekday.head() )

print("====================================================")
print("====================================================")
print("====================================================")
print("====================================================")
print("====================================================")

print(customerjoindata.head())
print("---------------------------------------------------")
print(uselogcustomer.head())
print("---------------------------------------------------")
print( uselogweekday.head() )
print("---------------------------------------------------")

customerjoindata = pd.merge( customerjoindata, uselogcustomer, on="customer_id", how="left")

customerjoindata = pd.merge( customerjoindata, uselogweekday[["customer_id","routine_flg"]], on="customer_id", how="left")
print(customerjoindata.head())

print( customerjoindata.isnull().sum() )

print("====================================================")
print("====================================================")
print("====================================================")
print("====================================================")
print("====================================================")

'''
import datetime
from dateutil.relativedelta import relativedelta

d = datetime.datetime(2023,4,1)
r = relativedelta(years=1)
print(d+r)
'''

from dateutil.relativedelta import relativedelta

customerjoindata["calc_date"] = customerjoindata["end_date"]
customerjoindata["calc_date"] = customerjoindata["calc_date"].fillna(pd.to_datetime("20190430"))
customerjoindata["membership_period"] = 0

for i in range( len(customerjoindata) ):
    delta = relativedelta(customerjoindata["calc_date"].iloc[i], customerjoindata["start_date"].iloc[i] )
    customerjoindata.loc[i, "membership_period"] = delta.years*12 + delta.months
    
print(customerjoindata.head())
print(customerjoindata[1000:1020])
print(customerjoindata.loc[10])
print(customerjoindata[ customerjoindata["customer_id"] == "OA789036" ] )

print("====================================================")

#print( customerjoindata.describe() )
print( customerjoindata.describe().loc[["min","max","mean","50%"]] )
routinedata = customerjoindata.groupby("routine_flg").count()["customer_id"]
print(routinedata)

plt.hist( customerjoindata["membership_period"] )
plt.savefig("exer3.png")

print("====================================================")

customerend = customerjoindata.loc[ customerjoindata["is_deleted"]==1 ]
print( customerend.describe() )

customerstay = customerjoindata.loc[ customerjoindata["is_deleted"]==0 ]
print( customerstay.describe() )

customerjoindata.to_csv("customer_join2.csv", index=False)

print("====================================================")







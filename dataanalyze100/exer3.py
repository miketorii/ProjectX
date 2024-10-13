import pandas as pd

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


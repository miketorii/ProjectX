import pandas as pd

############### 91

survey = pd.read_csv("./data/survey.csv")
print(len(survey))
print(survey.head())

num = survey.isna().sum()
print(num)

survey = survey.dropna()

num = survey.isna().sum()
print(num)

############### 92

survey["comment"] = survey["comment"].str.replace("AA","")
print( survey.head() )





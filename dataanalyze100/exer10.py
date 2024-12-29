import pandas as pd
import matplotlib.pyplot as plt

############### 91

survey = pd.read_csv("./data/survey.csv")
print(len(survey))
print(survey.head())

print("------------------------------")

num = survey.isna().sum()
print(num)

survey = survey.dropna()

num = survey.isna().sum()
print(num)

############### 92

survey["comment"] = survey["comment"].str.replace("AA","")
print("------------------------------")
print( survey.head() )

survey["comment"] = survey["comment"].str.replace("\(.+?\)","",regex=True)
survey["comment"] = survey["comment"].str.replace("\（.+?\）","",regex=True)
print("------------------------------")
print( survey.head() )

############### 93 ######################

survey["length"] = survey["comment"].str.len()

print("------------------------------")
print( survey.head() )

plt.hist(survey["length"])
plt.savefig("exer10.png")

############### 9x ######################
############### 9x ######################
############### 9x ######################










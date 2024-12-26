import pandas as pd

survey = pd.read_csv("./data/survey.csv")
print(len(survey))
print(survey.head())

num = survey.isna().sum()
print(num)

survey = survey.dropna()

num = survey.isna().sum()
print(num)



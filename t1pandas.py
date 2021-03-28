import pandas as pd
import matplotlib.pyplot as plt

# print version
print(pd.__version__)

data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

df = pd.DataFrame(data)

print(df)
print(df.loc[1])

df = pd.read_csv('data2.csv')
print(df)

x = df.Height
y = df.Latitude
plt.scatter(x,y)
plt.show()

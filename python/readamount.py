import pandas as pd
import matplotlib.pyplot as plt

filename = "salesamount.xlsx"
df = pd.read_excel(filename)
df = df.fillna("NA")

print(df.head)

df = df.T
data_dict = df.to_dict(orient="list")
lines = list(data_dict.values())
drawlines = lines[2][1:]

plt.plot(range(1,13),drawlines, marker="o", label="Sales")
plt.xlabel("Month")
plt.ylabel("Sales Count")
plt.title("Monthly Salses")
plt.ylim(3000,5000)
plt.xticks(range(1,13),["1","2","3","4","5","6","7","8","9","10","11","12"])
plt.legend()
plt.grid(True)
plt.show()
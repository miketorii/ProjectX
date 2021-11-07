import pandas as pd
import matplotlib.pyplot as plt

d = {'score':[10,20,25,40],
     'usr':['usr1','usr2','usr1','usr2']
     }

df = pd.DataFrame(d)
print(df)

p = df.groupby('usr')['score'].sum()
print(p)

p.plot(kind='bar')
plt.show()










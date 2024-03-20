import pandas as pd
import matplotlib.pyplot as plt

#d = {'3.0':20, '1.0':30, '2.0':50 }

d = {'purpose':['3.0','1.0','2.0'],'val':[30, 20, 50]}
df = pd.DataFrame(d)
print(df)

df1 = df.sort_values(by='purpose')
print(df1)

df1.plot(x=df1.columns[0], kind='bar')
plt.show()

#######################################

d2 = {'purpose':['3.0','1.0','2.0'],'val':[40, 80, 10]}
dftmp = pd.DataFrame(d2)
print(dftmp)

df2 = dftmp.sort_values(by='purpose')
print(df2)

df2.plot(x=df1.columns[0], kind='bar')
plt.show()

#######################################

d3 = {'purpose':['3.0','1.0','2.0','4.0'],'val':[40, 30, 10,20]}
dftmp = pd.DataFrame(d3)
print(dftmp)

df3 = dftmp.sort_values(by='purpose')
print(df3)

d4 = {'purpose':['3.0','1.0','2.0'],'val':[40, 80, 70]}
dftmp = pd.DataFrame(d4)
print(dftmp)

df4 = dftmp.sort_values(by='purpose')
print(df4)

fig, axes = plt.subplots(2,2,figsize=(10,5))
df1.plot(x=df1.columns[0], kind='bar',ax=axes[0,0])
df2.plot(x=df1.columns[0], kind='bar',ax=axes[0,1])
df3.plot(x=df1.columns[0], kind='bar',ax=axes[1,0])
df4.plot(x=df1.columns[0], kind='bar',ax=axes[1,1])
plt.show()






import numpy as np
from dezero import Variable
import dezero.functions as F

print("------------start-------------------")
a = np.array([1,2,3])
b = np.array([4,5,6])
a,b = Variable(a), Variable(b)

print(a)
print(b)

c = F.matmul(a,b)
print('--------------')
print(c)

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

print(a)
print(b)

c = F.matmul(a,b)
print('--------------')
print(c)


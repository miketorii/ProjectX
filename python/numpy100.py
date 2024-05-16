import numpy as np

print(np.__version__)

Z = np.zeros(10)
print(Z)

print(Z.size * Z.itemsize)

#np.info(np.add)

ar10 = np.zeros(10)
ar10[4] = 1
print(ar10)

ar = np.arange(10,50)
print(ar)

ar = np.arange(10)
arfp = np.flipud(ar)
#arfp = ar[::-1]
print(arfp)

ar = np.arange(9)
ar1 = ar.reshape(3,3)
print(ar1)

ar = np.array([1,2,0,0,4,0])
idx = np.where(ar!=0)
nonzeroidx = np.nonzero(ar)
print(ar)
print(idx)
print(nonzeroidx)

arI = np.eye(3)
print(arI)

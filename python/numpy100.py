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

ar = np.random.random((3,3,3))
print(ar)

ar = np.random.rand(10,10)
print(ar)
armax = np.amax(ar)
armin = np.amin(ar)
print("max=", armax, "min=", armin)

ar = np.random.rand(30)
print(ar)
arm = np.mean(ar)
print("mean=", arm)

ar = np.ones((3,3))
print(ar)
ar[1,1] = 0
print(ar)
ar = np.ones((10,10))
print(ar)
ar[1:-1,1:-1] = 0
print(ar)

#ar = np.random.rand(10,10)
ar = np.ones((10,10))
print(ar)
sp = ar.shape
print("shape=", sp, sp[0], sp[1])
xb = sp[0]-1
yb = sp[1]-1
ar[0,0:] = 0
ar[xb,0:] = 0
ar[0:,0] = 0
ar[0:,yb] = 0
print(ar)

ar = np.ones((10,10))
print(ar)
ar = np.pad(ar, 1, mode="constant", constant_values=0)
print(ar)

print(0*np.nan)  #nan
print(np.nan==np.nan) #false
print(np.inf > np.nan) #false
print(np.nan - np.nan) #nan
print(0.3 == 3*0.1) #false

ar = np.zeros((5,5))
print(ar)
arin = np.array([1,2,3,4])
ar = ar + np.diag(arin, k=-1)
print(ar)

ar = 1+np.arange(4)
Z = np.diag(ar, k=-1)
print(Z)

ar = np.array([[1,0],[0,1]])
ar = np.tile(ar, (4,4))
print(ar)

print("===use slice= start:end:step===")
ar = np.zeros((8,8))
ar[1::2, ::2] = 1
ar[::2, 1::2] = 1
print(ar)

ar = np.unravel_index(100, (6,7,8))
print(ar)

ar = np.random.random((5,5))
norms = np.linalg.norm(ar, axis=1,keepdims=True)
print("norms=", norms)
ar = ar / norms
print(ar)

Z = np.random.rand(5,5)
print(Z)
Zmax = Z.max()
print(Zmax)
Zmin = Z.min()
print(Zmin)
Z = (Z - Zmin) / (Zmax - Zmin)
print(Z)

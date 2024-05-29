import numpy as np

print(np.__version__)

'''
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

ar = np.dtype([
("r", np.ubyte),("g", np.ubyte),("b", np.ubyte)
])
print(ar)
print(ar["r"])

A = np.arange(15).reshape(5,3)
B = np.arange(6).reshape(3,2)
C = A.dot(B)
print(C)

###################################
#
print(range(5))
print(sum(range(5),-1))
from numpy import *
print(range(5))
print(sum(range(5),-1))

ar = np.arange(5)
print(2 << ar >>  2)

ar = np.arange(0)
print(ar/ar)
print(ar//ar)
#ar = np.arange(np.nan)
#print("astype ", ar.astype(int).astype(float))

#ar = np.arange(10, dtype= "float")
ar = np.random.rand(10)
print(ar)
ar = np.floor(ar)
print(ar)

Z = np.random.uniform(-10, 10, 10)
print(Z)
X = np.trunc(Z + np.copysign(0.5, Z))
print(X)

ar1 = np.random.randint(0,10,10)
ar2 = np.random.randint(0,10,10)
print(ar1)
print(ar2)
ar3 = np.intersect1d(ar1, ar2)
print(ar3)

#print(np.emath.sqrt(-1)==np.sqrt(-1))

today = np.datetime64("today", "D")
print(today)
y = today - np.timedelta64(1, "D")
print(y)
t = today + np.timedelta64(1, "D")
print(t)

dates = np.arange("2016-06","2016-07",dtype="datetime64[D]")
print(dates)

#################################

A = np.ones(5)
B = np.ones(5) * 2
print(A)
print(B)
B += A
A *= -1
B *= A
B /= 2
print(B)
#C = (A+B) * -A / 2
#print(C)
A = np.ones(5)
B = np.ones(5) * 2
print(A)
print(B)
np.add(A, B, out=B)
np.multiply(A, B, out=B)
np.negative(B, out=B)
np.divide(B, 2, out=B)
print(B)

################################
#
Z = np.random.uniform(0,10,10)
print(Z)
X = np.trunc(Z)
print("trunc",X)
X = np.floor(Z)
print("floor", X)
X = np.ceil(Z)
print("ceil", X)
X = np.fix(Z)
print("fix", X)
X = np.rint(Z)
print("rint", X)
X = Z.astype(int)
print("Z.astype", X)

ar = np.random.randint(0,5,25).reshape(5,5)
print(ar)

def create_int10():
    ar = np.random.randint(0,10,10)
    return ar
Z = create_int10()
print(Z)

def generate_10integer():
    for i in range(10):
        yield i
Z = np.fromiter(generate_10integer(), dtype="float")
print("generate()", Z)


#39
ar = np.random.rand(10)
print(ar)
ar = np.linspace(0,1,12, endpoint=True)[1:-1]
print(ar)

'''

##############################################################
#

#40
Z = np.random.random(10)
Z.sort()
print("40 sort\n", Z)

#41
Z = np.arange(10)
print(Z)
ret = np.add.reduce(Z)
print("41 faster add=", ret)

#42
#ar1 = np.array([1,2,3,4])
#ar2 = np.array([1,2,3,4])
ar1 = np.random.randint(0,2,3)
ar2 = np.random.randint(0,2,3)
print(ar1)
print(ar2)
ret = np.array_equal(ar1, ar2)
print("42 two arrays = ", ret)

#43
ar = np.array([1,2,3,4])
ar.flags.writeable = False
print("43 read-only ", ar)
#ar[0] = 10

#44
Z = np.random.random((10,2))
print(Z)
x, y = Z[:,0], Z[:,1]
r = np.sqrt(x**2+y**2)
t = np.arctan2(y, x)
print("=== 44 ===")
print("r=", r)
print("t=", t)

#45
print("=== 45 ===")
ar = np.random.random(10)
print(ar)
idx = np.argmax(ar)
print(idx)
ar[idx]=0
print("max 0 = ", ar)

#46
print("=== 46 ===")
Z = np.zeros((5,5), [("i",float),("j",float)])
xv = np.linspace(0, 1, 5)
yv = np.linspace(0, 1, 5)
Z["i"], Z["j"] = np.meshgrid(xv, yv)
print(Z)

#47
#x = np.random.randint(0,5,3)
#y = np.random.randint(0,5,3)
#x = np.arange(3)
#y = np.arange(3)
x = np.array([5,6,7])
y = np.array([1,2,3])
print(x)
print(y)
Z = np.subtract.outer(x, y)
print(Z)
C = 1/Z
print(C)

#48
max = np.iinfo(np.int64).max
min = np.iinfo(np.int64).min
print(max, min)
max = np.finfo(np.float64).max
min = np.finfo(np.float64).min
print(max, min)

#49
print("===49===")
#np.set_printoptions(threshold=5)
#Z = np.arange(1000)
#print(Z)
np.set_printoptions(threshold=float("inf"))
Z = np.arange(1000)
print(Z)

#50
print("===50===")
ar = np.arange(100)
v = np.random.uniform(0,100)
print(ar)
print(v)
sub = np.abs(ar-v)
index = sub.argmin()
print(index)
print(ar[index])





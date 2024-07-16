import numpy as np
from numpy._typing import NDArray

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


#51
Z = np.zeros(10, [ ("position", [("x",float, 1), ("y",float,1)]),
                   ("color", [("r",float, 1), ("g",float,1), ("b",float,1)] )
                  ])
print(Z)

#52
Z = np.random.randint(0,3,20).reshape(10,2)
print(Z)
X, Y = np.atleast_2d(Z[:,0], Z[:,1])
print(X)
print(Y)
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

#53
#X = np.arange(0, 10, 1, dtype=np.float32)
X = 100*np.random.rand(10).astype(np.float32)
print(X)
Y = X.astype(np.int32)
print(Y)

#54
Z = np.genfromtxt("csvdata.txt",
                    skip_header=1, usecols=[1, 2],
                    missing_values="NA",
                    delimiter=",")
print(Z)


#############################################################
#

#55
Z = np.arange(9).reshape(3,3)
print(Z)
for position, val in np.ndenumerate(Z):
  print(position, val)

#56
def normal(x, mu, sigma):
      ret = np.exp( - ((x - mu)**2) / (2 * (sigma**2)) ) / np.sqrt(2 * np.pi * (sigma**2) )
      return ret
    
X = np.linspace(-5,5,100)
mu = 0
sigma = 1
Y = normal(X, mu, sigma)

print(Y)

'''
'''
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.plot(X, Y)
plt.show()
'''
'''
#57
Z = np.zeros(100)
print(Z)

p = 3
idx = np.random.choice(100, p, replace=False)
print(idx)

#np.put(Z, [1,2,3], 99)
np.put(Z, idx, 99)
print(Z)

#58
row = 3
Z = np.arange(9).reshape(row,3)
print(Z)
'''
'''
for i in range(row):
      print(Z[i, :])
      mu = np.mean(Z[i, :])
      print("mean=", mu)
X = Z.mean(axis=1, keepdims=True)
print(X)
'''
'''
Y = Z - Z.mean(axis=1, keepdims=True)
print(Y)      

#59
Z = np.random.randint(0,10,(3,3))
print(Z)
X = np.sort(Z, axis=0)
print(X)
Y = np.sort(Z, axis=1)
print(Y)
print(Z[Z[:,1].argsort()])

#60
Z = np.random.randint(0,3,(3,3))
print(Z)
a = ~Z.any(axis=0)
print(a)
b = a.any()
print(b)
print((~Z.any(axis=0)).any())

Z = np.array([
    [1,2,np.nan],
    [2,3,np.nan],
    [4,5,np.nan]
])
print(Z)
ret = np.isnan(Z).all(axis=0)
print(ret)

#61
Z = np.random.randint(0,100,10)
print(Z)
val = 51
diff=100000
nearval = 0
nd = np.nditer(Z, flags=['c_index'])
while not nd.finished:
    print(Z[nd.index])
    tmpdiff = abs(Z[nd.index]-val)
    if diff > tmpdiff:
        nearval = Z[nd.index]
        diff = tmpdiff
    nd.iternext()
print("nearest val=", nearval)

val=51
nearval = Z.flat[ np.abs(Z-val).argmin() ]
print("nearest val=", nearval)
'''
'''
#62
A = np.random.randint(0,3,(3,1))
B = np.random.randint(0,3,(1,3))
print(A)
print(B)

'''
'''
na = np.nditer(A, flags=['c_index'])
val = 0
z = np.zeros((3,3),dtype=np.uint)
while not na.finished:
    nb = np.nditer(B, flags=['c_index'])
    while not nb.finished:
        z[na.index][nb.index] = A[na.index] + B[0][nb.index]
        nb.iternext()
    na.iternext()
print("use iter\n", z)
'''
'''
it = np.nditer([A,B,None])
for x,y,z in it: 
    z[...] = x + y
print(it.operands[2])

#63
class MyArray():
    def __init__(self, argarray, argname="nothing"):
        self.inarray = argarray
        self.name = argname
        print(self.inarray)

    def getName(self):
        return self.name

ar = np.random.randint(0,10,10)
myarray = MyArray(ar, "mikearray")
print("attr in MyArray = ", myarray.getName())

class NamedArray(np.ndarray):
    def __new__(cls, array, name="nothing"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', "nothing")
Z = NamedArray(np.arange(10), "range_10")
print(Z.name)

#64
ar = np.arange(10)
br = np.array([1,3,5])
print(ar)
print(br)
np.add.at(ar,br,1)
print(ar)

#65
I = np.array([1,3,0,3,4,1])
X = np.array([1,2,3,4,5,6])
F = np.zeros(6)
for i in I:
    F[i] += X[i]
print(F)

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)


#66
w, h = 2, 2#10, 10 # 256, 256
I = np.random.randint(0,3,(w,h,3)).astype(np.ubyte)
print(I)
c = np.unique(I.reshape(-1,3), axis=0)
print(c)
n = len(c)
print(n)

#67
Z = np.random.randint(0,10,(3,4,3,4))
print(Z)
sum = Z.sum(axis=(-2,-1))
print(sum)

s = np.sum([[0,1],[5,6]], axis=0)
print(s)
s = np.sum([[0,1],[5,6]], axis=1)
print(s)

#68
D = np.random.randint(0,100,10)
#D = np.random.uniform(0,1,10)
S = np.random.randint(0,10,10)
print(D)
print(S)
D_sums = np.bincount(S, weights=D)
print(D_sums)
D_counts = np.bincount(S)
print(D_counts)
D_means = D_sums / D_counts
print(D_means)

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

#69
A = np.random.randint(0,3,(2,2))
print(A)
B = np.random.randint(0,3,(2,2))
print(B)
Z = np.dot(A, B)
print(Z)
d = np.diag(Z)
print(d)

#70
X = np.array([1,2,3,4,5])
print(X)
length = len(X)
Y = np.zeros(length+3*(length-1))
Y[::4] = X
print(Y)

#71
A = np.random.randint(0,10,(5,5,3))
B = np.random.randint(0,10,(5,5))
print(A)
print(B)
C = B[:,:, None]
print(C)
D = A*C
print(D)

A = np.ones((5,5,3))
B = 2*np.ones((5,5))
C = B[:,:, None]
D = A*C
print(D)

# 72
A = np.random.randint(0,10,(3,3))
print(A)
A[[0,1]] = A[[1,0]]
print(A)

A = np.arange(25).reshape(5,5)
print(A)
A[[0,1]] = A[[1,0]]
print(A)

#73
T1 = np.array([0,1,1,5,10,10,20,30,30])
print(T1)
T2 = np.unique(T1)
print(T2)
T3 = np.roll(T2,3)
print(T3)

A = np.random.randint(0,10,(10,3))
print(A)

B=A.repeat(2,axis=1)
print(B)

C = np.roll(B, -1, axis=1)
print(C)

D = C.reshape(len(C)*3,2)
print(D)

F = np.sort(D,axis=1)
print(F)

G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
print(G)

H = np.unique(G)
print(H)

# 74
A = np.array([1,1,2,3,4,4,6])
print(A)

C = np.bincount(A)
print(C)

D = np.arange(len(C))
print(D)

F = np.repeat(D, C)
print(F)

#75
#a = np.array([[1,2,3],[4,5,6]])
n = 3
a = np.array([1,2,3,4,5,6,7,8,9,10])
A = np.cumsum(a)
print(A)
#A[n:] = A[n:] - A[:-n]
print(A[n:])
print(A[:-n])
A[n:] = A[n:] - A[:-n]
print(A)
C = A[n-1:] / n
print(C)

#76
from numpy.lib.stride_tricks import sliding_window_view

Z = np.arange(10)
X = sliding_window_view(Z, window_shape=3)
print(X)


#77
t = np.logical_not([True, False, 0, 1])
print(t)
t2 = np.negative([1,-1])
print(t2)

Z = np.random.rand(5)
X = np.negative(Z)
print(X)

Z = np.logical_not([True, False])
print(Z)

#78
def calc_distance(p0, p1, p):
    mid = np.array([ (p1[0]-p0[0])/2, (p1[1]-p0[1])/2 ] )
    d = np.sqrt( (mid[0]-p[0])**2 + (mid[1]-p[1])**2 )
    return d 

P0 = np.array([1,2])
P1 = np.array([3,4])
p = np.array([1,0])
dis = calc_distance(P0,P1,p)
print(dis)

def calc_distance_array(P0,P1,p):
    dis = []
    for i in range(len(P0)):
        d = calc_distance(P0[i],P1[i],p)
        dis = np.append(dis,d)
    return dis

P0 = np.random.randint(0,3,(3,2))
P1 = np.random.randint(0,3,(3,2))
p = np.array([0,0])
print(P0)
print(P1)
d = calc_distance_array(P0,P1,p)
print(d)

#79

def calc_distance2(p0, p1, p):
    mid = np.array([ (p1[0]-p0[0])/2, (p1[1]-p0[1])/2 ] )
    d = np.sqrt( (mid[0]-p[0])**2 + (mid[1]-p[1])**2 )
    return d 

def calc_distance_array2(P0,P1,PX):
    dis = []
    for i in range(len(P0)):
        d = calc_distance2(P0[i],P1[i],PX[i])
        dis = np.append(dis,d)
    return dis

P0 = np.random.randint(0,3,(3,2))
P1 = np.random.randint(0,3,(3,2))
PX = np.random.randint(0,3,(3,2))
d = calc_distance_array2(P0,P1,PX)
print(d)

#80

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (2,2)
print(Z)

R = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        R[i][j] = Z[i+position[0]][j+position[1]]

print(R)

#81
from numpy.lib.stride_tricks import sliding_window_view
from numpy.lib.stride_tricks import as_strided

Z = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
print(Z)

R = as_strided(Z, shape=(11,4), strides=(8,8))
print(R)


#82
A = np.array([[1,1,4,0,1],
              [0,3,1,3,2],
              [1,3,0,0,1],
              [2,4,3,1,1]
              ])
r = np.linalg.matrix_rank(A)
print(r)

B = np.array([[1,2],[3,4]])
print(B)
U, s, V = np.linalg.svd(B, full_matrices=True)
print(U)
print(np.diag(s))
print(V)

BB = np.dot( np.dot(U, np.diag(s)) , V)
print(BB)

#83
X = np.array([1,2,3,3,3,4,5,6])
print(X)
Y = np.bincount(X)
print(Y)
print("max=",Y.argmax())
m = np.argmax(Y)
print(m)


#84
from numpy.lib.stride_tricks import sliding_window_view

X = np.random.randint(0,10,(10,10))
print(X)
Y = sliding_window_view(X, (3,3))
print(Y)

#85

class Symetric():
    def symetric(self, Z):
        X = Z + Z.T - np.diag(Z.diagonal())
        print(X)

A = np.arange(9).reshape((3,3))
print(A)
print(np.diag(A.diagonal()))
s = Symetric()
s.symetric(A)

#86
print("---------")
n=3
p=2
X = np.ones((p,n,n))
Y = np.ones((p,n,1))
print(X)
print(Y)
Z = np.tensordot(X,Y,axes=[[0,2],[0,1]])
print("axes=02 01")
print(Z)

# axes=0: axb  tensor product
# axes=1: a*b  dot product(naiseki)
# axes=2: a:b  double contraction(2jyu syukuyaku) default
print("---------")
a = np.arange(1,9).reshape(2,2,2)
print(a)
A = np.array(['a','b','c','d'],dtype=object).reshape(2,2)
print(A)
Z = np.tensordot(a,A) #axes=2
print("axes=2")
print(Z)
X = np.tensordot(a,A,1) #axes=1
print("axes=1")
print(X)
Y = np.tensordot(a,A,0) #axes=0
print("axes=0")
print(Y)
# a [0,2] -> aa
# A [0,1] -> AA
# XX = aa : AA
XX = np.tensordot(a,A,axes=[[0,2],[0,1]])
print("axes=02 01")
print(XX)

#0201
#['abbcccccdddddd' 'aaabbbbcccccccdddddddd']
#0101
#['abbbcccccddddddd' 'aabbbbccccccdddddddd']
#1001
#['abbbbbcccddddddd' 'aabbbbbbccccdddddddd']

#87
from numpy.lib.stride_tricks import sliding_window_view

#k = 2
#n = 4
k = 4
n = 16
A = np.arange(n*n).reshape((n,n))
print(A)

Y = sliding_window_view(A, (k,k))
print(Y)
print("block sum =")
S = Y[::k, ::k, ...].sum(axis=(-2, -1))
print(S)

'''
#### 88. How to implement the Game of Life using numpy arrays? (★★★)
#`No hints provided...`

#セルの生死:
#セルは「生」（1）または「死」（0）の状態を持ちます。
#セルの次の世代の状態は、周囲の8つのセルの現在の状態によって決まります。
#ルール:
#誕生: 死んでいるセルに隣接する生きたセルがちょうど3つあれば、次の世代が誕生します。
#生存: 生きているセルに隣接する生きたセルが2つか3つならば、次の世代でも生存します。
#過疎: 生きているセルに隣接する生きたセルが1つ以下ならば、過疎により死滅します。
#過密: 生きているセルに隣接する生きたセルが4つ以上ならば、過密により死滅します。
'''
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)

#89
n = 3
Z = np.array([2,3,5,1,0,9,4])
print(Z)
X = np.argsort(Z)
print(X)
print(Z[X])
A = X[-n:]
print(A)
B = Z[A]
print(B)

#90
X = np.arange(9).reshape(3,3)
print(X)
Z = np.indices(X.shape)
print(Z)
#print(Z[0])
#print(Z[1])

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)
    print(arrays)

    ix = np.indices(shape, dtype=int)
    print(ix)
    ix = ix.reshape(len(arrays),-1).T
    print(ix)

    for n, arr in enumerate(arrays):
        AR = arrays[n][ix[:,n]]
        print("AR=",AR)
        ix[:,n] = AR
    
    return ix
 
#ret = cartesian(([1,2,3],[4,5],[6,7]))
ret = cartesian(([1,2,3],[4,5,6]))
print("---90---")
print(ret)

'''

#91
x1 = np.array([1,2,3,4])
x2 = np.array(["mike","torii","john","kerry"])
x3 = np.array([4.0, 2.1, 2.3, 5.1])
r = np.core.records.fromarrays([x1,x2,x3], dtype=np.dtype([("a", np.int32),("b", "S5"),("c", np.float32)]))
print(r)

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)
#`hint: np.power, *, np.einsum`

Z = np.array([1,2,3,4])
print(Z)
m1 = np.power(Z,3)
m2 = Z**3
m3 = np.einsum("i,i,i->i",Z,Z,Z)
print(m1)
print(m2)
print(m3)

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)
#`hint: np.where`
#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
#`No hints provided...`
#### 95. Convert a vector of ints into a matrix binary representation (★★★)
#`hint: np.unpackbits`
#### 96. Given a two dimensional array, how to extract unique rows? (★★★)
#`hint: np.ascontiguousarray | np.unique`
#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)
#`hint: np.einsum`
#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?
#`hint: np.cumsum, np.interp`
#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
#`hint: np.logical_and.reduce, np.mod`
#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
#`hint: np.percentile`
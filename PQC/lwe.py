import math
import numpy as np

def uniform_distribution(q, n):
    return np.random.randint(0, q, size=n)

def normal_distribution(n):
    a = np.random.normal(0, sigma, size=n)
    return np.round(a).astype(int)

def Signal(x):
    if balanced(x) in range(-math.floor(q/4), round(q/4)+1):
        return 0
    else:
        return 1

Signal = np.frompyfunc(Signal, 1, 1)

def Encode(x, s):
    return balanced( (x + s*(q-1)//2 ) % q) % 2

def balanced(x):
    if 0 <= x <= q//2:
        return x
    else:
        return x-q

n = 1024
q = 40961
sigma = 8 / math.sqrt(2*math.pi)

def main():
    print("---start---")
    
    M = uniform_distribution(q, (n,n))
    print(M)

    # Alice
    sA = np.matrix(normal_distribution(n)).transpose()
    eA = np.matrix(normal_distribution(n)).transpose()    
    pA = np.matrix( (M.dot(sA) + 2*eA ) % q )
    print("---Alice---")
    print("sA=", sA)
    print("eA=", eA)
    print("pA=", pA)
    
    # Bob
    sB = np.matrix(normal_distribution(n)).transpose()
    eB = np.matrix(normal_distribution(n)).transpose()    
    pB = np.matrix( (M.transpose().dot(sB) + 2*eB ) % q )
    print("---Bob---")
    print("sB=", sB)
    print("eB=", eB)
    print("pB=", pB)

    # Bob key
    eB_prime = normal_distribution(1)
    kB = ( (pA.transpose()).dot(sB) + 2*eB_prime ) % q
    print("kB=", kB)
    s = Signal(kB).astype(int)
    skB = Encode(kB, s)
    print("skB=", skB)

    # Alice key
    eA_prime = normal_distribution(1)
    kA = ( (sA.transpose()).dot(pB) + 2*eA_prime ) % q
    print("kA=", kA)
    skA = Encode(kA, s)
    print("skA=", skA)    

    is_same = (skA.tolist() == skB.tolist())
    print("skA == skB: ", is_same)
    
    print("---end---")    
    
if __name__ == "__main__":
    main()
    

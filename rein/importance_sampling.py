import numpy as np

x = np.array([1,2,3])
pi = np.array([0.1,0.1,0.8])

########################
#
e = np.sum(x*pi)
print('E_pi[x]', e)

#######################
# Monte Carlo
#
n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)    
    #print(s)
    samples.append(s)
print('MC: {:.2f} (var: {:.2f})'.format(np.mean(samples), np.var(samples)))

######################
#
b = np.array([0.2,0.2,0.6])
samples = []
for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i]/b[i]
    samples.append(rho*s)
print('IS: {:.2f} (var: {:.2f})'.format(np.mean(samples), np.var(samples)))

    





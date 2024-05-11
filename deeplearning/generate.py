import os
import numpy as np
import matplotlib.pyplot as plt

def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** d * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y

def draw(original_xs, new_xs):
    plt.scatter(original_xs[:,0], original_xs[:,1], alpha=0.7, label="original")
    plt.scatter(new_xs[:,0], new_xs[:,1], alpha=0.7, label="generated")

    plt.xlabel("Eruptions(Min)")
    plt.ylabel("Waiting(Min)")
    plt.show()

if __name__ == "__main__":
    original_xs = np.loadtxt("old_faithful.txt")

    #
    mus = np.array([[2.0, 54.50], 
                    [4.3, 80.0]])

    covs = np.array([ [ [0.07, 0.44], 
                        [0.44, 33.7] ],
                      [ [0.17, 0.94], 
                        [0.94, 36.00] ] ])

    phis = np.array([0.35, 0.65])

    print(mus)
    print(covs)
    print(phis)

    N = 500

    new_xs = np.zeros((N, 2))
    for n in range(N):
        k = np.random.choice(2, p=phis)
        mu, cov = mus[k], covs[k]
        new_xs[n] = np.random.multivariate_normal(mu, cov)

    draw(original_xs, new_xs)    

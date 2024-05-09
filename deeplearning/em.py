import os
import numpy as np
import matplotlib.pyplot as plt

def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp( (x-mu).T @ inv @ (x-mu) / -2.0 )
    return y

def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y

def likelihood(xs, phis, mus, covs):
    eps = 1e-8
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y+eps)
    return L/N

def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i,j], Y[i,j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i,j] += w[k] * multivariate_normal(x, mu, cov)
    
    plt.contour(X,Y,Z)

def draw(xs, phis, mus, covs):
    plt.scatter(xs[:,0], xs[:,1])
    plot_contour(phis, mus, covs)
    plt.xlabel("Eruptions(Min)")
    plt.ylabel("Waiting(Min)")
    plt.show()

if __name__ == "__main__":
    print("--start--")
    path = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
    xs = np.loadtxt(path)
    print(xs.shape)

    phis = np.array([0.5,0.5])
    mus = np.array([[0.0, 500],[00, 100.0]])
    covs = np.array([np.eye(2), np.eye(2)])

    K = len(phis)
    N = len(xs)
    MAX_ITERS = 100
    THESHOLD = 1e-4

    print(K, N)

    current_likelihood = likelihood(xs, phis, mus, covs)
    print(current_likelihood)

    draw(xs, phis, mus, covs)



#    draw(X,Y,Z)

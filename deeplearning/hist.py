import os
import numpy as np
import matplotlib.pyplot as plt

def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x-mu)**2 /(2* sigma**2))
    return y

if __name__=="__main__":
    path = os.path.join(os.path.dirname(__file__), 'height.txt')
    xs = np.loadtxt(path)
    print(xs.shape)

    mu = np.mean(xs)
    sigma = np.std(xs)
    print("mu=", mu, "sigma=", sigma)

    x = np.linspace(150, 190, 1000)
    y = normal(x, mu, sigma)

    plt.hist(xs, bins='auto', density=True)
    plt.plot(x, y)
    plt.xlabel('Height(cm)')
    plt.ylabel('Probability Density')
    plt.show()




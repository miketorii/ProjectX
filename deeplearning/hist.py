import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x-mu)**2 /(2* sigma**2))
    return y

def calcprob(mu=0, sigma=1):
    p1 = norm.cdf(160, mu, sigma)
    print('probability(x<=160)=', p1)

    p2 = norm.cdf(180, mu, sigma)
    print('probability(x>=180)=', 1-p2)

if __name__=="__main__":
    ### load sample data
    path = os.path.join(os.path.dirname(__file__), 'height.txt')
    xs = np.loadtxt(path)
    print(xs.shape)

    ### calculate mu and sigma
    mu = np.mean(xs)
    sigma = np.std(xs)
    print("mu=", mu, "sigma=", sigma)

    ### create data using mu and sigma
    sample = np.random.normal(mu, sigma)
    print("sample=", sample)

    samples = np.random.normal(mu, sigma, 10000)

    ### caluculate probability with CDF
    calcprob(mu, sigma)

    ### create data using model (in this case normal distribution)
    x = np.linspace(150, 190, 1000)
    y = normal(x, mu, sigma)

    ### draw
    plt.hist(xs, bins='auto', density=True, alpha=0.5, label='original')
    plt.plot(x, y)
    plt.hist(samples, bins='auto', density=True, alpha=0.5, label='generated')
    plt.xlabel('Height(cm)')
    plt.ylabel('Probability Density')
    plt.show()




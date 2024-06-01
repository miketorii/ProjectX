import os
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    path = os.path.join(os.path.dirname(__file__), 'height_weight.txt')
    xs = np.loadtxt(path)

    print(xs.shape)

    return xs

def draw_scatter(small_xs):
    plt.scatter(small_xs[:,0], small_xs[:,1])
    plt.xlabel("Height(cm)")
    plt.ylabel("Weight(kg)")
    plt.show()

def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp( (x-mu).T @ inv @ (x-mu) / -2.0 )
    return y

def create_data(mu, cov):   
    X, Y = np.meshgrid( np.arange(150, 195, 0.5),
                        np.arange(45, 75, 0.5) )
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([ X[i,j], Y[i,j] ])
            Z[i,j] = multivariate_normal(x, mu, cov)

    print(Z)

    return X, Y, Z

def draw(X,Y,Z):
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.plot_surface(X,Y,Z, cmap='viridis')

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.contour(X,Y,Z)

    plt.show()


if __name__ == "__main__":
    xs = load_data()

    small_xs = xs[:500]
    draw_scatter(small_xs)

    mu = np.mean(xs, axis=0)
    cov = np.cov(xs, rowvar=False)

    print(mu, cov)

    X, Y, Z = create_data(mu, cov)

    draw(X, Y, Z)





import numpy as np


def gen_kura_data(params):
    num = params.num_samples
    z1 = np.random.uniform(low=-1, high=+1, size=(num))
    z2 = np.random.uniform(low=-1, high=+1, size=(num))

    X = np.empty(shape=(num, 3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = 0.5 * (z1**2 - z2**2)
    return X


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    X = gen_data(num_samples=100)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.show()

    X = gen_hierarichical_data(30)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x in X:
        print(X.shape, x.shape)
        ax.scatter(x[:, 0], x[:, 1])
    plt.show()

    X = gen_multi_kura()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x in X:
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    plt.show()

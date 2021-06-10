from matplotlib.pyplot import figure
import numpy as np


def gen_saddle_shape(num_samples, random_seed=None, noise_scale=0.05):
    np.random.seed(random_seed)
    z1 = np.random.uniform(low=-1, high=+1, size=(num_samples,))
    z2 = np.random.uniform(low=-1, high=+1, size=(num_samples,))

    X = np.empty((num_samples, 3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = 0.5 * (z1**2 - z2**2)
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)

    return X


def gen_2d_sin_curve(num_samples, random_seed=None, noise_scale=0.01):
    np.random.seed(random_seed)
    z = np.random.uniform(low=-1, high=+1, size=(num_samples,))

    X = np.empty((num_samples, 2))
    X[:, 0] = z
    X[:, 1] = np.sin(z * np.pi) * 0.5
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)

    return X


if __name__ == '__main__':
    from matplotlib import pyplot as plt


    X = gen_saddle_shape(200, random_seed=0, noise_scale=0.05)
    # X = gen_2d_sin_curve(100, random_seed=0, noise_scale=0.01)

    _, D = X.shape
    projection = '3d' if D >= 3 else 'rectilinear'

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=projection)

    xs = (X[:, 0], X[:, 1], X[:, 2]) if D >= 3 else (X[:, 0], X[:, 1])
    ax.scatter(*xs)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlabel("x3")

    plt.show()

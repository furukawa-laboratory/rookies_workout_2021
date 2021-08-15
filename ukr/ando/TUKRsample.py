import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")


def E(Z: tf.Variable, X: np.ndarray) -> tf.Variable:
    Dzz = Z[:, None, :] - Z[None, :, :]
    D = tf.reduce_sum(tf.square(Dzz), axis=2)
    H = tf.exp(-0.5 * D)
    G = tf.reduce_sum(H, axis=1, keepdims=True)
    R = H / G
    Y = R @ X
    result = 0.5 * tf.reduce_sum((Y - X)**2)
    return result


def fit(Z1: np.ndarray, Z2: np.ndarray, X: np.ndarray, index1: np.ndarray, index2: np.ndarray, n_epoch: int, eta: float):
    N1, L1 = Z1.shape
    N2, L2 = Z2.shape
    tZ = tf.Variable(np.concatenate((Z1[index1, :], Z2[index2, :]), axis=1))
    count1 = np.bincount(index1, minlength=N1)
    count2 = np.bincount(index2, minlength=N2)

    optimizer = tf.keras.optimizers.SGD(learning_rate=eta)
    for epoch in range(n_epoch):
        with tf.GradientTape() as tape:
            result = E(tZ, X)
        grad = tape.gradient(result, tZ)
        grad1 = tf.concat([tf.math.bincount(index1, grad[:, i], minlength=N1)[:, None] for i in range(L1)], axis=1) / count1[:, None]
        grad2 = tf.concat([tf.math.bincount(index2, grad[:, i+L1], minlength=N2)[:, None] for i in range(L2)], axis=1) / count2[:, None]
        grad = tf.concat([tf.gather(grad1, index1), tf.gather(grad2, index2)], axis=1)
        optimizer.apply_gradients([(grad, tZ)])

    _, index1_inverse = np.unique(index1, return_index=True)
    _, index2_inverse = np.unique(index2, return_index=True)
    Z1 = tZ.numpy()[index1_inverse, :L1]
    Z2 = tZ.numpy()[index2_inverse, L1:]
    return Z1, Z2


if __name__ == '__main__':
    import itertools
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    N1 = 20
    N2 = 30
    D = 3
    L = 1  # This will work in more than two dimensions
    seed = 100
    observation_rate = 0.5
    n_epoch = 100
    eta = 0.1
    np.random.seed(seed)

    # create true latent variable
    oZ1 = np.random.uniform(-1, 1, (N1, L))
    oZ2 = np.random.uniform(-1, 1, (N2, L))
    oZ = np.array(list(itertools.product(oZ1, oZ2))).reshape((N1, N2, -1))

    # create full data
    X_full = np.zeros((N1, N2, D))
    X_full[:, :, 0] = oZ[:, :, 0]
    X_full[:, :, 1] = oZ[:, :, L]
    X_full[:, :, 2] = oZ[:, :, 0] ** 2 - oZ[:, :, L] ** 2

    # create observed data
    Gamma = np.random.binomial(1, observation_rate, (N1, N2))
    index1, index2 = Gamma.nonzero()
    X = X_full[index1, index2, :]

    # init latent variable
    eZ1 = np.random.normal(0.0, 0.1, (N1, L))
    eZ2 = np.random.normal(0.0, 0.1, (N2, L))

    # learn
    eZ1, eZ2 = fit(eZ1, eZ2, X, index1, index2, n_epoch=n_epoch, eta=eta)
    eZ = np.array(list(itertools.product(eZ1, eZ2)))

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    # ax1.scatter(X_full[:, :, 0], X_full[:, :, 1], X_full[:, :, 2])
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax2 = fig.add_subplot(122, aspect='equal')
    ax2.scatter(eZ[:, 0], eZ[:, L])
    plt.show()
    
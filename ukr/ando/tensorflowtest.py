import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")

#目的関数の値の推定
def E(Z: tf.Variable, X: np.ndarray) -> tf.Variable:
    Dzz = Z[:, None, :] - Z[None, :, :]
    D = tf.reduce_sum(tf.square(Dzz), axis=2)
    H = tf.exp(-0.5 * D)
    G = tf.reduce_sum(H, axis=1, keepdims=True)
    R = H / G
    Y = R @ X
    result = 0.5 * tf.reduce_sum((Y - X)**2)
    return result


def fit(Z: np.ndarray, X: np.ndarray, n_epoch: int, alpha: float) -> np.ndarray:
    tZ = tf.Variable(Z)

    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha)
    for epoch in range(n_epoch):
        with tf.GradientTape() as tape:
            result = E(tZ, X)
        grad = tape.gradient(result, tZ)
        optimizer.apply_gradients([(grad, tZ)])

    return tZ.numpy()
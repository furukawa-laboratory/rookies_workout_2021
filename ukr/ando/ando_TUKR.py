import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import make_grid
tf.keras.backend.set_floatx("float64")

#写像の推定
def estimate_Y(Z1: tf.Variable, Z2: tf.Variable, X: np.ndarray, sigma: float) -> tf.Variable:
    #ドメイン１のデータの距離の計算
    distance_N = Z1[:, None, :] - Z1[None, :, :]
    dist_N = tf.reduce_sum(tf.square(distance_N), axis=2) 
    #ドメイン２のデータの距離
    distance_M = Z2[:, None, :] - Z2[None, :, :]
    dist_M = tf.reduce_sum(tf.square(distance_M), axis=2) 
    #写像の推定
    k_N = tf.exp(-1/(2*(sigma**2))*dist_N)
    k_M = tf.exp(-1/(2*(sigma**2))*dist_M)
    k = k_N*k_M
    K_N = tf.reduce_sum(k_N, axis=1, keepdims=True)
    K_M = tf.reduce_sum(k_M, axis=1, keepdims=True)
    K = K_N*K_M
    Y = (k@X)/K
    result = 0.5 * tf.reduce_sum((Y - X)**2)
    return result

#潜在変数の推定
def estimate_Z(Z1: np.ndarray, Z2: np.ndarray, X: np.ndarray, T: int, eta: float) -> np.ndarray:
    tZ1 = tf.Variable(Z1)
    tZ2 = tf.Variable(Z2)

    optimizer = tf.keras.optimizers.SGD(learning_rate=eta)
    for t in range(T):
        with tf.GradientTape() as tape:
            result = estimate_Y(tZ1, tZ2, X)
        grad = tape.gradient(result, tZ1, tZ2)#ここで微分
        optimizer.apply_gradients([(grad, tZ1, tZ2)])
    Z1 = tZ1.numpy()
    Z2 = tZ2.numpy()

    return Z1,Z2
#学習用の関数
def fit():
    #写像の推定の部分の準備
    #誤差関数の計算
    #自動微分
    #潜在変数の更新


if __name__ == '__main__':
    import dataTUKR

    N1 = 30 #ドメイン１のデータ数
    N2 = 20 #ドメイン２のデータ数
    D = 3 #データ一つあたりの次元数
    L = 2 #潜在空間の次元数
    seed = 0
    T = 100 #学習回数
    eta = 0.1 #勾配法で用いるステップ幅
    sigma = 0.1 #カーネル関数で使う.
    np.random.seed(seed)

    #人口データ
    X = dataTUKR.load_kura_tsom(N1, N2, retz=False)



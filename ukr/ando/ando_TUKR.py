import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import make_grid
tf.keras.backend.set_floatx("float64")

#目的関数の値の用意
def E(Z1: tf.Variable, Z2: tf.Variable, X: np.ndarray, sigma: float) -> tf.Variable:
    #ドメイン１のデータの距離の計算
    distance_N1 = Z1[:, None, :] - Z1[None, :, :]
    dist_N1 = tf.reduce_sum(tf.square(distance_N1), axis=2) 
    #ドメイン２のデータの距離
    distance_N2 = Z2[:, None, :] - Z2[None, :, :]
    dist_N2 = tf.reduce_sum(tf.square(distance_N2), axis=2) 
    #
    k_N1 = tf.exp(-1/(2*(sigma**2))*dist_N1)
    k_N2 = tf.exp(-1/(2*(sigma**2))*dist_N2)
    h1 = k_N1[:, :, None, None]*k_N2[None, None, :, :]
    K = tf.reduce_sum(h1,axis=(1,3),keepdims=True)
    h2 = k_N1[:, :, None, None, None]*k_N2[None, None, :, :, None]*X[None, :, None, :, :]
    k= tf.reduce_sum(h2,axis=(1,3),keepdims=True)
    Y = k/K[:, :, None]
    result = 0.5 * tf.reduce_sum((Y - X)**2)
    return result

#目的関数の微分
def fit(Z1: np.ndarray, Z2: np.ndarray, X: np.ndarray, T: int, eta: float) -> np.ndarray:
    tZ1 = tf.Variable(Z1)
    tZ2 = tf.Variable(Z2)

    optimizer = tf.keras.optimizers.SGD(learning_rate=eta)
    for t in range(T):
        with tf.GradientTape() as tape:
            result = E(tZ1, tZ2, X)
        grad1 = tape.gradient(result, tZ1)#ここで微分
        grad2 = tape.gradient(result, tZ2)
        optimizer.apply_gradients([(grad1, tZ1),(grad2, tZ2)])
    Z1 = tZ1.numpy()
    Z2 = tZ2.numpy()

    return Z1,Z2
#学習用の関数
#def fit(Z1: np.nadarry, Z2: np.ndarray, X: np.ndarray,):
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



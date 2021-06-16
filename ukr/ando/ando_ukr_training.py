#拡張機能のインポート
import numpy as np
import matplotlib.pyplot as plt

#各変数の初期設定
N = 200 #データ数
D = 2   #データの次元数
L = 3   #潜在空間の次元数
T = 100 #総学習回数(epoch数のこと)
#観測データの生成の際に用いる変数
seed = 0
noise_scale = 0.05
np.random.seed(seed)

#使用する観測データ
def gen_saddle_shape(num, seed, noise_scale):
    np.random.seed(seed)
    z1 = np.random.uniform(low=-1, high=+1, size=(num,))
    z2 = np.random.uniform(low=-1, high=+1, size=(num,))

    X = np.empty((num, 3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = 0.5 * (z1**2 - z2**2)
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)

    return X

#観測データの描画(データの確認)
X = gen_saddle_shape(N, seed, noise_scale)
print("Xのシェイプ",X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') #三次元マップの描画
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.show()

#----------------------------ここからアルゴリズム部----------------------------

#初期化、潜在変数zを乱数によって初期化
Z = np.random.rand(N, L)

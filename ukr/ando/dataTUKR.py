import numpy as np
import random
import itertools

def load_kura_tsom(xsamples, ysamples, missing_rate=None,retz=False):
    z1 = np.linspace(-1, 1, xsamples)
    z2 = np.linspace(-1, 1, ysamples)

    z1_repeated, z2_repeated = np.meshgrid(z1, z2, indexing='ij')
    x1 = z1_repeated
    x2 = z2_repeated
    x3 = (z1_repeated ** 2.0 - z2_repeated ** 2.0)+np.random.normal(loc = 0.0, scale = 0.1, size=(xsamples,ysamples)) 
    #ノイズを加えたい時はここをいじる,locがガウス分布の平均、scaleが分散,size何個ノイズを作るか
    #このノイズを加えることによって三次元空間のデータ点は上下に動く

    x = np.concatenate((x1[:, :, np.newaxis], x2[:, :, np.newaxis], x3[:, :, np.newaxis]), axis=2)
    truez = np.concatenate((z1_repeated[:, :, np.newaxis], z2_repeated[:, :, np.newaxis]), axis=2)

    #欠損値を入れない場合(missing_rateが0か特に指定していない場合はそのまま返す)
    if missing_rate == 0 or missing_rate == None:
        if retz:
            return x, truez
        else:
            return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xsamples = 20 #ドメイン１のデータ数
    ysamples = 30 #ドメイン２のデータ数

    #欠損なしver
    x, truez = load_kura_tsom(xsamples, ysamples, retz=True) #retzは真の潜在変数を返す,Xは観測データ,truezは真の潜在変数
    #Xは左側の図の作成に必要、truez右側の図の作成に必要（実行結果の図より）
    # 欠損ありver
    #x, truez, Gamma = load_kura_tsom(xsamples, ysamples, retz=True,missing_rate=0.7)

    fig = plt.figure(figsize=[10, 5])
    ax_x = fig.add_subplot(1, 2, 1, projection='3d')
    ax_truez = fig.add_subplot(1, 2, 2)
    ax_x.scatter(x[:, :, 0].flatten(), x[:, :, 1].flatten(), x[:, :, 2].flatten(), c=x[:, :, 0].flatten())
    ax_truez.scatter(truez[:, :, 0].flatten(), truez[:, :, 1].flatten(), c=x[:, :, 0].flatten())
    ax_x.set_title('Generated three-dimensional data')
    ax_truez.set_title('True two-dimensional latent variable')
    plt.show()

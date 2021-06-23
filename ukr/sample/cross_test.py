import numpy as np
from numpy.random import seed
import data
from somf import UnsupervisedKernelRegression as SomfUKR
from ukr_uniform import UKR as UniformUKR
from ukr_gaussian import UKR as GaussianUKR
from ukr_autograd import UKR as AutogradUKR
from matplotlib import pyplot as plt


SEED = 0
N = 100

# 潜在変数を一様分布で初期化するための関数
# UKR クラスの外で潜在変数初期化の処理を書いたのは
# somf の UKR が，引数で潜在変数を受け取れるような仕様になっていたため
def init_uniform_Z(params):
    scale = params['scale']
    clipping = params['clipping']
    L = params['latent_dim']

    width = (max(clipping) - min(clipping))
    mid = sum(clipping) / 2
    low, high = mid - scale*width, mid + scale* width
    np.random.seed(SEED)
    Z = np.random.uniform(low=low, high=high, size=(N, L))
    return Z

# 潜在変数をガウス分布で初期化するための関数
def init_gaussian_Z(params):
    scale = params['scale']
    L = params['latent_dim']

    np.random.seed(SEED)
    Z = np.random.normal(scale=scale, size=(N, L))
    return Z



# 一様潜在変数の UKR （ukr_uniform と somf）のクロステスト
def test_mode1(X, show_result):
    # それぞれの UKR について同じパラメータを設定する．
    # 実装の違いがあるのでそこはうまいこと合わせる（少々面倒）
    eta = 2
    num_epoch = 100
    params_for_myukr = dict(
        latent_dim=2,
        eta=eta,
        sigma=0.1,
        scale=1e-3,
        clipping=(-1, +1),
    )
    init = init_uniform_Z(params_for_myukr)
    params_for_somf = dict(
        X=X,
        n_components=params_for_myukr['latent_dim'],
        bandwidth_gaussian_kernel=params_for_myukr['sigma'],
        is_compact=True,
        lambda_=0.0,
        init=init,
        is_save_history=True,
    )

    # UKR をそれぞれ実行し，学習結果を history_* に格納
    ukr_1 = SomfUKR(**params_for_somf)
    history_1 = ukr_1.fit(nb_epoch=num_epoch, eta=eta)
    ukr_2 = UniformUKR(**params_for_myukr)
    history_2 = ukr_2.fit(X, num_epoch=num_epoch, seed=SEED, init=init)

    # numpy の allclose 関数を用いて
    # 目的関数の値，写像，潜在変数について比較
    # （正味 潜在変数が一致していれば OK）
    # 一致していれば True が返る
    is_E_close = np.allclose(history_1['obj_func'], history_2['E'])
    is_Y_close = np.allclose(history_1['y'], history_2['Y'])
    is_Z_close = np.allclose(history_1['z'], history_2['Z'])
    print("E: ", is_E_close)
    print("Y: ", is_Y_close)
    print("Z: ", is_Z_close)

    if show_result:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(num_epoch), history_1['obj_func'], label="somf")
        ax1.plot(np.arange(num_epoch), history_2['E'], label="myukr")

        ax2 = fig.add_subplot(122)
        epo =-1
        ax2.scatter(history_1['z'][epo, :, 0], history_1['z'][epo, :, 1], label="somf")
        ax2.scatter(history_2['Z'][epo, :, 0], history_2['Z'][epo, :, 1], label="myukr")
        plt.show()


# ガウス潜在変数の UKR （ukr_gaussian と somf）のクロステスト
def test_mode2(X, show_result):
    eta = 2
    num_epoch = 100
    params_for_myukr = dict(
        latent_dim=2,
        eta=eta,
        sigma=0.1,
        scale=1e-3,
        rambda=1e-04,
    )
    init = init_gaussian_Z(params_for_myukr)
    params_for_somf = dict(
        X=X,
        n_components=params_for_myukr['latent_dim'],
        bandwidth_gaussian_kernel=params_for_myukr['sigma'],
        lambda_=params_for_myukr['rambda'],
        init=init,
        is_save_history=True,
    )

    ukr_1 = SomfUKR(**params_for_somf)
    history_1 = ukr_1.fit(nb_epoch=num_epoch, eta=eta)
    ukr_2 = GaussianUKR(**params_for_myukr)
    history_2 = ukr_2.fit(X, num_epoch=num_epoch, seed=SEED, init=init)

    # 目的関数の値を計算するタイミングが違うため結果は合わない
    # is_E_close = np.allclose(history_1['obj_func'], history_2['E'])
    is_Y_close = np.allclose(history_1['y'], history_2['Y'])
    is_Z_close = np.allclose(history_1['z'], history_2['Z'])
    print("Y: ", is_Y_close)
    print("Z: ", is_Z_close)

    if show_result:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(num_epoch), history_1['obj_func'], label="somf")
        ax1.plot(np.arange(num_epoch), history_2['E'], label="myukr")

        ax2 = fig.add_subplot(122)
        epo =-1
        ax2.scatter(history_1['z'][epo, :, 0], history_1['z'][epo, :, 1], label="somf")
        ax2.scatter(history_2['Z'][epo, :, 0], history_2['Z'][epo, :, 1], label="myukr")
        plt.show()


# 手動微分版の UKR （ukr_uniform）と自動微分版の UKR（ukr_autograd）のクロステスト
def test_mode3(X, show_result):
    num_epoch = 100
    params_for_init = dict(
        latent_dim=2,
        eta=0.1,
        sigma=0.1,
        scale=1e-3,
        clipping=(-1, +1),
    )
    params_for_fit = dict(
        X=X.copy(),
        num_epoch=num_epoch,
        seed=SEED,
        f_resolution=10,
        init='random',
    )

    ukr_1 = UniformUKR(**params_for_init)
    history_1 = ukr_1.fit(**params_for_fit)
    ukr_2 = AutogradUKR(**params_for_init)
    history_2 = ukr_2.fit(**params_for_fit)

    is_E_close = np.allclose(history_1['E'], history_2['E'])
    is_Y_close = np.allclose(history_1['Y'], history_2['Y'])
    is_Z_close = np.allclose(history_1['Z'], history_2['Z'])
    print("E: ", is_E_close)
    print("Y: ", is_Y_close)
    print("Z: ", is_Z_close)

    if show_result:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(num_epoch), history_1['E'], label="hand")
        ax1.plot(np.arange(num_epoch), history_2['E'], label="auto")

        ax2 = fig.add_subplot(122)
        epo =-1
        ax2.scatter(history_1['Z'][epo, :, 0], history_1['Z'][epo, :, 1], label="hand")
        ax2.scatter(history_2['Z'][epo, :, 0], history_2['Z'][epo, :, 1], label="auto")
        plt.show()


if __name__ == '__main__':

    mode = 'all'
    show_result = False

    X = data.gen_saddle_shape(num_samples=N, random_seed=0, noise_scale=0.05)

    # mode に 1, 2, 3 以外を指定すると test_mode* 関数全てを実行する
    # この書き方を真似する必要はないです
    if mode == 1:  # somf uniform vs my uniform ukr
        test_mode1(X, show_result)
    elif mode == 2:  # somf gaussian vs my gaussian ukr
        test_mode2(X, show_result)
    elif mode == 3:
        test_mode3(X, show_result)
    else:
        for i in range(1, 4):
            eval(f"test_mode{i}")(X, show_result)

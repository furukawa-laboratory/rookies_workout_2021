import numpy as np
from numpy.random import seed
import data
from ando_ukr_training import UKR as andoUKR
from ukr_uniform import UKR as UniformUKR
from ukr_gaussian import UKR as GaussianUKR
from matplotlib import pyplot as plt


SEED = 0
if __name__ == '__main__':

    N = 100
    num_epoch = 100
    eta = 1
    X = data.gen_saddle_shape(num_samples=N, random_seed=0, noise_scale=0.05)

    params_for_gukr = dict(
        latent_dim=2,
        eta=eta,
        sigma=0.1,
        scale=1e-3,
        rambda=0,
    )

    #init = init_uniform_Z(params_for_myukr)
    params_for_andoukr = dict(
        N=N,
        D=3,
        L=2,
        eta=eta,
        sigma=0.1,
        scale=1e-3
    )

    ukr_1 = GaussianUKR(**params_for_gukr)
    history_1 = ukr_1.fit(X, num_epoch=num_epoch, seed=SEED)
    ukr_2 = andoUKR(**params_for_andoukr)
    history_2 = ukr_2.fit(X, T=num_epoch, f_reso=10, seed=SEED)

    # is_E_close = np.allclose(history_1['obj_func'][0], history_2['E'][0])
    is_Y_close = np.allclose(history_1['Y'], history_2['Y'])
    is_Z_close = np.allclose(history_1['Z'], history_2['Z'])
    # print("E: ", is_E_close)
    print("Y: ", is_Y_close)
    print("Z: ", is_Z_close)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    # ax1.plot(np.arange(num_epoch), history_1['obj_func'], label="somf")
    # ax1.plot(np.arange(num_epoch), history_2['E'], label="myukr")

    ax2 = fig.add_subplot(122)
    epo =-1
    ax2.scatter(history_1['Z'][epo, :, 0], history_1['Z'][epo, :, 1], label="somf")
    ax2.scatter(history_2['Z'][epo, :, 0], history_2['Z'][epo, :, 1], label="myukr")
    plt.show()

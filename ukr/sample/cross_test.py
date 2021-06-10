import numpy as np
from numpy.random import seed
import data
from somf import UnsupervisedKernelRegression as SomfUKR
from ukr_uniform import UKR as UniformUKR
from ukr_gaussian import UKR as GaussianUKR
from matplotlib import pyplot as plt


SEED = 0

def init_Z(params):
    scale = params['scale']
    clipping = params['clipping']
    L = params['latent_dim']

    width = (max(clipping) - min(clipping))
    mid = sum(clipping) / 2
    low, high = mid - scale*width, mid + scale* width
    np.random.seed(SEED)
    Z = np.random.uniform(low=low, high=high, size=(N, L))
    return Z


if __name__ == '__main__':

    mode = 1

    N = 100
    X = data.gen_saddle_shape(num_samples=N, random_seed=0, noise_scale=0.05)

    if mode == 1:
        eta = 2
        num_epoch = 1
        params_for_myukr = dict(
            latent_dim=2,
            eta=eta,
            sigma=0.1,
            scale=1e-3,
            clipping=(-1, +1),
        )
        init = init_Z(params_for_myukr)
        params_for_somf = dict(
            X=X,
            n_components=params_for_myukr['latent_dim'],
            bandwidth_gaussian_kernel=params_for_myukr['sigma'],
            is_compact=True,
            lambda_=0.0,
            init=init,
            is_save_history=True,
        )

        ukr_1 = SomfUKR(**params_for_somf)
        history_1 = ukr_1.fit(nb_epoch=num_epoch, eta=eta)
        ukr_2 = UniformUKR(**params_for_myukr)
        history_2 = ukr_2.fit(X, num_epoch=num_epoch, seed=SEED, init=init)

        is_E_close = np.allclose(history_1['obj_func'][0], history_2['E'][0])
        is_Y_close = np.allclose(history_1['y'][0], history_2['Y'][0])
        is_Z_close = np.allclose(history_1['z'][0], history_2['Z'][0])
        print("E: ", is_E_close)
        print("Y: ", is_Y_close)
        print("Z: ", is_Z_close)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(num_epoch), history_1['obj_func'], label="somf")
        ax1.plot(np.arange(num_epoch), history_2['E'], label="myukr")

        ax2 = fig.add_subplot(122)
        epo =-1
        ax2.scatter(history_1['z'][epo, :, 0], history_1['z'][epo, :, 1], label="somf")
        ax2.scatter(history_2['Z'][epo, :, 0], history_2['Z'][epo, :, 1], label="myukr")
        plt.show()

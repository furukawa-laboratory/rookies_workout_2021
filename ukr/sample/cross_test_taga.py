import numpy as np
import data
from somf import UnsupervisedKernelRegression as SomfUKR
from ukr_uniform import UKR as UniformUKR
from ukr_gaussian import UKR as GaussianUKR
from ukr_taga import UKR as tagaUKR
from matplotlib import pyplot as plt

seed = 1

if __name__ == '__main__':
    N = 400
    X = data.gen_saddle_shape(num_samples=N, random_seed=seed, noise_scale=0.05)

    params_for_gukr = dict(
        latent_dim=2,
        eta=1,
        rambda=0,
        sigma=0.1,
        scale=1e-2
    )
    params_for_tagaukr = dict(
        N=N,
        D=3,
        L=2,
        eta=1,
        sigma=0.1,
        scale=1e-2
    )

    ukr_1 = GaussianUKR(**params_for_gukr)
    history_1 = ukr_1.fit(X, num_epoch=100, seed=seed, f_resolution=10)
    ukr_2 = tagaUKR(**params_for_tagaukr)
    history_2 = ukr_2.fit(X, num_epoch=100, seed=seed, resolution=10)

    is_Y_close = np.allclose(history_1['Y'], history_2['Y'])
    is_f_close = np.allclose(history_1['f'], history_2['f'])
    is_Z_close = np.allclose(history_1['Z'], history_2['Z'])
    print("Y: ", is_Y_close)
    print("f: ", is_f_close)
    print("Z: ", is_Z_close)
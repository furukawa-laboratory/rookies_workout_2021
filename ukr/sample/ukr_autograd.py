# Copyright 2021 tanacchi
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software
# and associated documentation files (the "Software"),
# to deal in the Software without restriction,
# including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import jax.numpy as jnp
import numpy as np
from jax import grad
from tqdm import tqdm

from utils import make_grid


class UKR(object):
    def __init__(self, latent_dim, eta, sigma, scale, clipping=(-1, +1)):
        self.L = latent_dim
        self.eta = eta
        self.sigma = sigma
        self.scale = scale
        self.clipping = clipping

    def fit(self, X, num_epoch=50, seed=0, f_resolution=10, init='random'):
        N, D = X.shape
        if init != 'random':
            Z = init.copy()
        else:
            width = (max(self.clipping) - min(self.clipping))
            mid = sum(self.clipping) / 2
            low, high = mid - self.scale*width, mid + self.scale* width
            np.random.seed(seed)
            Z = np.random.uniform(low=low, high=high, size=(N, self.L))
        X, Z = jnp.array(X), jnp.array(Z)

        history = dict(
            E=np.zeros((num_epoch,)),
            Y=np.zeros((num_epoch, N, D)),
            f=np.zeros((num_epoch, f_resolution**self.L, D)),
            Z=np.zeros((num_epoch, N, self.L)))

        for epoch in tqdm(range(num_epoch)):
            Y = estimate_f(Z, Z, self.sigma)
            Z = estimate_z(X, Z, self.sigma, self.eta, self.clipping)
            Z_new = make_grid(f_resolution, (Z.min(), Z.max()))
            Z_new = jnp.array(Z_new)
            Y_new = estimate_f(Z_new, Z, self.sigma)

            history['E'][epoch] = np.array(obf(X, Z, self.sigma))
            history['Y'][epoch] = np.array(Y)
            history['Z'][epoch] = np.array(Z)
            history['f'][epoch] = np.array(Y_new)
        return history


def estimate_f(Z1, Z2, sigma):
    dists = ((Z1[:, np.newaxis, :] - Z2[np.newaxis, :, :])**2).sum(axis=2)
    R = jnp.exp(-0.5 * dists / sigma**2)
    R /= R.sum(axis=1, keepdims=True)
    return R @ X


def estimate_z(X, Z, sigma, eta, clipping):
    dZ = grad(lambda z: obf(X, z, sigma))(Z)
    Z -= eta * dZ
    Z = np.clip(Z, clipping[0], clipping[1])
    return Z


def obf(X, Z, sigma):
    return jnp.sum((estimate_f(Z, Z, sigma) - X)**2)


if __name__ == '__main__':
    import data
    from visualizer import visualize_history


    X = data.gen_saddle_shape(num_samples=200, random_seed=0, noise_scale=0.05)
    # X = data.gen_2d_sin_curve(100, random_seed=0, noise_scale=0.01)
    ukr = UKR(latent_dim=2, eta=0.1, sigma=0.2, scale=1e-3, clipping=(-1, 1))
    history = ukr.fit(X, num_epoch=300)
    visualize_history(X, history['f'], history['Z'], save_gif=False)

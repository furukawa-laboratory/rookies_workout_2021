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


import numpy as np
try:
    from scipy.spatial import distance
    cdist = distance.cdist
except ModuleNotFoundError:
    print("scipy is not installed, so the custom cdist defined.")
    cdist = lambda XA, XB, metric: np.sum((XA[:, None] - XB[None, :])**2, axis=2)

from tqdm import tqdm

import utils


class SOM(object):
    def __init__(self,
                 latent_dim=2,
                 resolution=10,
                 sigma_max=2.2,
                 sigma_min=0.2,
                 tau=40,
                 seed=None):
        self.L = latent_dim
        self.reso = resolution
        self.K = resolution**latent_dim
        self.sigma = lambda t: max(sigma_min,
                                   (sigma_min - sigma_max) * t / tau + sigma_max)
        self.seed = seed

    def fit(self, X, num_epoch=50):
        N, D = X.shape
        np.random.seed(self.seed)
        Z = np.random.uniform(-1, +1, size=(N, self.L))
        Zeta = utils.make_meshgrid(self.reso, dim=self.L)

        history = {
            'Y': np.zeros(shape=(num_epoch, self.K, D)),
            'Z': np.zeros(shape=(num_epoch, N, self.L)),
            'Zeta': np.zeros(shape=(self.K, self.L)),
        }
        history['Zeta'] = Zeta.copy()

        for epoch in tqdm(range(num_epoch)):
            sigma = self.sigma(epoch)
            Y = m_step(X, Z, Zeta, sigma)
            Z = e_step(X, Y, Zeta)

            history['Y'][epoch] = Y.copy()
            history['Z'][epoch] = Z.copy()
        return history


def m_step(X, Z, Zeta, sigma):
    dists = cdist(Zeta, Z, 'sqeuclidean')
    R = np.exp(-dists / (2 * sigma**2))
    R /= R.sum(axis=1, keepdims=True)
    return R @ X


def e_step(X, Y, Zeta):
    dists = cdist(Y, X, 'sqeuclidean')
    bmu = np.argmin(dists, axis=0)
    return Zeta[bmu]

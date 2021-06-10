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
    cdist = lambda XA, XB: np.sum((XA[:, None] - XB[None, :])**2, axis=2)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print("tqdm is not installed, so progress bar won't be shown.")
    tqdm = iter

from utils import make_grid


class UKR(object):
    def __init__(self, latent_dim, eta, sigma, scale, clipping=(-1, +1)):
        self.L = latent_dim
        self.eta = eta
        self.sigma = sigma
        self.scale = scale
        self.clipping = clipping
        self.kernel = lambda Z1, Z2: np.exp(-cdist(Z1, Z2)**2 / (2 * self.sigma**2))

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
        history = dict(
            E=np.zeros((num_epoch,)),
            Y=np.zeros((num_epoch, N, D)),
            f=np.zeros((num_epoch, f_resolution**self.L, D)),
            Z=np.zeros((num_epoch, N, self.L)))

        for epoch in tqdm(range(num_epoch)):
            Y, R = self.estimate_f(X, Z)
            Z = self.estimate_e(X, Y, Z, R)

            Z_new = make_grid(f_resolution,
                              bounds=(np.min(Z), np.max(Z)),
                              dim=self.L)
            f, _ = self.estimate_f(X, Z_new, Z)

            history['Y'][epoch] = Y
            history['f'][epoch] = f
            history['Z'][epoch] = Z
            history['E'][epoch] = np.sum((Y - X)**2) / N
        return history

    def estimate_f(self, X, Z1, Z2=None):
        Z2 = np.copy(Z1) if Z2 is None else Z2
        kernels = self.kernel(Z1, Z2)
        R = kernels / np.sum(kernels, axis=1, keepdims=True)
        return R @ X, R

    def estimate_e(self, X, Y, Z, R):
        d_ii = Y - X
        d_in = Y[:, np.newaxis, :] - X[np.newaxis, :, :]
        d_ni = - d_in
        δ_in = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]
        δ_ni = - δ_in

        diff_left = np.einsum("ni,nd,nid,nil->nl",
                              R,
                              d_ii,
                              d_ni,
                              δ_ni,
                              optimize=True)
        diff_right = np.einsum("in,id,ind,inl->nl",
                               R,
                               d_ii,
                               d_in,
                               δ_in,
                               optimize=True)
        diff = 2 * (diff_left - diff_right) / (self.sigma**2 * X.shape[0])
        Z -= self.eta * diff
        Z = np.clip(Z, self.clipping[0], self.clipping[1])
        return Z


if __name__ == '__main__':
    import data
    from visualizer import visualize_history


    X = data.gen_saddle_shape(num_samples=200, random_seed=0, noise_scale=0.05)
    # X = data.gen_2d_sin_curve(100, random_seed=0, noise_scale=0.01)
    ukr = UKR(latent_dim=2, eta=8, sigma=0.2, scale=1e-3, clipping=(-1, 1))
    history = ukr.fit(X, num_epoch=300)
    visualize_history(X, history['f'], history['Z'], save_gif=False)

import numpy as np
from tqdm import tqdm
from utils import make_grid

seed = 1

class UKR(object):
    def __init__(self, N, D, L, eta, sigma, scale):
        self.N = N
        self.D = D
        self.L = L
        self.eta = eta
        self.sigma = sigma
        self.scale = scale

    #データの距離
    def distance(self, A, B):
        #Aはデータ(こっちで用意するやつ),Bは構造(ノード)(NxN)
        Dist = np.sum((A[:, None, :]-B[None, :, :])**2, axis=2)
        return Dist
    
    #写像の推定
    def estimate_y(self, Z1, Z2, X):
        #ガウス関数による重み行列k(z_n,z_i)(NxN)
        k = np.exp((-1/(2*(self.sigma**2)))*self.distance(Z1, Z2))
        #kをiでsumした行列K(z_n)(Nx1)
        K = np.sum(k, axis=1) 
        Y = (k@X)/K[:,None]
        return Y

    #潜在変数の推定
    def estimate_z(self, Z1, Z2, X, Y):
        #ガウス関数による重み行列k(z_n,z_i)(NxN)
        k = np.exp((-1/(2*(self.sigma**2)))*self.distance(Z1, Z2))
        #kをiでsumした行列K(z_n)(Nx1)
        K = np.sum(k, axis=1)
    
        r = k/K[:,None]
        d = Y[:,None,:] - X[None,:,:]
        delta = Z1[:,None,:] - Z2[None,:,:]
    
        B = Y - X
        A = np.sum(B[:,None,:]*d,axis=2)
    
        C = r*A
        CC = C+C.T
        E_bibun = (2/(self.N*(self.sigma**2)))*np.sum(CC[:,:,None]*delta,axis=1)
    
        Z_new = Z1 - self.eta*E_bibun
    
        return Z_new

    def fit(self, X, num_epoch, seed, resolution):
        np.random.seed(seed)
        Z = np.random.normal(scale=self.scale, size=(self.N, self.L))

        history = dict(
            Y=np.zeros((num_epoch, self.N, self.D)),
            f=np.zeros((num_epoch, resolution**self.L, self.D)),
            Z=np.zeros((num_epoch, self.N, self.L))
        )

        for epoch in tqdm(range(num_epoch)):
            Y = self.estimate_y(Z, Z, X)
            Z = self.estimate_z(Z, Z, X, Y)
    
            """
            A = np.linspace(Z[:,0].min(), Z[:,0].max(), resolution)
            B = np.linspace(Z[:,1].min(), Z[:,1].max(), resolution)
            XX, YY = np.meshgrid(A,B)
            M = np.concatenate([XX.reshape(-1)[:,None], YY.reshape(-1)[:,None]], axis=1)
            """
            #sampleプログラムに合わせた初期値，make_gridの中身はよくわかってない
            Z_new = make_grid(resolution,
                              bounds=(np.min(Z), np.max(Z)),
                              dim=self.L)
            f = self.estimate_y(Z_new, Z, X)

            history['Y'][epoch] = Y
            history['f'][epoch] = f
            history['Z'][epoch] = Z

        return history


if __name__ == '__main__':
    import data
    from visualizer import visualize_history
    N = 400
    D = 3
    L = 2
    T = 100
    eta = 1
    sigma = 0.1
    resolution = 20

    X = data.gen_saddle_shape(num_samples=400, random_seed=seed, noise_scale=0.05)
    ukr = UKR(N, D, L, eta, sigma, scale=1e-2)
    history = ukr.fit(X, num_epoch=T, seed=seed, resolution=resolution)
    visualize_history(X, history['Z'], history['f'], colormap=X[:,0], resolution=20, T=100, save_gif=False, filename="tmp")

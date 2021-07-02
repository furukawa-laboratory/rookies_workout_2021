import numpy as np
from tqdm import tqdm


class UKR(object):
    def __init__(self, N, D, L, T, eta, sigma, resolution, seed):
        self.N=N
        self.D=D
        self.L=L
        self.T=T
        self.eta = eta
        self.sigma = sigma
        self.resolution=resolution
        self.seed=seed

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
        E_bibun = (2/(N*(sigma**2)))*np.sum(CC[:,:,None]*delta,axis=1)
    
        Z_new = Z1 - self.eta*E_bibun
    
        return Z_new

    def fit(self, X):
        np.random.seed(self.seed)
        Z = 2*np.random.rand(self.N, self.L)-1
        Z *= 0.001

        y_hist=np.zeros((self.T, self.N, self.D))
        z_hist=np.zeros((self.T, self.N, self.L))

        for t in tqdm(range(self.T)):
            Y = self.estimate_y(Z, Z, X)
            Z = self.estimate_z(Z, Z, X, Y)
            z_hist[t]=Z
    
            A = np.linspace(Z[:,0].min(),Z[:,0].max(),self.resolution)
            B = np.linspace(Z[:,1].min(),Z[:,1].max(),self.resolution)
            XX, YY = np.meshgrid(A,B)
            M = np.concatenate([XX.reshape(-1)[:,None], YY.reshape(-1)[:,None]], axis=1)
    
            Y_view = self.estimate_y(M, Z, X)
            y_hist[t]=Y_view
        """
        history = dict(
            E=np.zeros(()),
            Y=np.zeros(()),
            f=np.zeros(()),
            Z=np.zeros(())
        )

        for epoch in tqdm(range(self.T)):
            ...

            history['Y'][epoch] = Y
            history['f'][epoch] = f
            history['Z'][epoch] = Z
            history['E'][epoch] = ...
        return history
        """
        return y_hist,z_hist


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
    seed = 1

    X = data.gen_saddle_shape(num_samples=400, random_seed=1, noise_scale=0.05)
    ukr = UKR(N, D, L, T, eta, sigma, resolution, seed)
    y_hist,z_hist = ukr.fit(X)
    visualize_history(X, z_hist, y_hist, colormap=X[:,0], resolution=20, T=100, save_gif=False, filename="tmp")

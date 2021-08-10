import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils import make_grid

#seed = 0
#import scipy.spatial.distance as dist
try:
    from scipy.spatial import distance
    cdist = distance.cdist
except ModuleNotFoundError:
    print("scipy is not installed, so the custom cdist defined.")
    cdist = lambda XA, XB, metric: np.sum((XA[:, None] - XB[None, :])**2, axis=2)
from tqdm import tqdm


class UKR(object):
    #各変数の準備　インスタンス変数？
    def __init__(self, N, D, L, eta, sigma, rambda, scale,clipping=(-1, +1)):
        self.N = N
        self.D = D
        self.L = L
        self.eta = eta
        self.sigma = sigma
        self.scale = scale
        self.clipping = clipping
        self.λ = rambda
    
    #データの距離
    def distance_function(self,Z1,Z2):
        distance= np.sum((Z1[:, None, :]-Z2[None, :, :])**2, axis=2)
        return distance

    #写像の推定
    def estimate_Y (self,X,Z1,Z2):
        k = np.exp((-1/(2*(self.sigma**2)))*self.distance_function(Z1,Z2))
        K = np.sum(k,axis=1,keepdims=True)
        Y = (k@X)/K
        return Y

    #潜在変数の推定
    def estimate_Z(self,Z1,Z2,X,Y):
        k = np.exp((-1/(2*self.sigma**2))*self.distance_function(Z1,Z2))
        K = np.sum(k,axis=1,keepdims=True)
        r = k/K #N*N
        d_ij = Y[:,None]-X[None,:] #N*N*D
        d_nn = Y-X #N*D
        delta = Z1[:,None]-Z2[None,:] #N*N*L
        #誤差関数の微分 (8)式 : (2/(N*sigma**2))*C (B = A+A_T) (C = Σ[B○δ])
        A = np.einsum("ni,nd,nid->ni",r,d_nn,d_ij) #N*I
        B = A+A.T #N*I
        C = np.einsum("ni,nil->nl",B,delta)
        dE = (2/(self.N*self.sigma**2))*C
        dE += 2 * self.λ * Z2
        #勾配法による潜在変数の更新
        Z_new = Z1-self.eta*dE
        return Z_new



    def fit(self, X, T, f_reso, seed):
        np.random.seed(seed)
        Z = np.random.normal(scale=self.scale, size=(self.N, self.L))

        history = dict(
            E=np.zeros((T,)),
            Z = np.zeros((T, self.N, self.L)),
            Y = np.zeros((T, self.N, self.D)),
            f = np.zeros((T,f_reso,f_reso,self.D))
        )

        for t in range(T):
            Y = self.estimate_Y(X,Z,Z)
            Z = self.estimate_Z(Z,Z,X,Y)

            history['Y'][t] = Y
            history['Z'][t] = Z
            history['E'][t] = np.sum((Y - X)**2) / N + self.λ * np.sum(Z**2)

            A = np.linspace(Z.min(),Z.max(),f_reso)
            B = np.linspace(Z.min(),Z.max(),f_reso)
            XX, YY = np.meshgrid(A,B)
            xx = XX.reshape(-1)
            yy = YY.reshape(-1)
            M = np.concatenate([xx[:, None], yy[:, None]], axis=1) #変数表でいうζkに該当する
            Z_new = make_grid(f_reso,
                              bounds=(np.min(Z), np.max(Z)),
                              dim=self.L)
            f = self.estimate_Y(X,M,Z)
            history['f'][t] = f.reshape(f_reso,f_reso,self.D)
    
        return history

if __name__ == '__main__':
    import data
    seed = 0
    #from visualizer import visualize_history
    N = 100
    D = 3
    L = 2
    T = 200
    eta = 2.0
    sigma = 0.1
    f_reso = 10

    X = data.gen_saddle_shape(num_samples=N, random_seed=1, noise_scale=0.05)
    ukr = UKR(N, D, L, eta, sigma, rambda=0, scale=1e-2,clipping=(-1, 1))
    history = ukr.fit(X, T, f_reso = f_reso, seed=seed)
    fig = plt.figure(figsize=(10, 5))
    ax_observable = fig.add_subplot(122, projection='3d')
    ax_latent = fig.add_subplot(121)
    #%matplotlib nbagg

    def update(i, history, x):
        #plt.cla()
        ax_latent.cla()
        ax_observable.cla()

        fig.suptitle(f"epoch: {i}")
        Z = history['Z'][i]
        f = history['f'][i]

        ax_latent.scatter(Z[:, 0], Z[:, 1], s=50, edgecolors="k", c=x[:, 0])
    
        ax_latent.set_xlim(-1.1, 1.1)
        ax_latent.set_ylim(-1.1, 1.1)

        ax_observable.scatter(x[:, 0], x[:, 1],x[:, 2], c=x[:, 0], s=50, marker='x')
        ax_observable.plot_wireframe(f[:, :, 0], f[:, :, 1],f[:, :, 2], color='black')
    

        ax_observable.set_xlim(x[:, 0].min(), x[:, 0].max())
        ax_observable.set_ylim(x[:, 1].min(), x[:, 1].max())

    ani = animation.FuncAnimation(fig, update, fargs=(history, X), interval=50, frames=T)
    ani.save("change_ver1.gif", writer = "pillow")
    plt.show()
    #HTML(ani.to_jshtml())
    
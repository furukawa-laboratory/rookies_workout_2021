import numpy as np
import matplotlib.pyplot as plt
N=100
L=2
def gen_kura_data(num):
    #num = params.num_samples
    np.random.seed(0)
    z1 = np.random.uniform(low=-1, high=+1, size=(num))
    z2 = np.random.uniform(low=-1, high=+1, size=(num))

    X = np.empty(shape=(num, 3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = 0.5 * (z1**2 - z2**2)
    return X

X = gen_kura_data(N)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:, 0], X[:, 1], X[:, 2])
#plt.show()

Z = np.random.rand(N,L)
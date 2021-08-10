# ver3
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")
from sklearn.base import BaseEstimator
from mmnet.lib.models.miyazaki.base.history_dataclass import History


class UKR(BaseEstimator):
    def __init__(self, data, learn_info, dev=False):
        self.dev = dev
        if data.data_type == 'multi_variate':
            self.X = data.values
            if self.X.ndim == 1:
                self.N = self.X.shape[0]
                self.D = 1
            else:
                self.N, self.D = self.X.shape
        else:
            raise ValueError("this is for multivariate data")

        self.nb_epoch = learn_info.nb_epoch
        self.eta = learn_info.step_width
        self.sigma = learn_info.kernel_width
        if learn_info.regularization_approach == 'penalty_term':
            self.is_clip = False
            self.alpha = learn_info.regularization_term
        elif learn_info.regularization_approach is None:
            self.is_clip = False
            self.alpha = 0
        elif learn_info.regularization_approach == 'restrict_z':
            self.is_clip = True
            self.bounded_shape = learn_info.shape
            self.bounded_range = learn_info.range
        else:
            raise ValueError('we dont know such approach')

        self.L = learn_info.latent_dim
        init = learn_info.init_method
        if isinstance(init, str) and init in 'random':
            try:
                seed = learn_info.initz_seed
            except KeyError:
                seed = 1
                print("This Initial_Z is made by default seed: {}").format(seed)
            random_state = np.random.RandomState(seed)

            if learn_info.latent_distribution == 'gaussian':
                self.Z = random_state.normal(0, 0.1, (self.N, self.L))
            elif learn_info.latent_distribution == 'uniform':
                self.Z = random_state.uniform(-1e-5, 1e-5, (self.N, self.L))
            else:
                raise ValueError('we dont know such init')
        elif isinstance(init, np.ndarray) and init.shape == (self.N, self.L):
            self.Z = init.copy()
            print("This Initial_Z is used External varibales")
        else:
            raise ValueError("invalid init: {}".format(init))

        self.history = History(self.X.shape, self.Z.shape, self.nb_epoch)
        self.history.z[0, :, :] = self.Z

    def nadaraya_watson_estimator(self, z1: tf.Variable, z2: tf.Variable, x, sigma=1):
    # This is Nadaraya-Watson Esitmator
        Dist = tf.reduce_sum(tf.square(z1[:, None, :] - z2[None, :, :]), axis=2)
        H = tf.exp(-0.5 * Dist / (sigma**2))
        G = tf.reduce_sum(H, axis=1, keepdims=True)
        R = H / G
        return R @ x

    def estimate_x(self, z1, z2, x, ret_numpy=False):
        x_hat = self.nadaraya_watson_estimator(z1, z2, x)
        if ret_numpy:
            x_hat = x_hat.numpy()
        return x_hat

    def E(self, Z: tf.Variable, X: np.ndarray) -> tf.Variable:
        self.x_hat = self.estimate_x(Z, Z, X)
        E = 0.5 * tf.reduce_sum((self.x_hat - X)**2) / self.N
        return E

    def fit(self) -> np.ndarray:
        tZ = tf.Variable(self.Z)
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.eta)
        for t in range(self.nb_epoch):
            with tf.GradientTape() as tape:
                result = self.E(tZ, self.X)
            grad = tape.gradient(result, tZ)
            optimizer.apply_gradients([(grad, tZ)])
            self.history.E[t + 1] = result.numpy()
            self.history.z[t + 1, :, :] = tZ.numpy()
            self.history.x_hat[t + 1, :, :] = self.x_hat

    def extrapolation_create_f(self, all_z, x, resolution):
        import itertools
        T, N, L = all_z.shape
        N , D = x.shape

        all_mesh1 = np.zeros((T, resolution, 1))
        all_mesh2 = np.zeros((T, resolution, resolution, L))
        all_zeta = np.zeros((T, resolution, resolution, L))
        all_f_2d = np.zeros((T, resolution, resolution, D))
        np.random.seed(1)
        a = np.sort(np.random.beta(0.5, 0.5, int(resolution/2)))
        for t in range(T):
            zmin = np.min(all_z[t, :, :])
            zmax = np.max(all_z[t, :, :])
            r = np.max([zmax, np.abs(zmin)])
            r = r + 0.6 * r
            b = np.concatenate([a,np.abs(1-a)])
            c = (b/(np.max(b)- np.min(b))) * 2 * r  - r
            mesh1 = np.sort(c)
            # mesh1 = np.linspace(zmin, zmax, resolution, endpoint=True)
            mesh2 = np.array(list(itertools.product(mesh1, repeat=2)))
            f = self.estimate_x(mesh2, all_z[t, :, :], x)
            all_mesh1[t, :, :] = mesh1[:, None] 
            all_zeta[t, :, :, :] = mesh2.reshape(resolution, resolution, L)
            all_mesh2[t, :, :, :] = mesh2.reshape(resolution, resolution, L)
            all_f_2d[t, :, :, :] = f.reshape(resolution, resolution, D)

        return all_mesh1, all_mesh2, all_f_2d

    def create_f(self, all_z, x, resolution):
        import itertools
        T, N, L = all_z.shape
        N , D = x.shape

        all_mesh1 = np.zeros((T, resolution, 1))
        all_mesh2 = np.zeros((T, resolution, resolution, L))
        all_zeta = np.zeros((T, resolution, resolution, L))
        all_f_2d = np.zeros((T, resolution, resolution, D))
        for t in range(T):
            zmin = np.min(all_z[t, :, :])
            zmax = np.max(all_z[t, :, :])
            mesh1 = np.linspace(zmin, zmax, resolution, endpoint=True)
            mesh2 = np.array(list(itertools.product(mesh1, repeat=2)))
            # f = self.nadaraya_watson_estimator(mesh2, all_z[t, :, :], x)
            f = self.estimate_x(mesh2, all_z[t, :, :], x, ret_numpy=True)
            all_mesh1[t, :, :] = mesh1[:, None] 
            all_zeta[t, :, :, :] = mesh2.reshape(resolution, resolution, L)
            all_mesh2[t, :, :, :] = mesh2.reshape(resolution, resolution, L)
            all_f_2d[t, :, :, :] = f.reshape(resolution, resolution, D)

        return all_mesh1, all_mesh2, all_f_2d


if __name__ == "__main__":

# self-made files
    from mmnet.lib.models.miyazaki.base.dataclass import ToyMultiVariate
    from mmnet.lib.models.miyazaki.base.learning_dataclass import LearningInfo
    from mmnet.lib.models.miyazaki.base.history_dataclass import History

    import dataclasses as dc
    from mmnet.lib.datasets.artificial.saddle_dc import load_saddle
    from mmnet.lib.graphics.plot_animation_2view import View

    seed = 1
    sample_num = 200
    data_latent_distribution = 'uniform'

    nb_epoch = 50
    step_width = 2.7 * sample_num
    regularization_approach = None

    # for view
    resolution = 10

    init_method = 'random'
    init_latent_distribution = 'uniform'
    latent_dim = 2

    saddle_data = ToyMultiVariate(
        data_name="saddle_shape",
        latent_distribution=data_latent_distribution,
        random_seed=seed
        )

    saddle_data.sample_num = sample_num
    saddle_data.values, saddle_data.latent_values = load_saddle(dc.asdict(saddle_data))
    saddle_data.sample_num, saddle_data.feature_num = saddle_data.values.shape
    print("Data_shape", saddle_data.values.shape)

    learn_info = LearningInfo(
        init_method, init_latent_distribution, latent_dim,
        step_width,
        regularization_approach,
        nb_epoch
        )
    model = UKR(saddle_data, learn_info)
    model.fit()
    model.history.mesh1, model.history.mesh2, model.history.f = model.create_f(model.history.z, model.X, resolution)
    # model.history.mesh1, model.history.mesh2, model.history.f = model.extrapolation_create_f(model.history.z, model.X, model.history.resolution)
    model.history.zeta = model.history.mesh2

    view = View(model)
    view.plot(show_mesh=False, rotation=True, check_reshape=False, show_f=True,show_legend=True)
    view.show()
    # view.save()

    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=[6, 8])
    # plt.plot(range(nb_epoch + 1), model.history.E)
    # plt.show()


    # print(model.get_params())
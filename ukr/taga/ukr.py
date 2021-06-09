import numpy as np
from tqdm import tqdm


class UKR(object):
    def __init__(self, _):
        NotImplemented

    def fit(self, _):
        history = dict(
            E=np.zeros(()),
            Y=np.zeros(()),
            f=np.zeros(()),
            Z=np.zeros(())
        )

        for epoch in tqdm(range(0)):
            ...

            history['Y'][epoch] = Y
            history['f'][epoch] = f
            history['Z'][epoch] = Z
            history['E'][epoch] = ...
        return history


if __name__ == '__main__':
    import data
    from visualizer import visualize_history

    # 動かなさそうで動いてしまうプログラム
    # お好きに書き直してください
    X = data.gen_saddle_shape(num_samples=200, random_seed=0, noise_scale=0.05)
    ukr = UKR(...)
    history = ukr.fit(...)
    visualize_history(X, history, save_gif=False)

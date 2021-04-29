from data import gen_kura_data
from som import SOM
import params
from visualizer import visualize_history


if __name__ == '__main__':
    X = gen_kura_data(params)

    som = SOM(resolution=params.resolution,
              latent_dim=params.latent_dim,
              sigma_max=params.sigma_max,
              sigma_min=params.sigma_min,
              tau=params.tau)
    history = som.fit(X, num_epoch=params.num_epoch)

    visualize_history(X, history, params)

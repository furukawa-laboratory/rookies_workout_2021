from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_history(X, history, params):
    N, D = X.shape
    Zeta = history['Zeta']
    colormap = X[:, 0]
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    def update(i):
        Y = history['Y'][i]
        Z = history['Z'][i]
        fig.suptitle(f"epoch: {i}")
        ax1.cla()
        if params.latent_dim == 2 and D == 3:
            Y = Y.reshape(params.resolution, params.resolution, D)
            ax1.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')
        else:
            ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colormap)
        ax2.cla()
        if params.latent_dim == 2:
            ax2.scatter(Zeta[:, 0], Zeta[:, 1], marker='x')
            ax2.scatter(Z[:, 0], Z[:, 1], c=colormap)
        else:
            ax2.scatter(Zeta[:, 0], [0] * params.resolution, marker='x')
            ax2.scatter(Z[:, 0], [0] * N, c=colormap)

    ani = FuncAnimation(fig,
                        update,
                        frames=params.num_epoch,
                        repeat=True,
                        interval=params.interval)

    if params.save_fig:
        ani.save(params.filename, writer=params.writer)
    else:
        plt.show()

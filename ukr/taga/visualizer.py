import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

def visualize_history(X, z_hist, f_hist, e_hist, color, resolution, save_gif=False, filename="tmp"):
    input_dim, latent_dim = X.shape[1], z_hist[0].shape[1]
    num_epoch = len(z_hist)
    projection = '3d' if input_dim > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10,7))
    fig.text(0.05, 0.9, f"{filename}")
    gs = GridSpec(nrows=2, ncols=2, height_ratios=[1,0.5])
    input_ax = fig.add_subplot(gs[0,0], projection=projection)
    latent_ax = fig.add_subplot(gs[0,1])
    objective_ax = fig.add_subplot(gs[1,:])

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=num_epoch,
        repeat=True,
        interval=100,
        fargs=(input_dim, latent_dim, input_ax, latent_ax, objective_ax, z_hist, f_hist, e_hist, 
               X, num_epoch, color, resolution))
    plt.show()

    if save_gif:
        ani.save(f"{filename}.mp4", writer='ffmpeg')

def update(epoch, input_dim, latent_dim, input_ax, latent_ax, objective_ax, z_hist, f_hist, e_hist, 
           X, num_epoch, color, resolution):
    input_ax.cla()
    latent_ax.cla()
    objective_ax.cla()
    Z = z_hist[epoch]
    Y = f_hist[epoch]
    E = e_hist

    if input_dim == 3:
        draw_observation_3D(input_ax, X, Y, latent_dim, color, resolution)
    else:
        draw_observation_2D
    
    if latent_dim == 2:
        draw_latent_2D(latent_ax, Z, z_hist, color)
    else:
        draw_latent_1D(latent_ax, Z, color)
    
    draw_objective(objective_ax, E, epoch, num_epoch)

def draw_observation_3D(ax, X, Y, latent_dim, color, resolution):
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
    if latent_dim == 2:
        ax.plot_wireframe(
            Y[:, 0].reshape(resolution,resolution),
            Y[:, 1].reshape(resolution,resolution),
            Y[:, 2].reshape(resolution,resolution),
            color='k')
    else:
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='k')

def draw_observation_2D(ax, X, Y, color):
    ax.scatter(X[:, 0], X[:, 1], c=color)
    ax.plot(Y[:, 0], Y[:, 1], c='k')

def draw_latent_2D(ax, Z, z_hist , color):
    ax.set_xlim(z_hist[:,:,0].min()-0.1,z_hist[:,:,0].max()+0.1)
    ax.set_ylim(z_hist[:,:,1].min()-0.1,z_hist[:,:,1].max()+0.1)
    ax.scatter(Z[:, 0], Z[:, 1], c=color, marker='x', linewidth=2)

def draw_latent_1D(ax, Z, color):
    ax.scatter(Z, np.zeros(Z.shape), c=color)
    ax.set_ylim(-1, 1)

def draw_objective(ax, E, i, num_epoch):
    x=np.linspace(1,num_epoch,num_epoch)
    ax.set_title(f"{i+1}epoch")
    ax.plot(x,E)
    ax.scatter(x[i], E[i], marker='d', c='red')
    ax.set_ylim(0, 0.5)
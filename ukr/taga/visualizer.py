import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize_history(X, z_hist, y_hist, colormap, resolution, T, save_gif=False, filename="tmp"):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    ani = animation.FuncAnimation(fig=fig, func=update, frames=range(T), repeat=True, fargs=(ax1, ax2, z_hist, y_hist, X, colormap, resolution))
    plt.show()

    if save_gif:
        #ani.save(f"learning_sigma{sigma}_N{N}_eta{eta}_resolution{resolution}_seed{seed}.gif", writer="pillow")
        ani.save(f"{filename}.mp4", writer='ffmpeg')

def update(i, ax1, ax2, z_hist, y_hist, X, colormap, resolution):
    ax1.cla()
    ax2.cla()
    Z = z_hist[i]
    Y = y_hist[i]
    
    plt.title(f"学習回数{i+1}回目", fontname="MS Gothic")
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colormap)
    ax1.plot_wireframe(Y[:, 0].reshape(resolution,resolution), Y[:, 1].reshape(resolution,resolution), Y[:, 2].reshape(resolution,resolution), color='k')
    #ax2.scatter(M[:, 0], M[:, 1], alpha=0.4, marker='D')
    ax2.set_xlim(z_hist[:,:,0].min()-0.1,z_hist[:,:,0].max()+0.1)
    ax2.set_ylim(z_hist[:,:,1].min()-0.1,z_hist[:,:,1].max()+0.1)
    ax2.scatter(Z[:, 0], Z[:, 1], c=colormap, marker='x', linewidth=2)
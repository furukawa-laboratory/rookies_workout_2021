# data
num_samples = 200


# model
resolution = 10
latent_dim = 1
sigma_max = 2.2
sigma_min = 0.2
tau = 40
num_epoch = 50

# visualizer
interval = 100
save_fig = False
extension = 'gif'
writer = 'pillow' if extension == 'gif' else 'ffmpeg'
filename = f"SOM_num={num_samples}_reso={resolution}_smax={sigma_max}_smin={sigma_min}_tau={tau}_epoch={num_epoch}." + extension

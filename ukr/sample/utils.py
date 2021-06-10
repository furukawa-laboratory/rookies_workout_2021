import numpy as np


def make_grid(resolution, bounds=(-1, +1), dim=2):
    mesh, step = np.linspace(bounds[0],
                             bounds[1],
                             resolution,
                             endpoint=False,
                             retstep=True)
    mesh += step / 2.0
    grid = np.meshgrid(*[np.copy(mesh)]*dim)
    return np.dstack(grid).reshape(-1, dim)


if __name__ == '__main__':
    grid = make_grid(5, dim=3)
    print(f"3D:\n{grid}\n")
    grid = make_grid(5, dim=2)
    print(f"2D:\n{grid}\n")
    grid = make_grid(5, dim=1)
    print(f"1D:\n{grid}\n")

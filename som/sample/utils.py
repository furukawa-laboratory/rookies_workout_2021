import numpy as np


def make_meshgrid(resolution, domain=(-1, +1), dim=2):
    base, step = np.linspace(domain[0],
                             domain[1],
                             resolution,
                             retstep=True,
                             endpoint=False)
    base += step / 2
    grids = np.meshgrid(*[base]*dim)
    return np.vstack(np.dstack(grids))


if __name__ == '__main__':
    grid = make_meshgrid(5)
    print("2D:")
    print(grid)

    print("==============")

    grid = make_meshgrid(5, dim=1)
    print("1D:")
    print(grid)

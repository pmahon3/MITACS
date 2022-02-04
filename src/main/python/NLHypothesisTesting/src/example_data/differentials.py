import numpy as np


def lorenz(steps, noise_scale: float = 0):
    def classic_lorenz(x, y, z, s=10, r=28, b=2.667):
        """
        Sourced from: https://matplotlib.org/3.1.0/gallery/mplot3d/lorenz_attractor.html
        Given:
           x, y, z: a point of interest in three dimensional space
           s, r, b: parameters defining the lorenz attractor
        Returns:
           x_dot, y_dot, z_dot: values of the lorenz attractor's partial
               derivatives at the point x, y, z
        """
        dx = s * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z
        return dx, dy, dz

    dt = 0.01

    # Need one more for the initial values
    xs = np.empty(steps + 1)
    ys = np.empty(steps + 1)
    zs = np.empty(steps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(steps):
        x_dot, y_dot, z_dot = classic_lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        xs[i + 1] = xs[i + 1] + noise_scale * abs(xs[i + 1]) * np.random.uniform(low=-1, high=1, size=(1,))[0]
        ys[i + 1] = ys[i] + (y_dot * dt)
        ys[i + 1] = ys[i + 1] + noise_scale * abs(ys[i + 1]) * np.random.uniform(low=-1, high=1, size=(1,))[0]
        zs[i + 1] = zs[i] + (z_dot * dt)
        zs[i + 1] = zs[i + 1] + noise_scale * abs(zs[i + 1]) * np.random.uniform(low=-1, high=1, size=(1,))[0]

    return xs, ys, zs

from streaking.components.gaussian_beam import SimpleGaussianBeam
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    beam = SimpleGaussianBeam()
    z = np.linspace(-2e-4, 2e-4, 10000)
    x = y = np.zeros_like(z) + 0.001
    E, B = beam.fields(x, y, z, 0)

    print(E.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(E[..., 0], E[..., 1], z * 1e6)
    ax.set_xlabel(r"$E_x$ / Vm$^{-1}$")
    ax.set_ylabel(r"$E_y$ / Vm$^{-1}$")
    ax.set_zlabel(r"$z$ / µm")
    plt.show()

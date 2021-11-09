from streaking.electrons import ClassicalElectrons
from streaking.conversions import spherical_to_cartesian, cartesian_to_spherical
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

if __name__ == "__main__":

    n = int(1e4)
    focus = 2e-5
    cone_angle = np.deg2rad(6.4)
    E = np.random.uniform(100, 100, n) * const.e
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.random.uniform(0, cone_angle, n)

    px, py, pz = spherical_to_cartesian(1, theta, phi)

    r = np.random.multivariate_normal((0, 0, 0), focus**2 * np.eye(3), n)
    p = np.vstack((pz, py, px)).T
    electrons = ClassicalElectrons(r, p, E, 0)
    electrons.cst_export('simulations/build/cst_6.4deg_+x_100eV_10k.pid')
    print(electrons.Ekin().mean() / const.e)

    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(*electrons[:100].r.T, *electrons[:100].p.T, length=1e-3, normalize=True, alpha=0.2)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(-1e-3, 1e-3), ylim=(-1e-3, 1e-3), zlim=(-1e-3, 1e-3))
    plt.show()
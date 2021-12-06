from streaking.electrons import ClassicalElectrons
from streaking.stats import rejection_sampling_spherical
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt


def pdf(ϑ, φ, cone_angle):
    return ϑ < cone_angle


if __name__ == "__main__":
    n = int(1e5)
    focus = 2e-5
    cone_angle = np.deg2rad(6.4)  # np.deg2rad(1.946572802)
    # E = np.random.uniform(50, 150, n) * const.e
    energy_steps = np.arange(1450, 1550, 1)
    E = np.concatenate([np.full(n // len(energy_steps), i) for i in energy_steps]) * const.e
    px, py, pz = rejection_sampling_spherical(pdf, n, params=(cone_angle,))
    r = np.random.multivariate_normal((0, 0, 0), focus**2 * np.eye(3), n)
    p = np.vstack((pz, py, px)).T
    electrons = ClassicalElectrons(r, p, E, 0)
    electrons.cst_export('simulations/build/cst_6.4deg_+x_1450eV_1550eV_1eV_100k.pid')
    print(electrons.Ekin().mean() / const.e)

    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(*electrons[:100].r.T, *electrons[:100].p.T, length=1e-3, normalize=True, alpha=0.2)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(-1e-3, 1e-3), ylim=(-1e-3, 1e-3), zlim=(-1e-3, 1e-3))
    # plt.show()
from streaking.electrons import ClassicalElectrons
from streaking.stats import rejection_sampling_spherical
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt


def pdf(ϑ, φ, cone_angle):
    return ϑ < cone_angle


def generate_electrons(energy_start, energy_end, energy_step):
    n = int(5e4)
    focus = 1e-5
    cone_angle = np.deg2rad(6.7)  # np.deg2rad(1.946572802)
    # E = np.random.uniform(50, 150, n) * const.e
    # energy_steps = np.arange(1450, 1550, 1)
    energy_steps = np.arange(energy_start, energy_end + energy_step, energy_step)
    print(energy_steps)
    E = np.repeat(energy_steps, n // len(energy_steps)) * const.e
    px, py, pz = rejection_sampling_spherical(pdf, n, params=(cone_angle,))
    r = np.random.multivariate_normal((0, 0, 0), focus**2 * np.eye(3), px.shape[0])
    r[:, 2] = np.random.uniform(-3.5e-3, 3.5e-3, px.shape[0])

    p = np.vstack((pz, py, px)).T
    electrons = ClassicalElectrons(r, p, E, 0)
    electrons.cst_export(f'simulations/build/cst_6.7deg_+x_{energy_start}eV_{energy_end}eV_{energy_step}eV_{n//1000}k_zspread.pid')

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.quiver(*electrons[:100].r.T, *electrons[:100].p.T, length=1e-3, normalize=True, alpha=0.2)
    # ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(-1e-3, 1e-3), ylim=(-1e-3, 1e-3), zlim=(-1e-3, 1e-3))
    # plt.show()


generate_electrons(351, 750, 2)
#if __name__ == '__main__':
#    for i in range(1002, 4002, 200):
#        generate_electrons(i, i + 200 - 2, 2)
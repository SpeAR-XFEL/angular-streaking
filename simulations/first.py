from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
import scipy.constants as const

if __name__ == "__main__":
    beam = SimpleGaussianBeam(energy=3.5e-8)

    XFEL_intensity = lambda t: (
        0.6 * scipy.stats.norm(0, 1e-15).pdf(t)
        + 0.4 * scipy.stats.norm(3e-15, 1e-15).pdf(t)
    )

    XFEL_photon_energy = scipy.stats.norm(1200, 0.3).pdf

    pe = ionizer_simple(
        2,  # β
        XFEL_intensity,
        XFEL_photon_energy,
        870.2,  # binding energy
        (1190, 1210),  # considered energy range
        (-5e-15, 7e-15),  # considered time range
        100000,  # number of electrons to generate (not yet based on cross section)
    )

    spe = classical_lorentz_streaker(pe, beam, (0, 1e-11))

    # x = np.linspace(-5e-15, 7e-15, 100)
    # plt.plot(x, XFEL_intensity(x))

    r, phi, theta = cartesian_to_spherical(*pe.p.T)
    sr, sphi, stheta = cartesian_to_spherical(*spe.p.T)
    rsr, _, _ = cartesian_to_spherical(*spe.r.T)
    # plt.hist(theta,  density=False, color="C0", alpha=0.7, bins=100, histtype='step')
    # plt.hist(stheta, density=False, color="C1", alpha=0.7, bins=100, histtype='step')
    # plt.show()

    # quit()
    plt.subplot(121)
    bins = [np.linspace(0, 2 * np.pi, 101), 100]
    plt.hist2d((theta + np.pi / 2) % (2 * np.pi), pe.Ekin() / const.e, bins=bins)
    plt.title("Unstreaked")
    plt.subplot(122)
    plt.title("Streaked")
    plt.hist2d((stheta + np.pi / 2) % (2 * np.pi), spe.Ekin() / const.e, bins=bins)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(E[..., 0], E[..., 1], z * 1e6)
    # ax.set_xlabel(r"$E_x$ / Vm$^{-1}$")
    # ax.set_ylabel(r"$E_y$ / Vm$^{-1}$")
    # ax.set_zlabel(r"$z$ / µm")
    # plt.show()

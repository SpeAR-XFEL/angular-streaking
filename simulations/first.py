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
    beam = SimpleGaussianBeam(energy=1e-8)
    z = np.linspace(-2e-4, 2e-4, 10000)
    x = y = np.zeros_like(z) + 0.001
    E, B = beam.fields(x, y, z, 0)

    XFEL_intensity = lambda t: (
        0.6 * scipy.stats.norm(0, 1e-15).pdf(t)
        + 0.4 * scipy.stats.norm(3e-15, 1e-15).pdf(t)
    )

    XFEL_photon_energy = scipy.stats.norm(914, 0.3).pdf

    photoelectrons = ionizer_simple(
        2,  # β
        XFEL_intensity,
        XFEL_photon_energy,
        870.2,  # binding energy
        (900, 960),  # considered energy range
        (-5e-15, 7e-15),  # considered time range
        100000,  # number of electrons to generate (not yet based on cross section)
    )

    streaked_electrons = classical_lorentz_streaker(photoelectrons, beam, (0, 1e-11))

    # x = np.linspace(-5e-15, 7e-15, 100)
    # plt.plot(x, XFEL_intensity(x))

    r, phi, theta = cartesian_to_spherical(*photoelectrons.p.T)
    sr, sphi, stheta = cartesian_to_spherical(*streaked_electrons.p.T)
    rsr, _, _ = cartesian_to_spherical(*streaked_electrons.r.T)
    E = photoelectrons.Ekin() / const.e
    # plt.hist(rsr,  density=False, color="C0", alpha=1, bins=100, histtype='step')
    # plt.hist(streaked_electrons.Ekin()/const.e, density=True, color="C1", alpha=1, bins=100, histtype='step')
    plt.hist2d(theta, photoelectrons.Ekin() / const.e, bins=100)
    # plt.show()
    # plt.close()
    # plt.hist2d(stheta, streaked_electrons.Ekin()/const.e, bins=100)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(E[..., 0], E[..., 1], z * 1e6)
    # ax.set_xlabel(r"$E_x$ / Vm$^{-1}$")
    # ax.set_ylabel(r"$E_y$ / Vm$^{-1}$")
    # ax.set_zlabel(r"$z$ / µm")
    # plt.show()

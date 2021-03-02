from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

if __name__ == "__main__":
    number_of_electrons = 1000
    binding_energy = 870.2  # eV
    β = 2
    xfel_duration = 1e-15  # s
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m

    xfel_time_energy_means = (0, xfel_energy)
    xfel_time_energy_covariance = np.diag((xfel_duration, xfel_energy_std)) ** 2
    pe = ionizer_simple(  # pe: photoelectrons
        β,
        xfel_time_energy_means,
        xfel_time_energy_covariance,
        binding_energy,
        xfel_focal_spot,
        number_of_electrons,
    )

    streaking_beam = SimpleGaussianBeam(
        focal_size=(500e-6 / 2.3548, 500e-6 / 2.3548),
        envelope_offset=0,
        cep=np.pi/3,
        wavelength=10e-6,
        energy=30e-6,
        duration=300e-15)
    streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, 1e-12), 1e-14)
    r, phi, theta = cartesian_to_spherical(*pe.p.T)
    sr, sphi, stheta = cartesian_to_spherical(*streaked_pe.p.T)

    # The simple assumption here is that the electrons are still pretty much in the origin,
    # meaning that the detector they end up is just determined by their current flight angle

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.hist2d((theta + np.pi / 2) % (2 * np.pi), pe.Ekin() / const.e, bins=64)
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    plt.subplot(122)
    plt.hist2d((stheta + np.pi / 2) % (2 * np.pi), streaked_pe.Ekin() / const.e, bins=64)
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    plt.tight_layout(pad=0.5)
    plt.show()

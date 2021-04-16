from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.streak import dumb_streaker
from streaking.detectors import constant_polar_angle_ring
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    number_of_electrons = 200000
    binding_energy = 870.2  # eV
    β = 2
    xfel_duration = 1e-15  # s
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m

    xfel_time_energy_means = (0, xfel_energy)
    xfel_time_energy_covariance = np.diag((xfel_duration, xfel_energy_std)) ** 2

    TEmap = MultivariateMapInterpolator.from_gauss_blob_list(
        (xfel_time_energy_means,),
        (xfel_time_energy_covariance,),
        (1,)
    )

    pe = ionizer_simple(  # pe: photoelectrons
        β,
        TEmap,
        xfel_focal_spot,
        binding_energy,
        number_of_electrons,
    )

    streaking_beam = SimpleGaussianBeam(
        focal_size=(500e-6, 500e-6),
        envelope_offset=0,
        cep=np.pi / 3,
        wavelength=10e-6,
        energy=30e-6,
        duration=300e-15)

    streaked_pe = dumb_streaker(pe, streaking_beam)

    histogram, bins_phi, bins_E = constant_polar_angle_ring(pe, np.pi / 2, 0.5, 64, 0.25, 'kinetic energy', 200)
    histogram_s, bins_phi_s, bins_E_s = constant_polar_angle_ring(streaked_pe, np.pi / 2, 0.5, 64, 0.25, 'kinetic energy', 200)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(histogram.T, aspect='auto', origin='lower', interpolation='none')
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    plt.subplot(122)
    plt.imshow(histogram_s.T, aspect='auto', origin='lower', interpolation='none')
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    plt.tight_layout(pad=0.5)
    plt.show()

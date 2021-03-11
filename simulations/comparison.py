from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_Sauter
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

if __name__ == "__main__":
    number_of_electrons = 200000
    binding_energy = 870.2  # eV
    β = 2
    xfel_duration = 1e-15  # s
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m
    theta_center = np.pi / 2
    theta_acceptance = np.pi / 3

    spectrum = np.load('simulations/inputs/spec.npy').T
    spectrum = np.flipud(spectrum)
    TEmap = MultivariateMapInterpolator(((-35.3e-15/2, 35.3e-15/2), (binding_energy, binding_energy + spectrum.shape[1])), spectrum.shape)
    TEmap.add_arbitrary(spectrum)
    
    #print(TEmap.map.shape, spectrum.shape)
    #plt.show()
    #quit()
    pe = ionizer_Sauter(TEmap, binding_energy, number_of_electrons)
    #quit()

    streaking_beam = SimpleGaussianBeam(
        focal_size=(400e-6 / 2.3548, 400e-6 / 2.3548),
        envelope_offset=0,
        cep=0,#np.pi,
        wavelength=10.6e-6,
        energy=250e-6,
        duration=300e-15)
    streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, 1e-12), 1e-14)
    r, theta, phi = cartesian_to_spherical(*pe.p.T)
    sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    mask = np.abs(stheta - theta_center) < theta_acceptance

    phibins = np.linspace(0, 2 * np.pi, 17)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    #plt.hist2d((phi[mask] + np.pi / 2) % (2 * np.pi), pe.Ekin()[mask] / const.e, bins=(64, 64))
    plt.xlabel(r"t / fs")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    plt.hist2d(pe.t0 * 1e15, pe.Ekin() / const.e, bins=100)
    plt.subplot(122)
    plt.hist2d((sphi[mask] + np.pi / 2) % (2 * np.pi), streaked_pe.Ekin()[mask] / const.e, bins=(phibins, 150))
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    plt.tight_layout(pad=0.5)
    plt.show()

from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_Sauter, ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import h5py
import sys

if __name__ == "__main__":
    number_of_electrons = 400000
    binding_energy = 870.2  # eV
    β = 2
    xfel_duration = 1e-15  # s
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m
    theta_center = np.pi / 2
    theta_acceptance = np.pi / 3

    f = h5py.File('simulations/inputs/Kick_0.1-100.0_16x200_1k.h5', 'r')
    idx = int(sys.argv[1])
    spectrum = f['spectrograms'][idx].T
    detector_image = f['detector_images'][idx]
    kick = f['kick'][idx]

    TEmap = MultivariateMapInterpolator(((-35.3e-15/2, 35.3e-15/2), (binding_energy, binding_energy + spectrum.shape[1])), spectrum.shape)
    TEmap.add_arbitrary(spectrum)
    
    pe = ionizer_Sauter(TEmap, binding_energy, number_of_electrons)


    streaking_beam = SimpleGaussianBeam(
        focal_size=(200e-6 / 2.3548, 200e-6 / 2.3548),
        envelope_offset=0,
        cep=np.pi,
        wavelength=10.6e-6,
        energy=30e-6 / 2100 * kick**2, # Adjusted by eye
        duration=300e-15)
    streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, 1e-12), 1e-14)
    r, theta, phi = cartesian_to_spherical(*pe.p.T)
    sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    mask = np.abs(stheta - theta_center) < theta_acceptance

    phibins = np.linspace(0, 2 * np.pi, 17)

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    #plt.hist2d((phi[mask] + np.pi / 2) % (2 * np.pi), pe.Ekin()[mask] / const.e, bins=(64, 64))
    plt.xlabel(r"t / fs")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    #plt.hist2d(pe.t0 * 1e15, pe.Ekin() / const.e, bins=100)
    plt.imshow(spectrum.T, origin='lower', aspect='auto', extent=((-35.3/2, 35.3/2, 0, spectrum.shape[1])))

    plt.title('Spectrogram')
    plt.subplot(132)
    plt.imshow(detector_image, origin='lower', aspect='auto', interpolation='none', extent=(0, detector_image.shape[1], 0, detector_image.shape[0]))
    plt.title('Existing Simulation')

    plt.subplot(133)
    plt.title('Dortmund Simulation')
    #plt.axhline(pe.Ekin().mean() / const.e, color='C1')
    #plt.axhline(streaked_pe.Ekin().mean() / const.e, color='C2')
    KE, sKE = pe.Ekin(), streaked_pe.Ekin()
    print(f'Ratio: {np.sum(sKE > KE) / np.sum(sKE < KE)}')
    #plt.axhline(pe.Ekin().mean() / const.e, color='C1')
    plt.hist2d((sphi[mask] + np.pi / 2) % (2 * np.pi), streaked_pe.Ekin()[mask] / const.e, bins=(phibins, np.arange(201)))
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    plt.tight_layout(pad=0.5)
    plt.savefig(f'simulations/build/comparison/{idx}.png', dpi=300)

from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_Sauter, ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker, dumb_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import sys

if __name__ == "__main__":
    number_of_electrons = 200000
    binding_energy = 870.2  # eV
    β = 2
    xfel_duration = 1e-15  # s
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m
    theta_center = np.pi / 2
    theta_acceptance = np.pi / 4

    dur = xfel_duration
    sigE = xfel_energy_std
    tEmeans = (0, xfel_energy)
    chirp = 0
    tEcov = np.array(((dur**2, chirp * dur * sigE), (chirp * dur * sigE, sigE**2)))
    pe = ionizer_simple(2, tEmeans, tEcov, binding_energy, xfel_focal_spot, number_of_electrons)

    streaking_beam = SimpleGaussianBeam(
        focal_size=(200e-6 / 2.3548, 200e-6 / 2.3548),
        envelope_offset=0,
        cep=np.pi,
        wavelength=10.6e-6,
        energy=30e-6,
        duration=300e-15)
    
    streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, 1e-12), 5e-15)
    #streaked_pe = dumb_streaker(pe, streaking_beam, dumb=False)
    r, theta, phi = cartesian_to_spherical(*pe.p.T)
    sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    mask = np.abs(stheta - theta_center) < theta_acceptance

    phibins = np.linspace(0, 2 * np.pi, 17)

    plt.figure(figsize=(10, 4))
    plt.subplot(141)
    plt.title('Spectrogram')
    plt.xlabel(r"t / fs")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    plt.hist2d(pe.t0 * 1e15, pe.Ekin() / const.e, bins=100)

    plt.subplot(142)
    plt.title('RK4')
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, 1e-12), 1e-14)
    r, theta, phi = cartesian_to_spherical(*pe.p.T)
    sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    mask = np.abs(stheta - theta_center) < theta_acceptance
    hist1, x1, y1 = np.histogram2d((sphi[mask] + np.pi / 2) % (2 * np.pi), streaked_pe.Ekin()[mask] / const.e, bins=(phibins, 200))
    plt.imshow(hist1.T, aspect='auto', extent=(x1[0], x1[-1], y1[0], y1[-1]), origin='lower', interpolation='none')

    plt.subplot(143)
    plt.title(r'Dumb $\vec A$ addition')
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    streaked_pe2 = dumb_streaker(pe, streaking_beam, dumb=False)
    sr2, stheta2, sphi2 = cartesian_to_spherical(*streaked_pe2.p.T)
    mask2 = np.abs(stheta2 - theta_center) < theta_acceptance
    hist2, x2, y2 = np.histogram2d((sphi2[mask2] + np.pi / 2) % (2 * np.pi), streaked_pe2.Ekin()[mask2] / const.e, bins=(x1, y1))
    plt.imshow(hist2.T, aspect='auto', extent=(x2[0], x2[-1], y2[0], y2[-1]), origin='lower', interpolation='none')

    plt.subplot(144)
    plt.title(f'Difference ({mask.sum()} total)')
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$\mathrm{\Delta} E_\mathrm{kin}$ / eV")
    plt.hist2d((sphi2[mask] + np.pi / 2) % (2 * np.pi), (streaked_pe2.Ekin()[mask] - streaked_pe.Ekin()[mask]) / const.e, bins=(phibins, 200))
    #d = (hist2.T-hist1.T)
    #norm = mcolors.TwoSlopeNorm(vmin=d.min(), vcenter=0, vmax=d.max())
    #plt.imshow(d, aspect='auto', extent=(x2[0], x2[-1], y2[0], y2[-1]), origin='lower', interpolation='none', cmap='RdBu_r', norm=norm)
    plt.colorbar()
    
    plt.tight_layout(pad=0.5)
    plt.show()

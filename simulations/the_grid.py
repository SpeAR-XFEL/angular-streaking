from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_Sauter, ionizer_simple
from streaking.conversions import cartesian_to_spherical, spherical_to_cartesian
from streaking.streak import classical_lorentz_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.electrons import ClassicalElectrons
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == "__main__":
    binding_energy = 870.2  # eV
    β = 2
    xfel_duration = 1e-15  # s
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m
    theta_center = np.pi / 2
    theta_acceptance = np.pi / 3

    phi = np.linspace(0, 2 * np.pi, 300)
    theta = np.linspace(0, np.pi, 300)

    phi_, theta_ = np.meshgrid(theta, phi)

    r = np.zeros((len(phi) * len(theta), 3)) + 1e-24
    p = spherical_to_cartesian(1, theta_.flatten(), phi_.flatten()).T

    pe = ClassicalElectrons(r, p, Ekin=50 * const.e)

    streaking_beam = SimpleGaussianBeam(
        focal_size=(100e-6, 100e-6),
        envelope_offset=0,
        cep=0,#np.pi/2,
        wavelength=10.6e-6,
        energy=300000e-6,
        duration=300e-15)

    #print(streaking_beam.field(1e-24, 1e-24, 1e-24, 0))
    streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, 1e-12), 1e-15)

    #r, theta, phi = cartesian_to_spherical(*pe.p.T)
    #sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    #mask = np.abs(stheta - theta_center) < theta_acceptance
    #diff = np.reshape(streaked_pe.Ekin() - pe.Ekin(), (theta.shape[0], phi.shape[0])) / const.e
    #diff = 1e24*np.reshape(np.linalg.norm(streaked_pe.p, axis=1) - np.linalg.norm(pe.p, axis=1), (phi.shape[0], theta.shape[0]))
    diff = np.reshape(streaked_pe.p[:,1] - pe.p[:,1], (theta.shape[0], phi.shape[0])) * 1e25
    #diff = diff - diff.mean()
    r1, t1, p1 = cartesian_to_spherical(*pe.p.T)
    r2, t2, p2 = cartesian_to_spherical(*streaked_pe.p.T)
    #diff = np.reshape((t2-t1), (theta.shape[0], phi.shape[0]))# * 1e24

    #print(diff.shape)
    #print(np.sum(diff>0) / np.sum(diff<0))

    #divnorm = mcolors.TwoSlopeNorm(vmin=diff.min(), vcenter=0, vmax=diff.max())

    fig, ax = plt.subplots()
    plt.title('Momentum difference parallel to the laser field')
    ax.set_xlabel(r'$\varphi$')
    ax.set_ylabel(r'$\vartheta$')
    im = plt.imshow(diff.T, origin='lower', aspect='auto', extent=(0, 2 * np.pi, 0, np.pi), interpolation='none', cmap='RdBu_r')#, norm=divnorm)
    plt.colorbar(label=r'$\mathrm{\Delta} p$ / $10^{-24}$ kg m/s')
    plt.tight_layout(pad=0)
    plt.show()

    #plt.plot(diff)

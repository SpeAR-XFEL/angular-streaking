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

    phi, theta = (0, np.pi), (0, 2 * np.pi)

    r = np.zeros((len(phi), 3)) + 1e-24
    p = spherical_to_cartesian(1, theta, phi).T

    pe = ClassicalElectrons(r, p, Ekin=100 * const.e)

    streaking_beam = SimpleGaussianBeam(
        focal_size=(100e-6 / 2.3548, 100e-6 / 2.3548),
        envelope_offset=0,
        cep=np.pi/2,
        wavelength=10.6e-6,
        energy=30e-6,
        duration=300e-15)

    

    keovert = [pe.p[:,0]]
    t = np.arange(0, 1e-12, 1e-14)
    for t_ in t[1:]:
        streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, t_), 1e-14)
        keovert.append(np.linalg.norm(streaked_pe.p, axis=1))

    keovert = np.asarray(keovert) / const.e

    plt.plot(t, keovert)
    plt.show()

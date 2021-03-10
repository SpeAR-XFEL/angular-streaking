import numpy as np
import scipy.constants as const
from streaking.electrons import ClassicalElectrons
from streaking.stats import rejection_sampling
from streaking.conversions import spherical_to_cartesian
from numpy import pi as π


def diff_cross_section_dipole(φ, β):
    return 1 + β * 1 / 2 * (3 * np.cos(φ) ** 2 - 1)


def ionizer_simple(β, tEmean, tEcov, EB, xfel_spotsize, electrons):
    """
    Generate randomly distributed photoelectrons
    """
    t0, E = np.random.multivariate_normal(tEmean, tEcov, electrons).T
    E -= EB
    E *= const.e  # in Joules

    if int(β) == 2:
        ϑ, φ = rejection_sampling(
            diff_cross_section_Sauter_lowEnergy, ((-π, π), (0, π)), electrons
        )
        px, py, pz = spherical_to_cartesian(1, ϑ, φ)
    else:
        # ClassicalElectrons normalizes this
        px, py, pz = np.random.normal(size=(3, electrons))

    r = np.random.multivariate_normal(
        (0, 0, 0), np.diag((xfel_spotsize, xfel_spotsize, 1e-15)) ** 2, electrons
    )  #
    return ClassicalElectrons(r, np.vstack((px, py, pz)).T, E, t0)


def diff_cross_section_Sauter(θ, φ, ɣ):
    β = np.sqrt(1 - 1 / ɣ ** 2)

    factor1 = 3 / (4 * π * ɣ ** 4) * np.sin(θ) ** 2 / (1 - β * np.cos(θ)) ** 4
    factor2 = (
        1
        + 3
        / 4
        * ɣ
        * (ɣ - 2)
        / (ɣ + 1)
        * (1 - 1 / (2 * β * ɣ ** 2) * np.log((1 + β) / (1 - β)))
    ) ** -1
    factor3 = np.cos(φ) ** 2 * (1 - ɣ / 2 * (ɣ - 1) * (1 - β * np.cos(θ))) + ɣ / 4 * (
        ɣ - 1
    ) ** 2 * (1 - β * np.cos(θ))
    return factor1 * factor2 * factor3


def diff_cross_section_Sauter_lowEnergy(angles):
    θ, φ = angles[:, 0], angles[:, 1]
    return np.sin(θ) ** 2 * np.cos(φ) ** 2


def ionizer_Sauter(TEmap, E_ionize, N_e):
    """
    Generate randomly distributed photoelectrons
    """

    # generate electron birthtimes and kinetic energy from TEmap
    birthtimes, E_photon = rejection_sampling(TEmap.eval, TEmap.domain, N_e)
    Ekin = (E_photon - E_ionize) * const.e  # in eV
    # mean_gamma = np.mean(1 + Ekin / (const.m_e * const.c**2))
    # generate emission angles from Sauter cross section
    theta, phi = rejection_sampling(
        diff_cross_section_Sauter_lowEnergy, ((-π, π), (0, π)), N_e
    )
    px, py, pz = spherical_to_cartesian(1, theta, phi)
    r = np.zeros((N_e, 3)) + 1e-24
    return ClassicalElectrons(r, np.stack((px, py, pz)).T, Ekin, birthtimes)


def naive_auger_generator(photoelectrons, ratio, lifetime, energy):
    N = int(len(photoelectrons) * ratio)
    choice = np.random.choice(len(photoelectrons), N)
    r = photoelectrons.r[choice]
    t0 = photoelectrons.t0[choice] + np.random.exponential(lifetime, N)
    p = np.random.normal(size=(N, 3))
    linewidth = const.hbar / lifetime
    Ekin = energy + np.random.standard_cauchy(N) * linewidth
    # Cut off energies below zero.
    mask = Ekin > 0
    return ClassicalElectrons(r[mask], p[mask], Ekin[mask], t0[mask])

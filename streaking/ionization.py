import numpy as np
import scipy.constants as const
from streaking.electrons import ClassicalElectrons
from streaking.stats import rejection_sampling, rejection_sampling_nD
from streaking.conversions import spherical_to_cartesian
from numpy import pi as π


def diff_cross_section_dipole(φ, β):
    return 1 + β * 1 / 2 * (3 * np.cos(φ) ** 2 - 1)


def ionizer_simple(β, tEmean, tEcov, EB, electrons):
    """
    Generate randomly distributed photoelectrons
    """
    φ = rejection_sampling(diff_cross_section_dipole, (-π, π), electrons, (β,))
    θ = np.zeros(electrons) + π/2 # fixed for now
    t0, E = np.random.multivariate_normal(tEmean, tEcov, electrons).T
    E -= EB

    E *= const.e  # in Joules
    px, py, pz = spherical_to_cartesian(1, θ, φ)
    r = np.random.multivariate_normal((0,0,0), np.diag((4e-4, 4e-4, 1e-12))**2, electrons) #
    return ClassicalElectrons(r, np.vstack((px, py, pz)).T, E, t0)


def diff_cross_section_Sauter(θ, φ, ɣ):
    β = np.sqrt(1 - 1 / ɣ**2)

    factor1 = 3 / (4 * π * ɣ**4) * np.sin(θ)**2 / (1 - β * np.cos(θ))**4
    factor2 = (1 + 3 / 4 * ɣ * (ɣ - 2) / (ɣ + 1) * (1 - 1 / (2 * β * ɣ**2) * np.log((1 + β) / (1 - β))))**-1
    factor3 = np.cos(φ)**2 * (1 - ɣ / 2 * (ɣ - 1) * (1 - β * np.cos(θ))) + ɣ / 4 * (ɣ - 1)**2 * (1 - β * np.cos(θ))
    return factor1 * factor2 * factor3


def diff_cross_section_Sauter_lowEnergy(θ, φ, params=()):
    return np.sin(θ)**2 * np.cos(φ)**2


def ionizer_Sauter(TEmap, E_ionize, N_e, polar_opening_angle=0.1):
    """
    Generate randomly distributed photoelectrons
    """
 
    # generate electron birthtimes and kinetic energy from TEmap
    birthtimes, E_photon = rejection_sampling_nD(
        TEmap.pdf,
        [[TEmap.t0, TEmap.t1], [TEmap.E0, TEmap.E1]],
        N_e,
        (),
    )
    Ekin = (E_photon - E_ionize) * const.e  # in eV
    # mean_gamma = np.mean(1 + Ekin / (const.m_e * const.c**2))
    # generate emission angles from Sauter cross section
    theta, phi = rejection_sampling_nD(
        diff_cross_section_Sauter_lowEnergy,
        np.array([[np.pi / 2 - polar_opening_angle, np.pi / 2 + polar_opening_angle], [-np.pi, np.pi]]),
        N_e,
    )
    px, py, pz = spherical_to_cartesian(1, theta, phi)
    r = np.zeros((N_e, 3)) + 1e-24
    return ClassicalElectrons(r, np.stack((px, py, pz)), Ekin, birthtimes)

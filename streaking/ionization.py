import numpy as np
import scipy.constants as const
from streaking.electrons import ClassicalElectrons
from streaking.stats import rejection_sampling, rejection_sampling_spherical
from scipy.special import eval_legendre as P
import scipy.interpolate


def diff_cross_section_dipole(ϑ, φ, β):
    return (1 + β * 1 / 2 * (3 * np.cos(ϑ) ** 2 - 1)) / 3


def diff_cross_section_1st_nondipole(ϑ, φ, β, Δβ, δ, ɣ, λ, µ, ν):
    # TODO: Improve performance (penalty is significant)
    return (
        1
        + (β + Δβ) * P(2, np.cos(ϑ))
        + (δ + ɣ * np.cos(ϑ) ** 2) * np.sin(ϑ) * np.cos(φ)
        + λ * P(2, np.cos(ϑ) * np.cos(2 * φ))
        + µ * np.cos(2 * φ)
        + ν * (1 + np.cos(2 * φ) * P(4, np.cos(ϑ)))
    ) / 3  # TODO: Verify whether 3 is the highest this can be


def ionizer_total_cs(initial_state, TEMap, xfel_pulse_energy, xfel_spot, target_length, target_density):
    data = np.genfromtxt(f'cross_sections/{initial_state.lower()}.txt')
    e = data.T[0]
    sigma = data.T[1:3].mean(axis=0)
    f = scipy.interpolate.interp1d(e, sigma)

    meanT, meanE = TEMap.get_centroid()
    photons = xfel_pulse_energy / (meanE * const.e)
    cross_section = f(meanE) * 1e6 * 1e-28
    pe_count = int(photons * cross_section * target_density * 1e6 * target_length)
    return ionizer_simple(2, TEMap, xfel_spot, 870.2, pe_count, target_length)


def ionizer_simple(β, TEmap, xfel_spotsize, EB, electrons, target_length):
    """
    Generate randomly distributed photoelectrons
    """
    # t0, E = np.random.multivariate_normal(tEmean, tEcov, electrons).T
    t0, E = rejection_sampling(TEmap.eval, TEmap.domain, electrons)
    E -= EB
    E *= const.e  # in Joules

    px, py, pz = rejection_sampling_spherical(
        diff_cross_section_dipole, electrons, params=(β,)
    )

    # px, py, pz = rejection_sampling_spherical(
    #    diff_cross_section_1st_nondipole, electrons, params=(1.9695, -0.01830, 0, 1.6406, 0.02711, 0.03790, -0.06501)#(1.9995, -0.00025, 0, 0.1982, 0.0004, 0.00056, -0.00097)
    # )

    # Transform coordinates, as in the cross sections, z is along the polarization vector.
    px, py, pz = -pz, py, px

    #r = np.random.multivariate_normal(
    #    (0, 0, 0), np.diag((xfel_spotsize, xfel_spotsize, target_length)) ** 2, electrons
    #)

    x = np.random.normal(0, xfel_spotsize, electrons)
    z = np.random.uniform(-target_length / 2, target_length / 2, electrons)

    r = np.array((x, x, z)).T

    t0 += r[:, 2] / const.c
    return ClassicalElectrons(r, np.vstack((px, py, pz)).T, E, t0)


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

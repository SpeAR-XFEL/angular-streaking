import numpy as np
import scipy.constants as const
from streaking.electrons import ClassicalElectrons
from streaking.stats import rejection_sampling
from streaking.conversions import spherical_to_cartesian


def diff_cross_section_dipole(ϑ, β):
    return 1 + β * 1 / 2 * (3 * np.cos(ϑ) ** 2 - 1)


def ionizer_simple(β, IX, EX, EB, E_range, t_range, electrons):
    """
    Generate randomly distributed photoelectrons
    """
    ϑ = rejection_sampling(diff_cross_section_dipole, (-np.pi, np.pi), electrons, (β,))
    ψ = np.random.uniform(-np.pi, np.pi, electrons)
    t0 = rejection_sampling(IX, t_range, electrons)  # birth time
    # not time-dependent for now. needs to be to account for chirp.
    E = rejection_sampling(EX, E_range, electrons) - EB  # in eV
    E *= const.e  # now in Joules
    px, py, pz = spherical_to_cartesian(1, ψ, ϑ)
    r = np.zeros((electrons, 3)) + 1e-24
    return ClassicalElectrons(r, np.stack((px, py, pz)), E, t0)

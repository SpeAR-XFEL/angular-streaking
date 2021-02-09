import numpy as np
from numpy import pi as π 
import scipy.constants as const
from streaking.electrons import ClassicalElectrons
from streaking.stats import rejection_sampling
from streaking.conversions import spherical_to_cartesian, cartesian_to_spherical

import matplotlib.pyplot as plt

def diff_cross_section_dipole(ϑ, β):
    return 1 + β * 1 / 2 * (3 * np.cos(ϑ) ** 2 - 1)

def diff_cross_section_dipole2(ϑ, β):
    return 1 + β * 1 / 2 * (3 * np.sin(ϑ) ** 2 - 1)


def ionizer_simple(β, IX, EX, EB, E_range, t_range, electrons):
    """
    Generate randomly distributed photoelectrons
    """
    ϑ = rejection_sampling(diff_cross_section_dipole, (-π, π), electrons, (β,))
    ψ = np.random.uniform(-π, π, electrons)
    t0 = rejection_sampling(IX, t_range, electrons)  # birth time
    # not time-dependent for now. needs to be to account for chirp.
    E = rejection_sampling(EX, E_range, electrons) - EB  # in eV
    E *= const.e  # now in Joules
    px, py, pz = spherical_to_cartesian(1, ψ, ϑ)
    r = np.zeros((electrons, 3)) + 1e-24
    return ClassicalElectrons(r, np.stack((px, py, pz)), E, t0)

def diff_cross_section_Sauter(ϑ, ψ, β, ɣ):
    constant=3/(3*π*ɣ**4) * np.sin(ϑ)**2/(1-β*np.cos(ϑ))**4
    factor1=(1+3/4*ɣ*(ɣ-2)/(ɣ+1) * (1-1/(2*β*ɣ**2)*np.log((1+β)/(1-β))))**-1
    factor2=np.cos(ψ)**2*(1-ɣ/2*(ɣ-1)*(1-β*np.cos(ϑ)))+ɣ/4*(ɣ-1)**2*(1-β*np.cos(ϑ))
    return constant*factor1*factor2

import numpy as np
import scipy.constants as const
from streaking.components.conversions import cartesian_to_spherical


class ClassicalElectrons:
    """
    Holds a set of electrons. Maybe this should become a `Particles` class at some point?
    """

    def __init__(self, r, p, Ekin=None, t0=None):
        """
        Parameters
        ----------
        r : (N, 3) array_like
            Position vectors in meter.
        p : (N, 3) array_like
            Momentum vectors in kilogram meter per second.
        Ekin : (N) array_like, optional
            If given, p is taken as unit vector of momentum.
        t0 : (N) array_like, optional
            Optional birth time. If not given, every electron is born at t0 = 0.
        """
        self.r = r
        if Ekin is None:
            self.p = p
        else:
            pmag = np.sqrt(2 * const.m_e * const.c ** 2 * Ekin + Ekin ** 2) / const.c
            self.p = pmag * (p / np.linalg.norm(p, axis=0))

        if t0 is None:
            self.t0 = np.zeros(self.r.shape[0])
        else:
            self.t0 = t0

    def Ekin(self):
        pmag = np.linalg.norm(self.p, axis=0)
        E0 = const.m_e * const.c ** 2
        return np.sqrt(E0 ** 2 + (pmag * const.c) ** 2) - E0

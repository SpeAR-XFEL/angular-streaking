import numpy as np
import scipy.constants as const


class ClassicalElectrons:
    """
    Holds a set of electrons. Maybe this should become a `ClassicalParticles` class at some point?
    """

    def __init__(self, r, p, Ekin=None, t0=None):
        """
        Parameters
        ----------
        r : array_like, shape (N, 3)
            Position vectors in meter.
        p : array_like, shape (N, 3)
            Momentum vectors in kilogram meter per second.
        Ekin : array_like, shape (N,), optional
            If given, p is taken as unit vector of momentum.
        t0 : array_like, shape (N,), optional
            Electron birth times. If not given, every electron is born at t0 = 0.
        """
        self.r = np.asarray(r)
        p = np.asarray(p)

        if self.r.shape != p.shape:
            raise ValueError("r and p need to have the same shape")
        if Ekin is None:
            self.p = p
        else:
            Ekin = np.atleast_1d(Ekin)
            pmag = np.sqrt(2 * const.m_e * const.c ** 2 * Ekin + Ekin ** 2) / const.c
            self.p = pmag[:, None] * (p / np.linalg.norm(p, axis=1)[:, None])

        if t0 is None:
            self.t0 = np.zeros(self.r.shape[0])
        else:
            self.t0 = t0

    def Ekin(self):
        """
        Calculates the kinetic energy in Joules for every electron.

        Returns
        -------
        Ekin : ndarray, shape (N,)
        """
        pmag = np.linalg.norm(self.p, axis=1)
        E0 = const.m_e * const.c ** 2
        return np.sqrt(E0 ** 2 + (pmag * const.c) ** 2) - E0

    def gamma(self):
        """
        Calculates the relativistic gamma factor for every electron.

        Returns
        -------
        gamma : ndarray, shape (N,)
        """
        return np.sqrt(
            1 + (np.linalg.norm(self.p, axis=0) / (const.m_e * const.c)) ** 2
        )

    def v(self):
        """
        Calculates the relativistic velocity for every electron.

        Returns
        -------
        v : ndarray, shape (N,)
        """
        return self.p / (const.m_e * self.gamma())

    def __len__(self):
        """
        Returns the number of electrons in the object.

        Returns
        -------
        len : int
        """
        return self.r.shape[0]

    def __iadd__(self, other):
        self.r = np.concatenate((self.r, other.r))
        self.p = np.concatenate((self.p, other.p))
        self.t0 = np.concatenate((self.t0, other.t0))
        return self

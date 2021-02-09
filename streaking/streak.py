import numpy as np
import scipy.constants as const
import scipy.integrate
from streaking.electrons import ClassicalElectrons


def classical_lorentz_ode(t, y, m, q, t0, beam):
    r, p = y.reshape(2, -1, 3)
    E, B = beam.fields(*r.T, t + t0)
    # Can’t use ClassicalElectron methods here...
    pmag = np.linalg.norm(p, axis=0)
    E0 = m * const.c ** 2
    Etot = np.sqrt(E0 ** 2 + (pmag * const.c) ** 2)
    gamma = Etot / E0
    v = p / (gamma * m)
    drdt = v
    dpdt = q * (E + np.cross(v, B))
    return np.ravel((drdt, dpdt))


def classical_lorentz_streaker(electrons, beam, t_span):

    y0 = np.ravel((electrons.r, electrons.p))
    result = scipy.integrate.solve_ivp(
        classical_lorentz_ode,
        t_span,
        y0,
        args=(const.m_e, -const.e, electrons.t0, beam),
        vectorized=True,
        first_step=1e-19,
        method='RK23'
    )

    return ClassicalElectrons(*result.y[:, -1].reshape(2, -1, 3), t0=electrons.t0)

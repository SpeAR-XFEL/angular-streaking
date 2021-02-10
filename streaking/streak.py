import numpy as np
import scipy.constants as const
import scipy.integrate
from streaking.electrons import ClassicalElectrons


def rk4(fun, t_span, y0, max_step, args=None):
    h = max_step
    y = y0
    t = t_span[0]

    f = lambda t, y, fun=fun: np.array(fun(t, y, *args))

    for i in range(int((t_span[1] - t_span[0]) / h)):
        t = i * h
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def classical_lorentz_ode(t, y, m, q, t0, beam):
    r, p = y  # y.reshape(2, -1, 3)
    E, B = beam.fields(*r.T, t + t0)
    # Can’t use ClassicalElectron methods here...
    pmag = np.linalg.norm(p, axis=0)
    E0 = m * const.c ** 2
    Etot = np.sqrt(E0 ** 2 + (pmag * const.c) ** 2)
    gamma = Etot / E0
    v = p / (gamma * m)
    drdt = v
    dpdt = q * (E + np.cross(v, B))
    return (drdt, dpdt)


def classical_lorentz_streaker(electrons, beam, t_span, t_step):
    y0 = (electrons.r, electrons.p)
    result = rk4(
        classical_lorentz_ode,
        t_span,
        y0,
        t_step,
        args=(const.m_e, -const.e, electrons.t0, beam),
    )

    return ClassicalElectrons(*result, t0=electrons.t0)


def F_lor(electrons, E, B):
    F = -const.e * (E + np.cross(electrons.p, B) / const.m_e / electrons.gamma())
    return F


def interaction_step(electrons, beam, t, t_step):
    E, B = beam.fields(*electrons.r.T, t)
    F = F_lor(electrons, E, B)
    electrons.p = electrons.p + F * t_step
    electrons.r = electrons.r + electrons.v() * t_step


def Arne_streaker(electrons, beam, t0, t_step, N_step):
    t = t0
    for i in range(N_step):
        interaction_step(electrons, beam, t + electrons.t0, t_step)
        t = t + t_step
    return electrons

import numpy as np
import scipy.constants as const
from streaking.electrons import ClassicalElectrons
from numpy import pi as π
from multiprocessing import Pool
from functools import partial
import os


def dumb_streaker(electrons, beam, dumb=True, return_A_kick=False):
    if dumb:
        A = beam.vector_potential(*electrons.r.T, electrons.t0)
    else:
        A = beam.vector_potential_Arne(*electrons.r.T, electrons.t0)
    if return_A_kick:
        return ClassicalElectrons(electrons.r, electrons.p + const.e * A), np.linalg.norm(A, axis=1).max() * const.e
    else:
        return ClassicalElectrons(electrons.r, electrons.p + const.e * A)


def _rk4(fun, t_span, y0, max_step, args=None):
    h = max_step
    y = y0
    t = t_span[0]

    def f(t, y):
        return np.asarray(fun(t, y, *args))

    for i in range(int((t_span[1] - t_span[0]) / h)):
        t = i * h
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def _classical_lorentz_ode(t, y, m, q, beam, t0):
    r, p = y
    E = beam.field(*r.T, t + t0)
    # Can’t use ClassicalElectron methods here...
    pmag = np.linalg.norm(p, axis=1)
    E0 = m * const.c ** 2
    Etot = np.sqrt(E0 ** 2 + (pmag * const.c) ** 2)
    gamma = Etot / E0
    v = p / (gamma[:, None] * m)
    drdt = v
    # Triple cross product optimization (a × (b × c) = (a ∙ c) ∙ b - (a ∙ b) ∙ c)
    var = - v[:, 2][:, None] * E
    var[:, 2] += np.einsum('ij,ij->i', v, E)
    dpdt = q * (E + (var / const.c))
    return (drdt, dpdt)


def _classical_lorentz_solve(t_span, t_step, args, rest):
    r, p, t = rest
    return _rk4(_classical_lorentz_ode, t_span, (r, p), t_step, args=(*args, t))


def classical_lorentz_streaker(
    electrons, beam, t_span, t_step, processes=int(1.7 * os.cpu_count())
):
    """
    Simulates the streaking interaction between `electrons` and `beam` in the time range
    `t_span` in time steps of `t_step` using the Lorentz force and a Runge-Kutta solver.
    Uses multiprocessing to utilize multicore processors.

    Parameters
    ----------
    electrons : ClassicalElectrons
        Set of electrons to be streaked
    beam : SimpleGaussianBeam
        Beam the electrons are streaked with
    t_span : array_like, shape (2,)
        Integration timespan in seconds
    t_step : scalar
        Integration step size in seconds
    processes: int
        Number of processes to start, optional, defaults to `int(1.7 * os.cpu_count())`
    """
    func = partial(
        _classical_lorentz_solve, t_span, t_step, (const.m_e, -const.e, beam)
    )
    splitr = np.array_split(electrons.r, processes)
    splitp = np.array_split(electrons.p, processes)
    splitt = np.array_split(electrons.t0, processes)
    with Pool(processes) as p:
        res = p.map(func, [z for z in zip(splitr, splitp, splitt)])
    #res = [func(z) for z in zip(splitr, splitp, splitt)]
    result = np.concatenate(res, axis=1)
    return ClassicalElectrons(*result, t0=electrons.t0)


# ---------------
# Here be dragons


def _interaction_step(electrons, beam, t, t_step):
    E = beam.field(*electrons.r.T, t)
    B = np.cross([0, 0, 1], E) / const.c
    F = -const.e * (E + np.cross(electrons.p, B) / const.m_e / electrons.gamma())
    electrons.p = electrons.p + F * t_step
    electrons.r = electrons.r + electrons.v() * t_step


def Arne_streaker(electrons, beam, t0, t_step, N_step):
    t = t0
    for i in range(N_step):
        _interaction_step(electrons, beam, t + electrons.t0, t_step)
        t = t + t_step
    return electrons


def _F_space_charge(electrons):
    """
    calculates the force vector on each electron caused by space charge

    Parameters
    ----------
    electrons : class ClassicalElectrons

    Returns
    -------
    F_sc : (N,3) array_like
        array of force vectors acting on each electron caused by space charge of all other electrons

    """

    x1, x2 = np.meshgrid(electrons.r[:, 0], electrons.r[:, 0])
    y1, y2 = np.meshgrid(electrons.r[:, 1], electrons.r[:, 1])
    z1, z2 = np.meshgrid(electrons.r[:, 2], electrons.r[:, 2])

    # (N,N,3) array_like where R[i,j,:] points from particle i to particle j
    R = np.stack([x2 - x1, y2 - y1, z2 - z1], axis=2)
    # get rid of diagonal elements, since an particle doesn't act on itself
    N = R.shape[0]
    mask = np.stack(
        [np.eye(N, dtype=bool), np.eye(N, dtype=bool), np.eye(N, dtype=bool)], axis=2
    )
    R = R[~mask].reshape(N, N - 1, 3)
    R_abs = np.linalg.norm(R, axis=2)

    print(R.shape)
    print(R_abs.shape)
    F_sc = (
        const.e ** 2
        / (4 * π * const.epsilon_0)
        * np.sum(R / R_abs[:, :, np.newaxis] ** 3, axis=1)
    )
    return F_sc

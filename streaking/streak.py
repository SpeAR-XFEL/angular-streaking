import numpy as np
import scipy.constants as const
import scipy.integrate
from streaking.electrons import ClassicalElectrons
from numpy import pi as π
from multiprocessing import Pool
from functools import partial
import os


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


def classical_lorentz_ode(t, y, m, q, beam, t0):
    r, p = y 
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


def classical_lorentz_solve(t_span, t_step, args, rest):
    r, p, t = rest
    return rk4(classical_lorentz_ode, t_span, (r, p), t_step, args=(*args, t))


def classical_lorentz_streaker(electrons, beam, t_span, t_step):
    y0 = (electrons.r, electrons.p)
    ecount = electrons.r.shape[0]
    func = partial(classical_lorentz_solve, t_span, t_step, (const.m_e, -const.e, beam))
    threads = os.cpu_count()
    splitr = np.array_split(electrons.r, threads)
    splitp = np.array_split(electrons.p, threads)
    splitt = np.array_split(electrons.t0, threads)
    with Pool(threads) as p:
        res = p.map(func, [z for z in zip(splitr, splitp, splitt)])
    result = np.concatenate(res, axis=1)
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


def F_space_charge(electrons):
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

    x1,x2=np.meshgrid(electrons.r[:,0],electrons.r[:,0])
    y1,y2=np.meshgrid(electrons.r[:,1],electrons.r[:,1])
    z1,z2=np.meshgrid(electrons.r[:,2],electrons.r[:,2])

    # (N,N,3) array_like where R[i,j,:] points from particle i to particle j 
    R=np.stack([x2-x1,y2-y1,z2-z1],axis=2)
    # get rid of diagonal elements, since an particle doesn't act on itself
    N=R.shape[0]
    mask=np.stack([np.eye(N,dtype=bool),np.eye(N,dtype=bool),np.eye(N,dtype=bool)],axis=2)
    R=R[~mask].reshape(N,N-1,3)
    R_abs=np.linalg.norm(R,axis=2)

    print(R.shape)
    print(R_abs.shape)
    F_sc = const.e**2/(4*π*const.epsilon_0) * np.sum(R/R_abs[:,:,np.newaxis]**3,axis=1)
    return F_sc
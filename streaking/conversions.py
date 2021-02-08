import numpy as np


def spherical_to_cartesian(r, φ, ϑ):
    x = r * np.sin(ϑ) * np.cos(φ)
    y = r * np.sin(ϑ) * np.sin(φ)
    z = r * np.cos(ϑ)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    r = np.linalg.norm((x, y, z), axis=0)
    φ = np.arctan2(y, x)
    ϑ = np.arccos(z / r)
    return r, φ, ϑ

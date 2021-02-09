import numpy as np


def spherical_to_cartesian(r, ϑ, φ):
    x = r * np.sin(ϑ) * np.cos(φ)
    y = r * np.sin(ϑ) * np.sin(φ)
    z = r * np.cos(ϑ)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    φ = np.arctan2(y, x)
    ϑ = np.arctan2(hxy, z)
    return r, ϑ, φ

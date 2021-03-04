import numpy as np


def spherical_to_cartesian(r, ϑ, φ):
    """
    Convert from spherical to cartesian coordinates.
    Origin: (0,0,0), Polar axis (ϑ=0): +z, Equatorial axis (φ=0): +x

    Parameters
    ----------
    r : array_like
        Radius, between 0 and infinity
    ϑ : array_like
        Polar angle, between 0 and π (0 is +z pole, π is -z pole)
    φ : array_like
        Azimuthal angle, between 0 and π (0 is +x, counter-clockwise seen from +z)

    Returns
    -------
    x : array_like
        Cartesian x coordinate
    y : array_like
        Cartesian y coordinate
    z : array_like
        Cartesian z coordinate
    """
    x = r * np.sin(ϑ) * np.cos(φ)
    y = r * np.sin(ϑ) * np.sin(φ)
    z = r * np.cos(ϑ)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Convert from cartesian to spherical coordinates.
    Origin: (0,0,0), Polar axis (ϑ=0): +z, Equatorial axis (φ=0): +x

    Parameters
    ----------
    x : array_like
        Cartesian x coordinate
    y : array_like
        Cartesian y coordinate
    z : array_like
        Cartesian z coordinate

    Returns
    -------
    r : array_like
        Radius, between 0 and infinity
    ϑ : array_like
        Polar angle, between 0 and π (0 is +z pole, π is -z pole)
    φ : array_like
        Azimuthal angle, between 0 and π (0 is +x, counter-clockwise seen from +z)
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    φ = np.arctan2(y, x)
    ϑ = np.arctan2(hxy, z)
    return r, ϑ, φ

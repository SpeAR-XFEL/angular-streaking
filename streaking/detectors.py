import numpy as np
import scipy.constants as const
from streaking.conversions import cartesian_to_spherical


def _sphere_intersection(electrons, radius):
    """
    Calculates where the electron trajectories intersect the sphere of radius `radius` in spherical coordinates.
    Uniform motion is assumed.
    """
    # There might be an easier way.
    # Currently I solve |r̲ + d * p̲| = radius for d (positive solution),
    # then r̲ + d * p̲ is the point of intersection.
    px, py, pz = electrons.p.T
    x, y, z = electrons.r.T
    rsq = x ** 2 + y ** 2 + z ** 2
    assert np.all(rsq < radius ** 2), "Electron position outside detector volume"
    psq = px ** 2 + py ** 2 + pz ** 2
    xpx = x * px + y * py + z * pz
    d = (-2 * xpx + np.sqrt(4 * xpx ** 2 - 4 * psq * (rsq - radius ** 2))) / (2 * psq)
    intersection_point = electrons.r + d[:, None] * electrons.p
    return cartesian_to_spherical(*intersection_point.T)


# TODO: Implement origin+normal vector or transformation matrix or something else for both variants


def constant_polar_angle_tofs(
    electrons,
    polar_center,
    polar_acceptance,
    azimuth_bins,
    variable,
    variable_bins,
    variable_quantile,
    radius,
    origin,
    normal_vector,
):
    """ Calling this 'tofs' might be a slight overstatement..."""

    r, theta, phi = _sphere_intersection(electrons, radius)
    mask = np.abs((theta - polar_center)) < polar_acceptance
    phibins = np.linspace(0, 2 * np.pi, azimuth_bins + 1)
    variable_bins = np.asarray(variable_bins)

    if variable == "kinetic energy":
        v = electrons.Ekin()[mask] / const.e
    elif variable == "momentum":
        v = 3e25 * np.linalg.norm(electrons.p, axis=1)[mask]
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    # If only bin count is given: Define bins using quantile
    if np.ndim(variable_bins) == 0:
        variable_bins = np.linspace(
            *np.quantile(v, (variable_quantile / 2, 1 - variable_quantile / 2)),
            variable_bins,
        )

    return np.histogram2d(
        (phi[mask] + np.pi / 2) % (2 * np.pi), v, bins=(phibins, variable_bins)
    )


def energy_integrated_4pi(electrons, polar_bins, azimuth_bins, radius, origin):
    r, theta, phi = _sphere_intersection(electrons, radius)

    b1 = np.linspace(0, 2 * np.pi, azimuth_bins + 1)
    b2 = np.arcsin(np.linspace(-1, 1, polar_bins + 1)) + np.pi / 2

    return np.histogram2d((phi + np.pi / 2) % (2 * np.pi), theta, bins=(b1, b2))

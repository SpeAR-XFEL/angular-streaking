import numpy as np
import scipy.constants as const
from streaking.conversions import cartesian_to_spherical
from scipy.spatial.transform import Rotation as R

def _cylinder_intersection(electrons, radius, origin=(0, 0, 0), rotation=None):
    origin = np.asarray(origin)[:, None]

    if rotation is not None:
        ep = rotation.inv().apply(electrons.p)
        er = rotation.inv().apply(electrons.r)
    else:
        ep = electrons.p
        er = np.copy(electrons.r)

    er -= origin.T

    px, py, pz = ep.T
    x, y, z = er.T
    rsq = x ** 2 + y ** 2
    assert np.all(rsq < radius ** 2), "Electron position outside detector volume"
    psq = px ** 2 + py ** 2
    xpx = x * px + y * py
    d = (-2 * xpx + np.sqrt(4 * xpx ** 2 - 4 * psq * (rsq - radius ** 2))) / (2 * psq)
    intersection_point = er + d[:, None] * ep
    return *(cartesian_to_spherical(*intersection_point.T)[(0, 2), :]), intersection_point[:, 2]


def _sphere_intersection(electrons, radius, origin=(0, 0, 0), rotation=None, cartesian=False):
    """
    Calculates where the electron trajectories intersect the sphere of radius `radius` in spherical coordinates.
    Uniform motion is assumed.
    """
    # There might be an easier way.
    # Currently I solve |r̲ + d * p̲| = radius for d (positive solution),
    # then r̲ + d * p̲ is the point of intersection.

    origin = np.asarray(origin)[:, None]

    if rotation is not None:
        ep = rotation.inv().apply(electrons.p)
        er = rotation.inv().apply(electrons.r)
    else:
        ep = electrons.p
        er = np.copy(electrons.r)

    er -= origin.T

    px, py, pz = ep.T
    x, y, z = er.T
    rsq = x ** 2 + y ** 2 + z ** 2
    assert np.all(rsq < radius ** 2), "Electron position outside detector volume"
    psq = px ** 2 + py ** 2 + pz ** 2
    xpx = x * px + y * py + z * pz
    d = (-2 * xpx + np.sqrt(4 * xpx ** 2 - 4 * psq * (rsq - radius ** 2))) / (2 * psq)
    intersection_point = er + d[:, None] * ep
    if cartesian:
        return intersection_point
    else:
        return cartesian_to_spherical(*intersection_point.T)


def cylindrical_energy_resolved(
    electrons,
    z_width,
    azimuth_bins,
    radius,
    variable,
    variable_bins,
    variable_quantile=0.01,
    origin=(0, 0, 0),
    rotation=None,
):
    r, phi, z = _cylinder_intersection(electrons, radius, origin, rotation)
    mask = np.abs(z) < z_width / 2
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
            variable_bins + 1,
        )

    return np.histogram2d(
        (phi[mask] + np.pi / 2) % (2 * np.pi), v, bins=(phibins, variable_bins)
    )


def constant_polar_angle_ring(
    electrons,
    polar_center,
    polar_acceptance,
    azimuth_bins,
    radius,
    variable,
    variable_bins,
    variable_quantile=0.01,
    origin=(0, 0, 0),
    rotation=None,
):
    r, theta, phi = _sphere_intersection(electrons, radius, origin, rotation)
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
            variable_bins + 1,
        )

    return np.histogram2d(
        (phi[mask]) % (2 * np.pi), v, bins=(phibins, variable_bins)
    )


def energy_integrated_4pi(electrons, polar_bins, azimuth_bins, radius, origin):
    r, theta, phi = _sphere_intersection(electrons, radius, origin=origin)

    b1 = np.linspace(0, 2 * np.pi, azimuth_bins + 1)
    b2 = np.arcsin(np.linspace(-1, 1, polar_bins + 1)) + np.pi / 2

    return np.histogram2d((phi) % (2 * np.pi), theta, bins=(b1, b2))


def flight_tubes(electrons, angles, tip_distance, tip_diameter, mcp_distance, mcp_diameter, variable, variable_bins, variable_quantile=0.01):
    tip = _sphere_intersection(electrons, tip_distance, cartesian=True)
    mcp = _sphere_intersection(electrons, mcp_distance, cartesian=True)

    # Rotate around origin such that detectors lie in polar / z direction
    rot = R.from_euler('YZ', np.stack((np.full_like(angles, np.pi / 2), angles)).T)
    tip = np.einsum('ijk,lk->jil', rot.as_matrix(), tip)
    mcp = np.einsum('ijk,lk->jil', rot.as_matrix(), mcp)
    _, tip_theta, _ = cartesian_to_spherical(*tip)
    _, mcp_theta, _ = cartesian_to_spherical(*mcp)
    tip_mask = tip_theta < np.arctan(0.5 * tip_diameter / tip_distance)
    mcp_mask = mcp_theta < np.arctan(0.5 * mcp_diameter / mcp_distance)
    comp_mask = tip_mask & mcp_mask

    if variable == "kinetic energy":
        v = electrons.Ekin() / const.e
    elif variable == "momentum":
        v = 3e25 * np.linalg.norm(electrons.p, axis=1)
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    # If only bin count is given: Define bins using quantile
    if np.ndim(variable_bins) == 0:
        variable_bins = np.linspace(
            *np.quantile(v, (variable_quantile / 2, 1 - variable_quantile / 2)),
            variable_bins + 1,
        )
    spectra = [np.histogram(v[mask], bins=variable_bins)[0] for mask in comp_mask]
    #print(tip_mask.sum(), mcp_mask.sum(), comp_mask.sum())
    # print(tip_theta)
    
    return np.array(spectra), np.linspace(0, 2 * np.pi, len(angles) + 1), variable_bins
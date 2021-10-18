from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.streak import dumb_streaker
from streaking.detectors import constant_polar_angle_ring, _cylinder_intersection
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.electrons import ClassicalElectrons
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

if __name__ == "__main__":
    number_of_electrons = 900000
    binding_energy = 870.2  # eV
    β = 2
    xfel_duration = 1e-15  # s
    xfel_energy = 960  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m

    xfel_time_energy_means = (0, xfel_energy)
    xfel_time_energy_covariance = np.diag((xfel_duration, xfel_energy_std)) ** 2

    TEmap = MultivariateMapInterpolator.from_gauss_blob_list(
        (xfel_time_energy_means,),
        (xfel_time_energy_covariance,),
        (1,)
    )

    pe = ionizer_simple(  # pe: photoelectrons
        β,
        TEmap,
        xfel_focal_spot,
        binding_energy,
        number_of_electrons,
        1e-4
    )

    streaking_beam = SimpleGaussianBeam(
        focal_size=(500e-6, 500e-6),
        envelope_offset=0,
        cep=np.pi / 3,
        wavelength=10e-6,
        energy=500e-6,
        duration=300e-15)

    streaked_pe = dumb_streaker(pe, streaking_beam)

    #for rotang in np.linspace(0, 67.5, 4):
    #    rot = scipy.spatial.transform.Rotation.from_euler('Z', rotang, degrees=True)
    #    r, phi, z = _cylinder_intersection(streaked_pe, 0.5)
    #    reduced = streaked_pe[np.logical_and(z < 20e-3, z > - 20e-3)]
    #    print(reduced.r.shape)
    #    rotated = ClassicalElectrons(rot.apply(reduced.r), rot.apply(reduced.p))
    #    rotated.cst_export(f'simulations/build/streaked_photoelectrons_{rotang:.1f}.pid')

    histogram, bins_phi, bins_E = constant_polar_angle_ring(pe, np.pi / 2, 0.5, 16*64, 0.25, 'kinetic energy', 16*64)
    histogram_s, bins_phi_s, bins_E_s = constant_polar_angle_ring(streaked_pe, np.pi / 2, 0.5, 16, 0.25, 'kinetic energy', 264)

    plt.figure(figsize=(4, 3))
    #plt.subplot(121)
    #plt.imshow(histogram.T, aspect='auto', origin='lower', interpolation='none', extent=(bins_phi[0], bins_phi[-1], bins_E[0], bins_E[-1]))
    #plt.xlabel(r"$\varphi$")
    #plt.ylabel(r"$E_\mathrm{kin}$ / eV")
    #plt.subplot(122)
    plt.imshow(histogram_s.T, aspect='auto', cmap='turbo', origin='lower', interpolation='none', extent=(bins_phi_s[0], bins_phi_s[-1], bins_E_s[0], bins_E_s[-1]))
    plt.xlabel(r"Winkel $\phi$ in rad", fontsize=10)
    plt.ylabel(r"Kinetische Energie $E$ in eV", fontsize=10)
    plt.colorbar()
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('simulations/build/niclas.pdf', bbox_inches='tight')

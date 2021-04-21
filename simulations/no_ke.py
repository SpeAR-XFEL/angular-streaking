from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import dumb_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.stats import covariance_from_correlation_2d
from streaking.detectors import constant_polar_angle_ring
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from p_tqdm import p_map


def streaking_sim(xfel_duration, cep, phibins, peaks, energy):
    number_of_electrons = 200000
    binding_energy = 870.2  # eV
    β = 2
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m

    theta_center = np.pi / 4
    theta_acceptance = np.pi / 10

    if peaks == 1:
        xfel_time_energy_means = (0, xfel_energy)
        xfel_time_energy_covariance = np.diag((xfel_duration, xfel_energy_std)) ** 2
        pe = ionizer_simple(  # pe: photoelectrons
            β,
            xfel_time_energy_means,
            xfel_time_energy_covariance,
            binding_energy,
            xfel_focal_spot,
            number_of_electrons,
        )
    elif peaks == 2:
        twopk_dist = xfel_duration
        dur = 1e-15
        sigE = xfel_energy_std
        muE = xfel_energy
        mu_t = (-twopk_dist / 2, twopk_dist / 2)
        mu_E = (muE, muE)  # eV
        sigma_t = (dur, dur)
        sigma_E = (sigE, sigE)
        corr_list = (0, 0)
        I_list = (0.5, 0.5)

        covs = covariance_from_correlation_2d(np.stack((sigma_t, sigma_E)), corr_list).T
        TEmap = MultivariateMapInterpolator.from_gauss_blob_list(np.stack((mu_t, mu_E)).T, covs, I_list)

        pe = ionizer_simple(2, TEmap, 1e-5, binding_energy, number_of_electrons)
    else:
        raise ValueError("Unsupported peak count")

    streaking_beam = SimpleGaussianBeam(
        focal_size=(5e-4, 5e-4),
        envelope_offset=0,
        cep=cep,
        wavelength=10e-6,
        energy=energy,
        duration=300e-15)
    streaked_pe = dumb_streaker(pe, streaking_beam)
    hist, x, y = constant_polar_angle_ring(streaked_pe, np.pi/2, 0.2, phibins, 0.25, 'kinetic energy', 1, 0.0)

    return hist[:, 0]


def make_image_cep(cep):
    durations = np.linspace(0e-15, 15e-15, 40)
    hist = [streaking_sim(duration, cep, 64, 2) for duration in tqdm(durations)]
    plt.imshow(hist, origin='lower', aspect='auto', extent=(0, 2 * np.pi, 1e15 * durations[0], 1e15 * durations[-1]))
    plt.title(f'Angular distribution of streaked electrons (energy integrated)\n at 200µJ & 100µm focus, CEP = {cep:.2f}, double pulses (1fs each)')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'XFEL double pulse separation / fs')
    plt.tight_layout(pad=0.5)
    plt.savefig(f'simulations/build/no_ke_new_2021_04_18/{cep:.2f}.png', dpi=400)
    plt.close()


def make_image_energy(energy):
    durations = np.linspace(0e-15, 15e-15, 40)
    hist = p_map(lambda duration: streaking_sim(duration, np.pi, 64, 2, energy), durations)
    plt.figure(figsize=(15, 5))
    plt.imshow(hist, origin='lower', aspect='auto', extent=(0, 2 * np.pi, 1e15 * durations[0], 1e15 * durations[-1]))
    plt.title(f'Angular distribution of streaked electrons from 1$\,$fs double pulses \n at {1e6*energy:.0f}$\,$µJ streaking pulse energy & 500$\,$µm focus')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'XFEL double pulse separation / fs')
    plt.tight_layout(pad=0.5)
    plt.savefig(f'simulations/build/no_ke_new_2021_04_18/{1e6*energy:.0f}.png', dpi=400)
    plt.close()


if __name__ == "__main__":
    [make_image_energy(energy) for energy in tqdm((30e-6,))]
    #[make_image_dur(dur) for dur in tqdm(np.linspace(10e-15, 20e-15, 20))]

from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple, ionizer_Sauter
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.stats import covariance_from_correlation_2d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def streaking_sim(xfel_duration, cep, phibins, peaks):
    number_of_electrons = 25000
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

        pe = ionizer_Sauter(TEmap, binding_energy, number_of_electrons)
    else:
        raise ValueError("Unsupported peak count")

    streaking_beam = SimpleGaussianBeam(
        focal_size=(1e-4, 1e-4),
        envelope_offset=0,
        cep=cep,
        wavelength=10e-6,
        energy=200e-6,
        duration=300e-15)
    streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, 1e-12), 1e-14, processes=21)
    sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    mask = np.abs(stheta - theta_center) < theta_acceptance
    sphi = (sphi[mask] + np.pi / 2) % (2 * np.pi)
    return np.histogram(sphi, bins=phibins)[0]


def make_image_cep(cep):
    phibins = np.linspace(0, 2 * np.pi, 64 + 1)
    durations = np.linspace(0e-15, 15e-15, 40)
    hist = [streaking_sim(duration, cep, phibins, 2) for duration in tqdm(durations)]
    plt.imshow(hist, origin='lower', aspect='auto', extent=(phibins[0], phibins[-1], 1e15 * durations[0], 1e15 * durations[-1]))
    plt.title(f'Angular distribution of streaked electrons (energy integrated)\n at 200µJ & 100µm focus, CEP = {cep:.2f}, double pulses (1fs each)')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'XFEL double pulse separation / fs')
    plt.tight_layout(pad=0.5)
    plt.savefig(f'simulations/build/no_ke_cep_2pk_forward_200uJ/{cep:.2f}.png', dpi=400)
    plt.close()


def make_image_dur(dur):
    phibins = np.linspace(0, 2 * np.pi, 64 + 1)
    ceps = np.linspace(0, 2 * np.pi, 40)
    hist = [streaking_sim(dur, cep, phibins, 2) for cep in tqdm(ceps)]
    plt.imshow(hist, origin='lower', aspect='auto', extent=(phibins[0], phibins[-1], ceps[0], ceps[-1]))
    plt.title(f'Angular distribution of streaked electrons (energy integrated)\n at 200µJ & 100µm focus, double pulses (1fs each), separation {1e15*dur:.2f} fs')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'CEP')
    plt.tight_layout(pad=0.5)
    plt.savefig(f'simulations/build/no_ke_dur_2pk_forward_200uJ/{dur*1e15:.2f}.png', dpi=400)
    plt.close()


if __name__ == "__main__":
    [make_image_cep(cep) for cep in tqdm(np.linspace(0, 2 * np.pi, 40))]
    #[make_image_dur(dur) for dur in tqdm(np.linspace(10e-15, 20e-15, 20))]

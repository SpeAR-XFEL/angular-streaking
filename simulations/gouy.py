from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import dumb_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.stats import covariance_from_correlation_2d
from streaking.detectors import cylindrical_energy_resolved
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from p_tqdm import p_map


def streaking_sim(xfel_duration, cep, phibins, peaks, positions):
    number_of_electrons = 800000
    binding_energy = 870.2  # eV
    β = 2
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m

    if peaks == 1:
        xfel_time_energy_means = (0, xfel_energy)
        xfel_time_energy_covariance = np.diag((xfel_duration, xfel_energy_std)) ** 2
        # TEmap = MultivariateMapInterpolator.from_gauss_blob_list()
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
    else:
        raise ValueError("Unsupported peak count")

    pe = ionizer_simple(β, TEmap, xfel_focal_spot, binding_energy, number_of_electrons, 0.1)

    streaking_beam = SimpleGaussianBeam(
        focal_size=(1e-4, 1e-4),
        envelope_offset=0,
        cep=cep,
        wavelength=10.6e-6,
        energy=5e-4,
        duration=300e-15)
    streaked_pe = dumb_streaker(pe, streaking_beam)

    width = 0.001
    radius = 0.01
    result = [cylindrical_energy_resolved(streaked_pe, width, phibins, radius, 'momentum', 1, origin=(0, 0, z))[0].T for z in positions]

    return np.asarray(result).squeeze()


def plot(idx, pulse_separation):
    zrange = np.linspace(-0.05, 0.05, 100)
    a = streaking_sim(pulse_separation, np.pi, 128, 2, zrange)
    plt.imshow(a.T, origin='lower', aspect='auto', extent=(zrange[0] - 0.001, zrange[-1] + 0.001, 0, 2 * np.pi))
    plt.xlabel('detector z position / m')
    plt.ylabel('angle / rad')
    plt.title(f'100 energy integrated ring detectors along focus,\n {pulse_separation*1e15:.1f} fs double pulse, 500µJ streaking @ 100µm focus')
    plt.savefig(f'simulations/build/10gouy/{idx}.png', dpi=150)


if __name__ == "__main__":
    [plot(idx+10, dur) for idx, dur in tqdm(enumerate(np.linspace(20e-15, 40e-15, 10)))]

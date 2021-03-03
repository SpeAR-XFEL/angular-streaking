from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from tqdm import tqdm


def streaking_sim(xfel_duration, cep, phibins):
    number_of_electrons = 10000
    binding_energy = 870.2  # eV
    β = 2
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m

    phibincount = 32
    theta_center = np.pi/2
    theta_acceptance = np.pi/10

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

    streaking_beam = SimpleGaussianBeam(
        focal_size=(1e-4, 1e-4),
        envelope_offset=0,
        cep=cep,
        wavelength=10e-6,
        energy=1e-3,
        duration=300e-15)
    streaked_pe = classical_lorentz_streaker(pe, streaking_beam, (0, 1e-12), 1e-14)
    sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    mask = np.abs(stheta - theta_center) < theta_acceptance
    sphi = (sphi[mask] + np.pi / 2) % (2 * np.pi)
    return np.histogram(sphi, bins=phibins)[0]


def make_image_cep(cep):
    phibins = np.linspace(0, 2 * np.pi, 64 + 1)
    durations = np.linspace(1e-16, 5e-15, 40)
    ceps = np.linspace(0, 2 * np.pi, 40)
    hist = [streaking_sim(duration, cep, phibins) for duration in tqdm(durations)]
    plt.imshow(hist, origin='lower', aspect='auto', extent=(phibins[0], phibins[-1], 1e15*durations[0], 1e15*durations[-1]))
    plt.title(f'Angular distribution of streaked electrons (energy integrated)\n at 1mJ & 100µm focus, CEP = {cep:.2f}')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'XFEL duration sigma / fs')
    plt.tight_layout(pad=0.5)
    plt.savefig(f'simulations/build/no_ke/{cep:.2f}.png', dpi=200)
    plt.close()


def make_image_dur(dur):
    phibins = np.linspace(0, 2 * np.pi, 64 + 1)
    ceps = np.linspace(0, 2 * np.pi, 40)
    hist = [streaking_sim(dur, cep, phibins) for cep in tqdm(ceps)]
    plt.imshow(hist, origin='lower', aspect='auto', extent=(phibins[0], phibins[-1], ceps[0], ceps[-1]))
    plt.title(f'Angular distribution of streaked electrons (energy integrated)\n at 1mJ & 100µm focus, XFEL duration = {1e15*dur:.2f} fs')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'CEP')
    plt.tight_layout(pad=0.5)
    plt.savefig(f'simulations/build/no_ke_dur/{dur*1e15:.2f}.png', dpi=200)
    plt.close()

if __name__ == "__main__":
    [make_image_cep(cep) for cep in tqdm(np.linspace(0, 2 * np.pi, 20))]
    [make_image_dur(dur) for dur in tqdm(np.linspace(1e-16, 5e-15, 15))]

from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_Sauter, ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker, dumb_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.stats import covariance_from_correlation_2d
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import sys
from ruamel.yaml import YAML
from p_tqdm import p_map
from collections import OrderedDict
import copy


def sample_config(config):
    c2 = copy.deepcopy(config)
    sample_config_inplace(c2)
    return c2


def sample_config_inplace(config):
    for key, item in config.items():
        if isinstance(item, OrderedDict):
            if 'random' in item:
                if item['random'] == 'uniform':
                    config[key] = np.random.uniform(*item['args'])
                elif item['random'] == 'normal':
                    config[key] = np.random.normal(*item['args'])
            else:
                sample_config_inplace(item)


def simulate(delay):
    number_of_electrons = 200000
    binding_energy = 870.2  # eV
    β = 2
    xfel_duration = 1e-15  # s
    xfel_energy = 930  # eV
    xfel_energy_std = 1  # eV
    xfel_focal_spot = 2e-5  # m
    theta_center = np.pi / 2
    theta_acceptance = np.pi / 4

    #dur = xfel_duration
    #sigE = xfel_energy_std
    #tEmeans = (0, xfel_energy)
    #chirp = 0
    #tEcov = np.array(((dur**2, chirp * dur * sigE), (chirp * dur * sigE, sigE**2)))
    #pe = ionizer_simple(2, tEmeans, tEcov, binding_energy, xfel_focal_spot, number_of_electrons)

    N_G = 10
    mu_t = np.random.normal(0, 3e-15, N_G)  # s
    mu_E = np.random.normal(xfel_energy, 0.01, N_G)  # eV
    sigma_t = np.abs(np.random.normal(1e-15, 0.2e-15, N_G))
    sigma_E = np.abs(np.random.normal(xfel_energy_std, 0.1, N_G))
    corr_list = np.random.normal(0, 0, N_G)
    I_list = np.abs(np.random.normal(10, 1, N_G))

    covs = covariance_from_correlation_2d(np.stack((sigma_t, sigma_E)), corr_list).T
    TEmap = MultivariateMapInterpolator.from_gauss_blob_list(np.stack((mu_t, mu_E)).T, covs, I_list)

    pe = ionizer_Sauter(TEmap, binding_energy, number_of_electrons)


    streaking_beam = SimpleGaussianBeam(
        focal_size=(100e-6 / 2.3548, 100e-6 / 2.3548),
        envelope_offset=delay,
        cep=np.pi,
        wavelength=10.6e-6,
        energy=60e-6,
        duration=300e-15)

    detector_bins = np.linspace(0, 2 * np.pi, 16 + 1)
    energy_bins = np.arange(200 + 1)
    t_bins = np.linspace(-35.3e-15/2, 35.3e-15/2, 80 + 1)
    streaked_pe, kick = dumb_streaker(pe, streaking_beam, return_A_kick=True)
    sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    mask = np.abs(stheta - theta_center) < theta_acceptance
    #hist, x, y = np.histogram2d((sphi[mask] + np.pi / 2) % (2 * np.pi), 3e25 * np.linalg.norm(streaked_pe.p, axis=1)[mask], bins=(detector_bins, energy_bins))
    hist, x, y = np.histogram2d((sphi[mask] + np.pi / 2) % (2 * np.pi), streaked_pe.Ekin()[mask] / const.e, bins=(detector_bins, energy_bins))
    
    spec, _, _ = np.histogram2d(pe.t0, pe.Ekin()/const.e, bins=(t_bins, energy_bins))
    timedist = np.sum(spec, axis=1)
    return timedist, spec.T / spec.max(), hist.T / hist.max(), kick * 1e25


if __name__ == "__main__":
    yaml = YAML()
    with open('simulations/configs/default.yaml') as f:
        config = yaml.load(f)
    scfg = sample_config(config)
    print(config['physical parameters']['laser']['delay'])
    print(scfg['physical parameters']['laser']['delay'])
    
    #results = p_map(simulate, np.random.uniform(-400e-15, 400e-15, 100))
    #timedist, spectrograms, images, kick = list(map(list, zip(*results)))
    #with h5py.File("simulations/build/test.hdf5", "w") as f:
    #    f.create_dataset("detector_images", data=images)
    #    f.create_dataset("kick", data=kick)
    #    f.create_dataset("spectrograms", data=spectrograms)
    #    f.create_dataset("time_distribution", data=timedist)

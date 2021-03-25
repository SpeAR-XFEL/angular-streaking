from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_Sauter, ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker, dumb_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.stats import covariance_from_correlation_2d
import numpy as np
import scipy.constants as const
import h5py
import sys
from ruamel.yaml import YAML
from p_tqdm import p_map
from tqdm import tqdm
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
                config[key] = getattr(np.random, item['random'])(*item['args'])
            else:
                sample_config_inplace(item)


def simulate(config):
    pconf = config['physical parameters']
    number_of_electrons = pconf['target']['number of electrons']
    binding_energy = pconf['target']['binding energy']

    if pconf['xfel pulse generator'] == 'gaussian train':
        gconf = pconf['xfel pulse generator settings']
        N_G = gconf['number of peaks']
        mu_t = np.random.normal(gconf['temporal peak center'], gconf['temporal peak std'], N_G)
        mu_E = np.random.normal(gconf['central energy mean'], gconf['central energy std'], N_G)
        sigma_t = np.abs(np.random.normal(gconf['temporal peak width'], gconf['temporal peak width std'], N_G))
        sigma_E = np.abs(np.random.normal(gconf['energy width mean'], gconf['energy width std'], N_G))
        corr_list = np.random.normal(gconf['correlation mean'], gconf['correlation std'], N_G)
        I_list = np.abs(np.random.normal(gconf['intensity mean'], gconf['intensity std'], N_G))

        covs = covariance_from_correlation_2d(np.stack((sigma_t, sigma_E)), corr_list).T
        TEmap = MultivariateMapInterpolator.from_gauss_blob_list(np.stack((mu_t, mu_E)).T, covs, I_list)

        pe = ionizer_Sauter(TEmap, binding_energy, number_of_electrons)
    else:
        raise ValueError(f'Unsupported XFEL pulse generator: {pconf["xfel pulse generator"]}')

    streaking_beam = SimpleGaussianBeam(
        focal_size=(pconf['laser']['focal size x'], pconf['laser']['focal size y']),
        envelope_offset=pconf['laser']['delay'],
        cep=pconf['laser']['cep'],
        wavelength=pconf['laser']['wavelength'],
        energy=pconf['laser']['pulse energy'],
        duration=pconf['laser']['pulse duration'])

    detector_bins = np.linspace(0, 2 * np.pi, pconf['detector']['phi bins'] + 1)
    eb = pconf['detector']['energy bins']
    energy_bins = np.linspace(*eb) if isinstance(eb, list) else eb
    tb = config['other parameters']['time bins']

    t_bins = np.linspace(*tb) if isinstance(tb, list) else tb
    streaked_pe, kick = dumb_streaker(pe, streaking_beam, return_A_kick=True)
    sr, stheta, sphi = cartesian_to_spherical(*streaked_pe.p.T)
    mask = np.abs(stheta - pconf['detector']['theta center']) < pconf['detector']['theta acceptance']
    variable = pconf['detector']['variable']
    if variable == 'kinetic energy':
        hist, x, y = np.histogram2d((sphi[mask] + np.pi / 2) % (2 * np.pi), streaked_pe.Ekin()[mask] / const.e, bins=(detector_bins, energy_bins))
    elif variable == 'momentum':
        hist, x, y = np.histogram2d((sphi[mask] + np.pi / 2) % (2 * np.pi), 3e25 * np.linalg.norm(streaked_pe.p, axis=1)[mask], bins=(detector_bins, energy_bins))
    else:
        raise ValueError(f'Unsupported variable: {variable}')

    spec, _, _ = np.histogram2d(pe.t0, pe.Ekin() / const.e, bins=(t_bins, energy_bins))
    timedist = np.sum(spec, axis=1)
    return hist.T / hist.max(), spec.T / spec.max(), kick * 1e25, timedist


if __name__ == '__main__':
    yaml = YAML()
    #  TODO: Maybe implement some real CLI with argparse
    with open('simulations/configs/default.yaml' if len(sys.argv) < 2 else sys.argv[1]) as f:
        config = yaml.load(f)

    s = config['other parameters']['samples']
    phi = config['physical parameters']['detector']['phi bins']
    ke = config['physical parameters']['detector']['energy bins']
    ke = (ke[2] - 1) if isinstance(ke, list) else ke
    t = config['other parameters']['time bins']
    t = (t[2] - 1) if isinstance(t, list) else t

    chunk = s if s < config['other parameters']['chunk size'] else config['other parameters']['chunk size']

    f = h5py.File(config['other parameters']['output file name'], 'w')
    h5opt = {'compression': 'gzip', 'compression_opts': 9}
    a = f.create_dataset('detector_images', (s, ke, phi), chunks=(chunk, ke, phi), **h5opt)
    b = f.create_dataset('spectrograms', (s, ke, t), chunks=(chunk, ke, t), **h5opt)
    c = f.create_dataset('kick', (s,), chunks=(chunk,), **h5opt)
    d = f.create_dataset('time_distribution', (s, t), chunks=(chunk, t), **h5opt)

    for i, j, k, l in tqdm(zip(a.iter_chunks(), b.iter_chunks(), c.iter_chunks(), d.iter_chunks()), total=int(np.ceil(s/chunk)), desc='Total'):
        results = p_map(simulate, [sample_config(config) for i in range(i[0].stop - i[0].start)], smoothing=0.1, leave=False, desc='Chunk')
        a[i], b[j], c[k], d[l] = list(map(list, zip(*results)))
    f.close()

    

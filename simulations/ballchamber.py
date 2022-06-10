from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.conversions import ellipticity_to_jones_vector
from streaking.streak import dumb_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.detectors import flight_tubes
import numpy as np
from ruamel.yaml import YAML
from collections import OrderedDict
import copy
from matplotlib.animation import FuncAnimation as animate
import matplotlib.pyplot as plt
import scipy.constants as const


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


def streak(config, spectrogram, time_range, energy_range):
    pconf = sample_config(config)
    number_of_electrons = pconf['target']['number of electrons']
    binding_energy = pconf['target']['binding energy']

    TEmap = MultivariateMapInterpolator.from_array(spectrogram,
                                                   (time_range, energy_range))
    pe = ionizer_simple(pconf['target']['beta'], TEmap,
                        pconf['xfel']['focal size'], binding_energy,
                        number_of_electrons, pconf['target']['length'])

    streaking_beam = SimpleGaussianBeam(
        focal_size=(pconf['laser']['focal size x'],
                    pconf['laser']['focal size y']),
        envelope_offset=pconf['laser']['delay'],
        cep=pconf['laser']['cep'],
        wavelength=pconf['laser']['wavelength'],
        energy=pconf['laser']['pulse energy'],
        duration=pconf['laser']['pulse duration'],
        polarization=ellipticity_to_jones_vector(pconf['laser']['ellipticity'],
                                                 pconf['laser']['tilt'], 1))

    eb = pconf['detector']['energy bins']
    energy_bins = np.linspace(*eb) if isinstance(eb, list) else eb

    streaked_pe, kick = dumb_streaker(pe, streaking_beam, return_A_kick=True)
    mean_momentum = np.linalg.norm(pe.p, axis=1).mean()
    E0 = const.m_e * const.c ** 2
    mean_kinetic_energy = np.sqrt(E0 ** 2 + (mean_momentum * const.c) ** 2) - E0
    kick = (mean_momentum + kick) ** 2 / (2 * const.m_e) / const.e - mean_kinetic_energy / const.e
    dcfg = pconf['detector']
    image, x, y = flight_tubes(streaked_pe, np.radians(dcfg['angles']),
                               dcfg['tip distance'], dcfg['tip diameter'],
                               dcfg['mcp distance'], dcfg['mcp diameter'],
                               'kinetic energy', energy_bins)
    return image, kick


if __name__ == '__main__':
    yaml = YAML()
    with open('simulations/configs/sqs_june2022.yaml') as f:
        config = yaml.load(f)

    # Just a stupid test spectrogram
    spectrogram = np.zeros((400, 400))
    spectrogram[100:300, 190:210] = 1

    fig, ax = plt.subplots()
    tmp = np.zeros((16, 160))
    im = plt.imshow(tmp,
                    extent=(1, 16, *config['detector']['energy bins'][:2]),
                    aspect='auto',
                    origin='lower')

    def update(i):
        #######################################################################
        #                                                                     #
        # The streak method generates the detector image and the kick value.  #
        # You have to pass configuration, spectrogram and spectrogram ranges. #
        #                                                                     #
        #######################################################################
        img, kick = streak(config, spectrogram, (-1e-15, 1e-15), (960, 1040))
        ax.set_title(f"Kick: {kick:.2f} eV")
        im.set_array(img.T)
        im.autoscale()
        return im,

    a = animate(fig, update, frames=np.arange(100), repeat=True, interval=1)
    plt.show()

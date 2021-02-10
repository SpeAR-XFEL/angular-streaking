from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
import numpy as np
import scipy.stats
import scipy.constants as const
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("GTK3Agg")


def streakthis(energy, cep, termination_time):
    beam = SimpleGaussianBeam(energy=energy, cep=cep)

    XFEL_intensity = lambda t: (
        0.6 * scipy.stats.norm(0, 1e-16).pdf(t)
        + 0.4 * scipy.stats.norm(3e-16, 1e-16).pdf(t)
    )

    XFEL_photon_energy = scipy.stats.norm(1200, 0.1).pdf

    pe = ionizer_simple(
        2,  # β
        XFEL_intensity,
        XFEL_photon_energy,
        870.2,  # binding energy
        (1190, 1210),  # considered energy range
        (-5e-16, 7e-16),  # considered time range
        10000,  # number of electrons to generate (not yet based on cross section)
    )

    return pe, classical_lorentz_streaker(pe, beam, (0, termination_time))


if __name__ == "__main__":

    # x = np.linspace(-5e-15, 7e-15, 100)
    # plt.plot(x, XFEL_intensity(x))

    # rsr, _, _ = cartesian_to_spherical(*spe.r.T)
    # plt.hist(r,  density=False, color="C0", alpha=0.7, bins=100, histtype='step')
    # plt.hist(sr, density=False, color="C1", alpha=0.7, bins=100, histtype='step')
    # plt.show()

    pe, spe = streakthis(1e-8, 0, 2e-12)

    r, phi, theta = cartesian_to_spherical(*pe.p.T)
    sr, sphi, stheta = cartesian_to_spherical(*spe.p.T)

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(4, 2, height_ratios=[20, 1, 1, 1], figure=fig)
    ax1 = plt.subplot(gs[0, 0])
    bins = [np.linspace(0, 2 * np.pi, 101), 100]

    # Create 2d Histogram
    data, x, y = np.histogram2d(
        (theta + np.pi / 2) % (2 * np.pi), pe.Ekin() / const.e, bins=bins
    )
    im1 = ax1.imshow(data.T, origin="lower", aspect="auto")
    im1.set_extent((x[0], x[-1], y[0], y[-1]))
    # plt.hist2d(pe.p.T[0], pe.p.T[1], bins=200)
    ax1.set_title("Unstreaked")
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title("Streaked")
    data, x, y = np.histogram2d(
        (stheta + np.pi / 2) % (2 * np.pi), spe.Ekin() / const.e, bins=bins
    )
    im2 = ax2.imshow(data.T, origin="lower", aspect="auto")
    im2.set_extent((x[0], x[-1], y[0], y[-1]))
    # plt.hist2d(spe.p.T[0], spe.p.T[1], bins=200)

    # plt.hist(pe.t0, bins=100)
    # plt.show()

    def update(val):
        # Change arguments and calculate new trajectories
        pe, spe = streakthis(s0.val, s1.val, s2.val)
        r, phi, theta = cartesian_to_spherical(*pe.p.T)
        sr, sphi, stheta = cartesian_to_spherical(*spe.p.T)
        data, x, y = np.histogram2d(
            (theta + np.pi / 2) % (2 * np.pi), pe.Ekin() / const.e, bins=bins
        )
        im1.set_data(data.T)
        im1.set_extent((x[0], x[-1], y[0], y[-1]))
        data, x, y = np.histogram2d(
            (stheta + np.pi / 2) % (2 * np.pi), spe.Ekin() / const.e, bins=bins
        )
        im2.set_data(data.T)
        im2.set_extent((x[0], x[-1], y[0], y[-1]))

    # Create three slider axes to modify α0-α2 on the fly
    slax0, slax1, slax2 = (
        plt.subplot(gs[1, :]),
        plt.subplot(gs[2, :]),
        plt.subplot(gs[3, :]),
    )
    s0 = Slider(slax0, r"Energy", 0, 30e-6, valfmt="%.1e", valinit=1e-8)
    s1 = Slider(slax1, r"CEP", 0, 2 * np.pi, valfmt="%.1e", valinit=0)
    s2 = Slider(slax2, r"time", 0, 1e-11, valfmt="%.1e", valinit=2e-12)
    for s in (s0, s1, s2):
        s.valtext.set_fontfamily("monospace")
        s.on_changed(update)

    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(E[..., 0], E[..., 1], z * 1e6)
    # ax.set_xlabel(r"$E_x$ / Vm$^{-1}$")
    # ax.set_ylabel(r"$E_y$ / Vm$^{-1}$")
    # ax.set_zlabel(r"$z$ / µm")
    # plt.show()

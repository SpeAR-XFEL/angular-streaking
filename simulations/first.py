from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.time_energy_map import Time_Energy_Map
from streaking.ionization import ionizer_simple, ionizer_Sauter
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
import numpy as np
import scipy.stats
import scipy.constants as const
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Gtk3Agg")
matplotlib.rcParams['figure.dpi'] = 150

if __name__ == "__main__":
    """
    XFEL_intensity = lambda t: (
        0.6 * scipy.stats.norm(0, 1e-15).pdf(t)
        + 0.4 * scipy.stats.norm(3e-15, 2e-15).pdf(t)
    )

    XFEL_photon_energy = scipy.stats.norm(1200, 0.1).pdf

    pe = ionizer_simple(
        2,  # β
        XFEL_intensity,
        XFEL_photon_energy,
        870.2,  # binding energy
        (1190, 1210),  # considered energy range
        (-5e-14, 7e-14),  # considered time range
        5000,  # number of electrons to generate (not yet based on cross section)
    )"""

    pe = None

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 2, 2], height_ratios=[5, 20, 5], figure=fig)
    
    # time-energy map
    ax0 = plt.subplot(gs[0, 1:])
    ax0.set_xlabel("$t$ / fs")
    ax0.xaxis.labelpad = -12
    ax0.set_ylabel(r"$h\nu$ / eV")
    teim = ax0.imshow([[1], [1]], origin="lower", aspect="auto")

    # 2d histograms for KE over angle
    bins = [np.linspace(0, 2 * np.pi, 51), 50]
    ax1 = plt.subplot(gs[1, 1])
    ax2 = plt.subplot(gs[1, 2])
    zerodata = np.zeros((len(bins[0]), bins[1]))
    im1 = ax1.imshow(zerodata, origin="lower", aspect="auto")
    im2 = ax2.imshow(zerodata, origin="lower", aspect="auto")
    ax1.set_title("Unstreaked")
    ax2.set_title("Streaked")
    for ax in (ax1, ax2):
        ax.set_ylabel(r"$E_\mathrm{kin}$ / eV")
        ax.tick_params(bottom=False, labelbottom=False)

    # marginal distributions
    axmarg1 = plt.subplot(gs[2, 1], sharex=ax1)
    axmarg2 = plt.subplot(gs[2, 2], sharex=ax2, sharey=axmarg1)
    marg = np.zeros(len(bins[0]))
    st1, = axmarg1.step(bins[0], marg, where='pre')
    st2, = axmarg2.step(bins[0], marg, where='pre', color='C0', alpha=1)
    st3, = axmarg2.step(bins[0], marg, where='pre', color='C1', alpha=0.5)
    fb1 = axmarg2.fill_between(bins[0], marg, marg, step='pre', alpha=0.5, color='C1')

    for ax in(axmarg1, axmarg2):
        ax.set_xlabel(r"$\varphi$")
        ax.set_xlim(bins[0][0], bins[0][-1])
        ax.tick_params(left=False, labelleft=False)


    def update_electrons(val):
        global pe
        N_G = int(sliders['peaks'].val)

        if N_G > 1:
            mu_t = np.random.normal(0, 1.5e-15, N_G)  # s
            mu_E = np.random.normal(1200, 2, N_G)  # eV
            sigma_t = np.abs(np.random.normal(0.4e-15, 0.2e-15, N_G))
            sigma_E = np.abs(np.random.normal(2, 0.5, N_G))
            corr_list = np.random.normal(0, 0, N_G)
            I_list = np.abs(np.random.normal(10, 0.1, N_G))
            stepsizes = (1e-16, 0.1)

            TEmap = Time_Energy_Map(
                mu_list=np.stack((mu_t, mu_E)),
                sigma_list=np.stack((sigma_t, sigma_E)),
                corr_list=corr_list,
                I_list=I_list,
                stepsizes=stepsizes,
            )
        else: 
            TEmap = Time_Energy_Map(
                mu_list=((0,), (1200,),),
                sigma_list=((sliders['xfel length (1 pk)'].val,), (0.1,),),
                corr_list=(0,),
                I_list=(1,),
                stepsizes=(1e-18, 0.1)
            )

        N_e = int(sliders['electrons'].val)
        E_ionize = 1150  # eV
        pe = ionizer_Sauter(TEmap, E_ionize, N_e)
        teim.set_data(TEmap.time_energy_map)
        teim.set_extent((TEmap.time_list[0] * 1e15, TEmap.time_list[-1] * 1e15, TEmap.Ekin_list[0], TEmap.Ekin_list[-1]))
        teim.autoscale()

        update_streaking(None)

    def update_streaking(val):
        global fb1

        beam = SimpleGaussianBeam(energy=sliders['str. energy / J'].val, cep=sliders['str. CEP'].val, envelope_offset=sliders['str. delay'].val, wavelength=sliders['str. lambda'].val)
        spe = classical_lorentz_streaker(pe, beam, (0, sliders['time / s'].val), sliders['stepsize / s'].val)
        r, phi, theta = cartesian_to_spherical(*pe.p.T)
        sr, sphi, stheta = cartesian_to_spherical(*spe.p.T)
        rsr, _, _ = cartesian_to_spherical(*spe.r.T)
        data1, x1, y1 = np.histogram2d(
            (theta + np.pi / 2) % (2 * np.pi), pe.Ekin() / const.e, bins=bins
        )
        data2, x2, y2 = np.histogram2d(
            (stheta + np.pi / 2) % (2 * np.pi), spe.Ekin() / const.e, bins=bins
        )
        im1.set_data(data1.T)
        im2.set_data(data2.T)
        im1.set_extent((x1[0], x1[-1], y1[0], y1[-1]))
        im2.set_extent((x2[0], x2[-1], y2[0], y2[-1]))
        im1.autoscale()
        im2.autoscale()
        marg1 = np.append(data1.T.sum(axis=0), 0)
        marg2 = np.append(data2.T.sum(axis=0), 0)
        st1.set_ydata(marg1)
        st2.set_ydata(marg1)
        st3.set_ydata(marg2)
        fb1.remove()
        fb1 = axmarg2.fill_between(x2, marg1, marg2, step='pre', alpha=0.5, color='C1')
        return im1, im2, st1, st2, st3

    # Sliders galore!
    sliders_spec = {
        "peaks":              (1,        10,        1,  5,     update_electrons),
        "xfel length (1 pk)": (1e-17,     5e-15, None,  1e-15, update_electrons),
        "electrons":          (1e3,       5e4,      1,  1e3,   update_electrons),
        "str. lambda":        (1e-7,      10e-6, None, 10e-6,  update_streaking),
        "str. delay":         (-1e-12,    1e-12, None,  0,     update_streaking),
        "str. energy / J":    (0,       100e-6,  None, 30e-6,  update_streaking),
        "str. CEP":           (0,      2*np.pi,  None,  0,     update_streaking),
        "time / s":           (0,         1e-11, None,  1e-12, update_streaking),
        "stepsize / s":       (5e-15,     2e-14, None,  1e-14, update_streaking),
    #   Name                   min        max    step  start   update function
    }

    gs_widgets = gs[:,0].subgridspec(30, 1)
    # radio_ax = fig.add_subplot(gs_widgets[0:1])
    # radio_ax.set_zorder(10)
    # rb = RadioButtons(radio_ax, labels=("simple", "complex"))
    sl_axes = [fig.add_subplot(gs_widgets[i]) for i in range(len(sliders_spec))]
    sliders = {}
    for ax, key in zip(sl_axes, sliders_spec.keys()):
        ax.set_zorder(10)
        mi, ma, ste, sta, fun = sliders_spec[key]
        if ste is None:
            sl = Slider(ax, key, mi, ma, valinit=sta, valfmt='%.1e')
        else:
            sl = Slider(ax, key, mi, ma, valinit=sta, valstep=ste, valfmt='%.1e')
        sl.on_changed(fun)
        sl.valtext.set_fontfamily("monospace")
        sliders[key] = sl

    plt.show()

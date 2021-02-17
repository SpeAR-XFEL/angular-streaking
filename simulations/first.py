from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.time_energy_map import Time_Energy_Map
from streaking.ionization import ionizer_simple, ionizer_Sauter
from streaking.conversions import cartesian_to_spherical
from streaking.streak import classical_lorentz_streaker
import numpy as np
import scipy.stats
import scipy.constants as const
from matplotlib.widgets import Slider, RadioButtons
from matplotlib import animation
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import time

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
    dpi = 100
    fig = plt.figure(constrained_layout=False, figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    #fig.set_size_inches(16, 9, True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 8], height_ratios=[1, 5], figure=fig)
    
    # time-energy map
    ax0 = fig.add_subplot(gs[0, 1])
    ax0.set_xlabel("$t$ / fs")
    ax0.xaxis.labelpad = -12
    ax0.set_ylabel(r"$h\nu$ / eV")
    teim = ax0.imshow([[1], [1]], origin="lower", aspect="auto")

    subgs = gs[1, 1].subgridspec(3, 2, hspace=0, wspace=0.1, height_ratios=(5, 2, 1))

    # 2d histograms for KE over angle
    phi_bin_count = 64
    bins = [np.linspace(0, 2 * np.pi, phi_bin_count+1), 50]
    ax1 = fig.add_subplot(subgs[0, 0])
    ax2 = fig.add_subplot(subgs[0, 1])
    zerodata = np.zeros((len(bins[0]), bins[1]))
    im1 = ax1.imshow(zerodata, origin="lower", aspect="auto")
    im2 = ax2.imshow(zerodata, origin="lower", aspect="auto")
    ax1.set_title("Unstreaked")
    ax2.set_title("Streaked")
    #loc = ticker.MaxNLocator(nbins=9, steps=(1,2,2.5,5,10), prune='both')
    for ax in (ax1, ax2):
        ax.set_ylabel(r"$E_\mathrm{kin}$ / eV")
        ax.tick_params(bottom=False, labelbottom=False)
        #ax.yaxis.set_major_locator(loc)

    # marginal distributions
    axmarg1 = fig.add_subplot(subgs[1, 0], sharex=ax1)
    axmarg2 = fig.add_subplot(subgs[1, 1], sharex=ax2, sharey=axmarg1)
    axdiff  = fig.add_subplot(subgs[2, 1], sharex=ax2)
    marg = np.zeros(len(bins[0]))
    st1, = axmarg1.step(bins[0], marg, where='pre')
    st2, = axmarg2.step(bins[0], marg, where='pre', color='C0', alpha=1)
    st3, = axmarg2.step(bins[0], marg, where='pre', color='C1', alpha=0.5)
    fb1 = axmarg2.fill_between(bins[0], marg, marg, step='pre', alpha=0.5, color='C1')
    axdiff.axhline(0, color='k', lw=1)
    st4, = axdiff.step(bins[0], marg, where='pre', color='C1')
    loc = ticker.FixedLocator((0,))
    axdiff.yaxis.set_major_locator(loc)
    axmarg2.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    axmarg1.tick_params(left=False, labelleft=False)

    for ax in (axmarg1, axdiff):
        ax.set_xlabel(r"$\varphi$")
        ax.set_xlim(bins[0][0], bins[0][-1])

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
                sigma_list=((sliders['xfel dur. (1pk) / s'].val,), (0.1,),),
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
        start = time.perf_counter()

        beam = SimpleGaussianBeam(
            energy=sliders['str. energy / J'].val,
            cep=sliders['str. CEP'].val,
            envelope_offset=sliders['str. delay / s'].val,
            wavelength=sliders['str. lambda / m'].val,
            duration=sliders['str. duration / s'].val)
        spe = classical_lorentz_streaker(pe, beam, (0, sliders['sim time / s'].val), sliders['stepsize / s'].val)
        r, phi, theta = cartesian_to_spherical(*pe.p.T)
        sr, sphi, stheta = cartesian_to_spherical(*spe.p.T)
        rsr, _, _ = cartesian_to_spherical(*spe.r.T)

        diff = time.perf_counter()-start
        per = diff/sliders['electrons'].val
        print(f'Streaking took {diff:.1f} s, {per*1e6:.1f} µs / e-')

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
        #im1.autoscale()
        #im2.autoscale()
        marg1 = np.append(data1.T.sum(axis=0), 0)
        marg2 = np.append(data2.T.sum(axis=0), 0)
        st1.set_ydata(marg1)
        st2.set_ydata(marg1)
        st3.set_ydata(marg2)
        st4.set_ydata(marg2 - marg1)
        axdiff.relim()
        axdiff.autoscale_view()
        fb1.remove()
        fb1 = axmarg2.fill_between(x2, marg1, marg2, step='pre', alpha=0.5, color='C1')
        return im1, im2, st1, st2, st3

    # Sliders galore!
    sliders_spec = { 
        "peaks":              (1,        10,        1,  1,     '%1d', update_electrons),
        "xfel dur. (1pk) / s":(1e-17,     1e-14, None,  1e-15, None, update_electrons),
        "electrons":          (1e3,       1e6,      1,  1e3,   None, update_electrons),
        "str. lambda / m":    (1e-7,      10e-6, None, 10e-6,  None, update_streaking),
        "str. duration / s":  (1e-14,     1e-12, None,  3e-13, None, update_streaking),
        "str. delay / s":     (-1e-12,    1e-12, None,  0,     None, update_streaking),
        "str. energy / J":    (0,       100e-6,  None, 30e-6,  None, update_streaking),
        "str. CEP":           (0,      2*np.pi,  None,  0,     '%1.2f', update_streaking),
        "sim time / s":       (0,         1e-11, None,  1e-12, None, update_streaking),
        "stepsize / s":       (5e-15,     2e-14, None,  1e-14, None, update_streaking),
    #   Name                   min        max    step  start   fmt   update function
    }

    gs_widgets = gs[:,0].subgridspec(30, 1)
    # radio_ax = fig.add_subplot(gs_widgets[0:1])
    # radio_ax.set_zorder(10)
    # rb = RadioButtons(radio_ax, labels=("simple", "complex"))
    sl_axes = [fig.add_subplot(gs_widgets[i]) for i in range(len(sliders_spec))]
    sliders = {}
    for ax, key in zip(sl_axes, sliders_spec.keys()):
        ax.set_zorder(10)
        mi, ma, ste, sta, fmt, fun = sliders_spec[key]
        if ste is None:
            sl = Slider(ax, key, mi, ma, valinit=sta, valfmt='%.1e' if fmt is None else fmt)
        else:
            sl = Slider(ax, key, mi, ma, valinit=sta, valstep=ste, valfmt='%.1e'if fmt is None else fmt)
        sl.on_changed(fun)
        sl.valtext.set_fontfamily("monospace")
        sliders[key] = sl


    frames = 20
    def animate(frame):#1e-17,     5e-15
        #sliders['str. CEP'].set_val(frame / 200 * 2 * np.pi)
        sliders['xfel dur. (1pk) / s'].set_val(frame / frames * (5e-15-1e-17) + 1e-17)
        return im1, im2, st1, st2, st3

    update_electrons(None)
    im1.autoscale()
    im2.autoscale()
    plt.tight_layout(pad=1)
    plt.show()
    #quit()
    #anim = animation.FuncAnimation(fig, animate,
    #                           frames=frames, blit=True)
    #anim.save('build/anim_dur.mp4', fps=5, dpi=100)

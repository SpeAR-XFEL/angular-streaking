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
from numpy import pi as π


matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 150

if __name__ == '__main__':
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
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 8], height_ratios=[1, 5], figure=fig)

    # time-energy map
    ax0 = fig.add_subplot(gs[0, 1:])
    ax0.set_xlabel('$t$ / fs')
    ax0.set_ylabel(r'$h\nu$ / eV')
    teim = ax0.imshow([[1], [1]], origin='lower', aspect='auto')

    subgs12 = gs[1, 1].subgridspec(1, 2, wspace=0.1)
    subgs1 = subgs12[0, 0].subgridspec(3, 2, hspace=0, wspace=0, height_ratios=(5, 2, 1), width_ratios=(4,1))
    subgs2 = subgs12[0, 1].subgridspec(3, 2, hspace=0, wspace=0, height_ratios=(5, 2, 1), width_ratios=(4,1))

    # 2d histograms for KE over angle
    phi_bin_count = 64
    bins = [np.linspace(0, 2 * np.pi, phi_bin_count+1), 50]
    ax1 = fig.add_subplot(subgs1[0, 0])
    ax2 = fig.add_subplot(subgs2[0, 0])
    zerodata = np.zeros((len(bins[0]), bins[1]))
    im1 = ax1.imshow(zerodata, origin='lower', aspect='auto')
    im2 = ax2.imshow(zerodata, origin='lower', aspect='auto')
    ax1.set_title('Unstreaked')
    ax2.set_title('Streaked')
    for ax in (ax1, ax2):
        ax.set_ylabel(r'$E_\mathrm{kin}$ / eV')
        ax.tick_params(bottom=False, labelbottom=False)

    # marginal distributions
    axmarg1x = fig.add_subplot(subgs1[1, 0], sharex=ax1)
    axmarg2x = fig.add_subplot(subgs2[1, 0], sharex=ax2, sharey=axmarg1x)
    axdiffx  = fig.add_subplot(subgs2[2, 0], sharex=ax2)
    axmarg1y = fig.add_subplot(subgs1[0, 1], sharey=ax1)
    axmarg2y = fig.add_subplot(subgs2[0, 1], sharey=ax2)
    margx = np.zeros(len(bins[0]))
    st1, = axmarg1x.step(bins[0], margx, where='post')
    st2, = axmarg2x.step(bins[0], margx, where='post', color='C0', alpha=1)
    st3, = axmarg2x.step(bins[0], margx, where='post', color='C1', alpha=0.5)
    fb1 = axmarg2x.fill_between(bins[0], margx, margx, step='pre', alpha=0.5, color='C1')
    axdiffx.axhline(0, color='k', lw=1)
    st4, = axdiffx.step(bins[0], margx, where='post', color='C1')
    loc = ticker.FixedLocator((0,))
    axdiffx.yaxis.set_major_locator(loc)
    st5, = axmarg1y.plot([1], [1], color='C0', drawstyle='steps-pre')
    st6, = axmarg2y.plot([1], [1], color='C0', drawstyle='steps-pre')
    axmarg2x.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    axmarg1x.tick_params(left=False, labelleft=False)
    axmarg1y.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    axmarg2y.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    for ax in (axmarg1x, axdiffx):
        ax.set_xlabel(r'$\varphi$')
        ax.set_xlim(bins[0][0], bins[0][-1])

    def update_electrons(val):
        global pe
        N_G = int(sliders['XFEL']['peaks'].val)
        N_e = int(sliders['simulation']['electrons'].val)
        E_ionize = sliders['target']['binding E / eV'].val  # eV
        β = sliders['target']['β (1pk)'].val

        start = time.perf_counter()

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

            pe = ionizer_Sauter(TEmap, E_ionize, N_e)
            imdata = TEmap.time_energy_map
            imextent = (TEmap.time_list[0] * 1e15, TEmap.time_list[-1] * 1e15, TEmap.Ekin_list[0], TEmap.Ekin_list[-1])
        else:
            dur = sliders['XFEL']['width (1pk) / s'].val
            sigE = sliders['XFEL']['σ(E) (1pk) / eV'].val
            tEmeans = (0, sliders['XFEL']['µ(E) (1pk) / eV'].val)
            # Please dont ask about this fs bullshit.. just for display purposes
            tEcov = np.diag((dur, sigE))**2
            tEcovfs = np.diag((dur*1e15, sigE))**2
            pe = ionizer_simple(β, tEmeans, tEcov, E_ionize, N_e)
            sigma_range = 4
            rangeI = (-sigma_range*dur, sigma_range*dur)
            rangeIfs = (-sigma_range*dur*1e15, sigma_range*dur*1e15)
            rangeE = (-sigma_range*sigE + tEmeans[1], sigma_range*sigE + tEmeans[1])
            A, B = np.meshgrid(np.linspace(*rangeIfs, 300), np.linspace(*rangeE, 300))
            imdata = scipy.stats.multivariate_normal(tEmeans, tEcovfs, allow_singular=True).pdf(np.dstack((A, B)))
            imextent = (*rangeIfs, *rangeE)

        diff = time.perf_counter() - start
        per = diff/sliders['simulation']['electrons'].val
        print(f'Generating electrons took {diff:.1f} s, {per*1e6:.1f} µs / e-. ', end='')

        teim.set_data(imdata)
        teim.set_extent(imextent)
        teim.autoscale()

        update_streaking(None)

    def update_streaking(val):
        global fb1
        start = time.perf_counter()

        beam = SimpleGaussianBeam(
            energy=sliders['streaking']['energy / J'].val,
            cep=sliders['streaking']['CEP'].val,
            envelope_offset=sliders['streaking']['delay / s'].val,
            wavelength=sliders['streaking']['wavelen. / m'].val,
            duration=sliders['streaking']['width / s'].val)
        spe = classical_lorentz_streaker(pe, beam, (0, sliders['simulation']['time / s'].val), sliders['simulation']['stepsize / s'].val)
        r, phi, theta = cartesian_to_spherical(*pe.p.T)
        sr, sphi, stheta = cartesian_to_spherical(*spe.p.T)
        rsr, _, _ = cartesian_to_spherical(*spe.r.T)

        diff = time.perf_counter()-start
        per = diff/sliders['simulation']['electrons'].val
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
        im1.autoscale()
        im2.autoscale()
        marg1x = np.append(data1.T.sum(axis=0), 0)
        marg2x = np.append(data2.T.sum(axis=0), 0)
        marg1y = np.append(data1.T.sum(axis=1), 0)
        marg2y = np.append(data2.T.sum(axis=1), 0)
        st1.set_ydata(marg1x)
        st2.set_ydata(marg1x)
        st3.set_ydata(marg2x)
        st4.set_ydata(marg2x - marg1x)
        axdiffx.relim()
        axdiffx.autoscale_view()
        fb1.remove()
        fb1 = axmarg2x.fill_between(x2, marg1x, marg2x, step='post', alpha=0.5, color='C1')
        st5.set_data(marg1y, y1)
        st6.set_data(marg2y, y2)
        for i, ax in enumerate((axmarg1y, axmarg2y, axmarg1x, axmarg2x)):
            ax.relim()
            ax.autoscale_view(scaley=(i >= 2), scalex=(i < 2))
        return im1, im2, st1, st2, st3, st4, st5, st6

    # Sliders galore!
    sliders_spec = {
        'XFEL': {
            'peaks':              (1,       10,    1,     1,      '%1d',   update_electrons),
            'width (1pk) / s':    (1e-17,   1e-14, None,  1e-15,  None,    update_electrons),
            'µ(E) (1pk) / eV':    (800,     2000,  None,  1200,   '%.0f',  update_electrons),
            'σ(E) (1pk) / eV':    (0.1,     10,    None,  0.5,    '%.1f',  update_electrons),
        },
        'target': {
            'binding E / eV':     (500,     1500,  None,  1150,   '%.0f',  update_electrons),
            'β (1pk)':                  (-1,      2,     None,  2,      '%.2f',  update_electrons),
        },
        'streaking': {
            'wavelen. / m':       (1e-7,    10e-6, None,  10e-6,  None,    update_streaking),
            'width / s':          (1e-14,   1e-12, None,  3e-13,  None,    update_streaking),
            'delay / s':          (-1e-12,  1e-12, None,  0,      None,    update_streaking),
            'energy / J':         (0,       1e-3,  None,  30e-6,  None,    update_streaking),
            'CEP':                (0,       2*π,   None,  0,     '%1.2f',  update_streaking),
        },
        'simulation': {
            'electrons':          (1e3,     5e5,   1,     5e4,    None,    update_electrons),
            'time / s':           (0,       1e-11, None,  1e-12,  None,    update_streaking),
            'stepsize / s':       (5e-15,   2e-14, None,  1e-14,  None,    update_streaking),
        },

    #        Name                  min      max    step   start   fmt       update function
    }

    gs_widgets = gs[:, 0].subgridspec(30, 1)
    idx = 0
    sliders = {}
    for cat in sliders_spec.keys():
        cat_ax = fig.add_subplot(gs_widgets[idx])
        cat_ax.axis('off')
        cat_ax.text(0, 0, cat, fontweight='bold')
        idx += 1
        sliders[cat] = {}
        for key in sliders_spec[cat].keys():
            ax = fig.add_subplot(gs_widgets[idx])
            idx += 1
            ax.set_zorder(10)
            mi, ma, ste, sta, fmt, fun = sliders_spec[cat][key]
            if ste is None:
                sl = Slider(ax, key, mi, ma, valinit=sta, valfmt='%.1e' if fmt is None else fmt)
            else:
                sl = Slider(ax, key, mi, ma, valinit=sta, valstep=ste, valfmt='%.1e'if fmt is None else fmt)
            sl.on_changed(fun)
            sl.valtext.set_fontfamily('monospace')
            sliders[cat][key] = sl

    frames = 400
    def animate(frame):#1e-17,     5e-15
        sliders['streaking']['CEP'].set_val(frame / frames * 2 * np.pi)
        #sliders['xfel dur. (1pk) / s'].set_val(frame / frames * (5e-15-1e-17) + 1e-17)
        return im1, im2, st1, st2, st3

    update_electrons(None)
    im1.autoscale()
    im2.autoscale()
    plt.tight_layout(pad=1)
    plt.show()
    #anim = animation.FuncAnimation(fig, animate,
    #                           frames=frames, blit=True)
    #anim.save('build/anim_cep2.mp4', fps=50, dpi=100)

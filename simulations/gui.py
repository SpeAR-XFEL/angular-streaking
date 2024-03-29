from streaking.gaussian_beam import SimpleGaussianBeam
from streaking.ionization import ionizer_simple, naive_auger_generator, ionizer_total_cs
from streaking.conversions import cartesian_to_spherical, spherical_to_cartesian, ellipticity_to_jones_vector
from streaking.streak import classical_lorentz_streaker, dumb_streaker
from streaking.multivariate_map_interpolator import MultivariateMapInterpolator
from streaking.stats import covariance_from_correlation_2d
from streaking.detectors import constant_polar_angle_ring, energy_integrated_4pi
import numpy as np
import scipy.constants as const
from scipy.spatial.transform import Rotation
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import matplotlib
import time
from numpy import pi as π


matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 150

if __name__ == '__main__':
    pe = None
    spe = None
    # Fraction of cut-off values in otherwise unbounded histograms 
    discard = 0.01
    dpi = 75
    fig = plt.figure(constrained_layout=False, figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 8], height_ratios=[1, 6], figure=fig, hspace=0.15, left=0.07, right=0.99, top=0.95, bottom=0.05)

    # time-energy map
    gshdr = gs[0, 1].subgridspec(1, 2, wspace=0.1)
    ax0 = fig.add_subplot(gshdr[0, 0])
    ax0.set_xlabel('$t$ / fs')
    ax0.set_ylabel(r'$h\nu$ / eV')
    teim = ax0.imshow([[1], [1]], origin='lower', aspect='auto', interpolation='none')
    ax0.set_title('Photon distribution')

    # kinetic energy histogram
    axke = fig.add_subplot(gshdr[0, 1])
    axke.set_xlabel('$t$ / fs')
    axke.set_ylabel(r'$e^-$ KE / eV')
    keim = axke.imshow([[1], [1]], origin='lower', aspect='auto', interpolation='none')
    axke.set_title('Photoelectron distribution')

    subgs12 = gs[1, 1].subgridspec(1, 2, wspace=0.1)
    subgs1 = subgs12[0, 0].subgridspec(3, 2, hspace=0, wspace=0, height_ratios=(5, 2, 5), width_ratios=(4,1))
    subgs2 = subgs12[0, 1].subgridspec(3, 2, hspace=0, wspace=0, height_ratios=(5, 2, 5), width_ratios=(4,1))

    # 2d histograms for KE over angle
    #phi_bin_count = 32
    #bins = [np.linspace(0, 2 * np.pi, phi_bin_count+1), 50]
    ax1 = fig.add_subplot(subgs1[0, 0])
    ax2 = fig.add_subplot(subgs2[0, 0])
    zerodata = np.zeros((1, 1))
    im1 = ax1.imshow(zerodata, origin='lower', aspect='auto', interpolation='none')
    im2 = ax2.imshow(zerodata, origin='lower', aspect='auto', interpolation='none')
    ax1.set_title('Unstreaked')
    ax2.set_title('Streaked')
    for ax in (ax1, ax2):
        ax.legend((), (), title='ring detector', loc='upper left', labelspacing=0)
        ax.set_ylabel(r'$E_\mathrm{kin}$ / eV')
        ax.tick_params(bottom=False, labelbottom=False)

    # 2d histograms azimuthal and polar angle
    ax3 = fig.add_subplot(subgs1[2, 0])
    ax4 = fig.add_subplot(subgs2[2, 0])
    zerodata = np.zeros((1, 1))
    im3 = matplotlib.image.NonUniformImage(ax3, origin='lower', extent=(0, 2 * π, 0, π))
    im4 = matplotlib.image.NonUniformImage(ax4, origin='lower', extent=(0, 2 * π, 0, π))
    ax3.images.append(im3)
    ax4.images.append(im4)
    sp3, = ax3.fill((0, 0), color='C3', alpha=0.25)
    sp4, = ax4.fill((0, 0), color='C3', alpha=0.25)

    for ax in (ax3, ax4):
        ax.set_xlabel(r'$\varphi$')
        ax.set_ylabel(r'$\vartheta$')
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, np.pi)
        legend = ax.legend((sp3,), ('ring det. acceptance',), title=r'4$\,$π detector', loc='upper left')
        legend._legend_box.align = "left"

    # marginal distributions
    axmarg1x = fig.add_subplot(subgs1[1, 0], sharex=ax1)
    axmarg2x = fig.add_subplot(subgs2[1, 0], sharex=ax2, sharey=axmarg1x)
    axmarg1y = fig.add_subplot(subgs1[0, 1], sharey=ax1)
    axmarg2y = fig.add_subplot(subgs2[0, 1], sharey=ax2)
    axmarg3y = fig.add_subplot(subgs1[2, 1], sharey=ax3)
    axmarg4y = fig.add_subplot(subgs2[2, 1], sharey=ax4)
    st1 = axmarg1x.add_patch(StepPatch((0,), (0, 1), ec='C0', fc='#1f77b444'))
    st2 = axmarg2x.add_patch(StepPatch((0,), (0, 1), ec='C0', fc='#1f77b444'))
    st3 = axmarg2x.add_patch(StepPatch((0,), (0, 1), ec='C1', fc='#ff7f0e44'))
    st7 = axmarg3y.add_patch(StepPatch((0,), (0, 1), ec='C0', fc='#1f77b444', orientation='horizontal'))
    st9 = axmarg4y.add_patch(StepPatch((0,), (0, 1), ec='C0', fc='#1f77b444', orientation='horizontal'))
    st8 = axmarg4y.add_patch(StepPatch((0,), (0, 1), ec='C1', fc='#ff7f0e44', orientation='horizontal'))
    st5 = axmarg1y.add_patch(StepPatch((0,), (0, 1), ec='C0', fc='#1f77b444', orientation='horizontal'))
    st6 = axmarg2y.add_patch(StepPatch((0,), (0, 1), ec='C0', fc='#1f77b444', orientation='horizontal'))
    for ax in (axmarg2x, axmarg1x, axmarg1y, axmarg2y, axmarg3y, axmarg4y):
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    def update_electrons(val):
        global pe, spe
        N_G = int(sliders['XFEL']['peaks'].val)
        #N_e = int(sliders['target']['photoelectrons'].val)
        E_ionize = sliders['target']['PE binding E / eV'].val  # eV
        β = sliders['target']['β'].val

        xfelE = sliders['XFEL']['pulse energy / J'].val
        xfelSpot = sliders['XFEL']['focal spot / m'].val
        targetLen = sliders['target']['length / m'].val
        targetDen = sliders['target']['number density·cm³'].val

        start = time.perf_counter()

        if N_G > 2:
            mu_t = np.random.normal(0, 5e-15, N_G)  # s
            mu_E = np.random.normal(940, 1, N_G)  # eV
            sigma_t = np.abs(np.random.normal(1e-15, 0.2e-15, N_G))
            sigma_E = np.abs(np.random.normal(2, 0.1, N_G))
            corr_list = np.random.normal(0, 0, N_G)
            I_list = np.abs(np.random.normal(1, 0.05, N_G))

            covs = covariance_from_correlation_2d(np.stack((sigma_t, sigma_E)), corr_list).T
            TEmap = MultivariateMapInterpolator.from_gauss_blob_list(np.stack((mu_t, mu_E)).T, covs, I_list)

            pe = ionizer_simple(β, TEmap, sliders['XFEL']['focal spot / m'].val, E_ionize, N_e)
            imdata = TEmap.map.T
            imextent = TEmap.domain.flatten()
            imextent[[0, 1]] *= 1e15
        elif N_G == 2:
            twopk_dist = sliders['XFEL']['distance (2pk) / s'].val
            dur = sliders['XFEL']['width (1-2pk) / s'].val
            sigE = sliders['XFEL']['σ(E) (1-2pk) / eV'].val
            muE = sliders['XFEL']['µ(E) (1-2pk) / eV'].val
            mu_t = (-twopk_dist / 2, twopk_dist / 2)
            mu_E = (muE, muE)  # eV
            sigma_t = (dur, dur)
            sigma_E = (sigE, sigE)
            corr_list = (0, 0)
            I_list = (0.5, 0.5)

            covs = covariance_from_correlation_2d(np.stack((sigma_t, sigma_E)), corr_list).T
            TEmap = MultivariateMapInterpolator.from_gauss_blob_list(np.stack((mu_t, mu_E)).T, covs, I_list)

            pe = ionizer_total_cs('Ne1s', TEmap, xfelE, xfelSpot, targetLen, targetDen)
            #pe = ionizer_simple(β, TEmap, sliders['XFEL']['focal spot / m'].val, E_ionize, N_e)
            #pe = ionizer_Sauter(TEmap, E_ionize, N_e)
            imdata = TEmap.map.T
            imextent = TEmap.domain.flatten()
            imextent[[0, 1]] *= 1e15
        else:
            dur = sliders['XFEL']['width (1-2pk) / s'].val
            sigE = sliders['XFEL']['σ(E) (1-2pk) / eV'].val
            muE = sliders['XFEL']['µ(E) (1-2pk) / eV'].val
            chirp = sliders['XFEL']['chirp (1pk)'].val
            sigma_t = (dur,)
            sigma_E = (sigE,)
            corr_list = (chirp,)
            I_list = (1,)

            covs = covariance_from_correlation_2d(np.stack((sigma_t, sigma_E)), corr_list).T
            TEmap = MultivariateMapInterpolator.from_gauss_blob_list(np.array(((0, muE),)), covs, I_list)

            pe = ionizer_simple(β, TEmap, sliders['XFEL']['focal spot / m'].val, E_ionize, N_e)
            imdata = TEmap.map.T
            imextent = TEmap.domain.flatten()
            imextent[[0, 1]] *= 1e15

        auger_ratio = sliders['target']['Auger ratio'].val
        if auger_ratio > 0:
            pe += naive_auger_generator(
                pe, auger_ratio, 
                sliders['target']['Auger lifetime / s'].val, 
                sliders['target']['Auger KE / eV'].val * const.e
            )

        diff = time.perf_counter() - start
        per = diff/len(pe)
        stats_text2.set_text(f'Sampling took {diff*1e3:.1f} ms, {per*1e9:.0f} ns / e$^-$')

        teim.set_data(imdata)
        teim.set_extent(imextent)
        teim.autoscale()

        kebins = np.linspace(*np.quantile(pe.Ekin(), (discard / 2, 1 - discard / 2)), 100)
        tbins  = np.linspace(*np.quantile(pe.t0,     (discard / 2, 1 - discard / 2)), 100)
        ke, kex, key = np.histogram2d(pe.t0, pe.Ekin(), bins=(tbins, kebins))
        keim.set_data(ke.T)
        keim.set_extent((*kex[[0, -1]] * 1e15, *key[[0, -1]] / const.e))
        keim.autoscale()

        return update_streaking(None)

    def update_streaking(val):
        global pe, spe
        start = time.perf_counter()
        foc = sliders['streaking laser 1']['focal spot / m'].val

        beam = SimpleGaussianBeam(
            energy=sliders['streaking laser 1']['energy / J'].val,
            cep=sliders['streaking laser 1']['CEP'].val,
            envelope_offset=sliders['streaking laser 1']['delay / s'].val,
            wavelength=sliders['streaking laser 1']['wavelen. / m'].val,
            duration=sliders['streaking laser 1']['width / s'].val,
            focal_size=(foc, foc),
            polarization=ellipticity_to_jones_vector(sliders['streaking laser 1']['ellipticity'].val, sliders['streaking laser 1']['tilt'].val, 1),
            rotation=Rotation.from_euler('y', sliders['streaking laser 1']['cross. angle / rad'].val)
        )

        # h = sliders['streaking laser harmonics']['harmonic'].val
        # if h > 0:
        #     foc2 = sliders['streaking laser harmonics']['focal spot / m'].val
        #     beam += SimpleGaussianBeam(
        #         energy=sliders['streaking laser harmonics']['energy / J'].val,
        #         cep=sliders['streaking laser 1']['CEP'].val,
        #         envelope_offset=sliders['streaking laser harmonics']['delay / s'].val,
        #         wavelength=sliders['streaking laser 1']['wavelen. / m'].val / h,
        #         duration=sliders['streaking laser 1']['width / s'].val,
        #         focal_size=(foc2, foc2))

        #spe = classical_lorentz_streaker(pe, beam, (0, sliders['simulation']['time / s'].val), sliders['simulation']['stepsize / s'].val)
        spe = dumb_streaker(pe, beam)
        diff = time.perf_counter()-start
        per = diff/len(spe)
        stats_text1.set_text(f'Streaking took {diff*1e3:.1f} ms, {per*1e9:.0f} ns / e$^-$')

        return update_detector(None)

    def update_detector(val):
        global fb1, sp3, sp4

        acc = sliders['detector'][r'ϑ accept. / rad'].val / 2
        center = sliders['detector'][r'ϑ center / rad'].val
        phibincount = int(sliders['detector'][r'φ bins'].val)
        thetabincount = int(sliders['detector'][r'ϑ bins'].val)

        r = Rotation.from_euler('YX', (sliders['detector'][r'Y rotation / rad'].val, sliders['detector'][r'X rotation / rad'].val))

        # Plot of the acceptance region
        phidet = np.linspace(-np.pi, np.pi, 1000)
        thetadet_l = np.ones_like(phidet) * center - acc
        thetadet_h = np.ones_like(phidet) * center + acc
        _, thetadet_l, phidet_l = cartesian_to_spherical(*r.apply(spherical_to_cartesian(1, thetadet_l, phidet).T).T)
        _, thetadet_h, phidet_h = cartesian_to_spherical(*r.apply(spherical_to_cartesian(1, thetadet_h, phidet).T).T)
        phidet_l = (phidet_l + np.pi / 2) % (2 * np.pi)
        phidet_h = (phidet_h + np.pi / 2) % (2 * np.pi)
        # Try to roll angles to obtain non-self-intersecting polygon...
        i = 0
        while(np.abs(np.diff(phidet_l)).max() > 100 * np.abs(np.diff(phidet_l)).mean() and i < len(phidet)):
            phidet_l = np.roll(phidet_l, 1)
            thetadet_l = np.roll(thetadet_l, 1)
            i += 1
        i = 0
        while(np.abs(np.diff(phidet_h)).max() > 100 * np.abs(np.diff(phidet_h)).mean() and i < len(phidet)):
            phidet_h = np.roll(phidet_h, -1)
            thetadet_h = np.roll(thetadet_h, -1)
            i += 1
        xy = np.array((np.concatenate((phidet_l, phidet_h[::-1])), np.concatenate((thetadet_l, thetadet_h[::-1])))).T
        sp3.set_xy(xy)
        sp4.set_xy(xy)

        data1, x1, y1 = constant_polar_angle_ring(pe, center, acc, phibincount, 0.25, 'kinetic energy', 100, discard, (0, 0, sliders['detector'][r'z offset / m'].val), r)
        data2, x2, y2 = constant_polar_angle_ring(spe, center, acc, phibincount, 0.25, 'kinetic energy', 100, discard, (0, 0, sliders['detector'][r'z offset / m'].val), r)

        im1.set_data(data1.T)
        im2.set_data(data2.T)
        im1.set_extent((x1[0], x1[-1], y1[0], y1[-1]))
        im2.set_extent((x2[0], x2[-1], y2[0], y2[-1]))
        im1.autoscale()
        im2.autoscale()

        data3, x3, y3 = energy_integrated_4pi(pe, thetabincount, phibincount, 0.25, (0, 0, sliders['detector'][r'z offset / m'].val))
        data4, x4, y4 = energy_integrated_4pi(spe, thetabincount, phibincount, 0.25, (0, 0, sliders['detector'][r'z offset / m'].val))
        x3im, x4im = 0.5 * (x3[1:] + x3[:-1]), 0.5 * (x4[1:] + x4[:-1])
        y3im, y4im = 0.5 * (y3[1:] + y3[:-1]), 0.5 * (y4[1:] + y4[:-1])
        im3.set_data(x3im, y3im, data3.T)
        im4.set_data(x4im, y4im, data4.T)
        im3.autoscale()
        im4.autoscale()

        marg1x = data1.T.sum(axis=0)
        marg2x = data2.T.sum(axis=0)
        marg1y = data1.T.sum(axis=1)
        marg2y = data2.T.sum(axis=1)
        marg3y = data3.sum(axis=0)
        marg4y = data4.sum(axis=0)
        st1.set_data(marg1x, x1)
        st2.set_data(marg1x, x1)
        st3.set_data(marg2x, x2)
        st5.set_data(marg1y, y1)
        st6.set_data(marg2y, y2)
        st7.set_data(marg3y, y3)
        st8.set_data(marg4y, y4)
        st9.set_data(marg3y, y3)
        axmarg1x.set_ylim(np.max((marg1x.max(), marg2x.max())) * 1.05, 0)
        axmarg1y.set_xlim(0, marg1y.max() * 1.05)
        axmarg2y.set_xlim(0, marg2y.max() * 1.05)
        axmarg3y.set_xlim(0, marg3y.max() * 1.05)
        axmarg4y.set_xlim(0, np.max((marg3y.max(), marg4y.max())) * 1.05)

        return im1, im2, st1, st2, st3, st5, st6, st7, st8, st9
    # Sliders galore!
    sliders_spec = {
        'XFEL': {
            'pulse energy / J':   (1e-4,    1e-3,  None,  1e-4,   None,    update_electrons),
            'peaks':              (1,       10,    1,     2,      '%1d',   update_electrons),
            'width (1-2pk) / s':  (1e-17,   15e-15,None,  8e-16,  None,    update_electrons),
            'µ(E) (1-2pk) / eV':  (800,     6000,  None,  1200,   '%.0f',  update_electrons),
            'σ(E) (1-2pk) / eV':  (0.1,     60,    None,  0.5,    '%.1f',  update_electrons),
            'chirp (1pk)':        (-0.999,  0.999, None,  0,      '%.1f',  update_electrons),
            'distance (2pk) / s': (0,       1e-14, None,  7e-15,  None,    update_electrons),
            'focal spot / m':     (1e-6,    1e-4,  None,  2e-5,   None,    update_electrons),
        },
        'target': {
            #'photoelectrons':     (1e3,     5e5,   1,     1e5,    '%1d',   update_electrons),
            'length / m':         (1e-2,    1e-1,  None,  1e-2,   None,    update_electrons),
            'number density·cm³': (1e11,    1e15,  None,  1e11,   None,    update_electrons),
            'PE binding E / eV':  (500,     1500,  None,  1150,   '%.0f',  update_electrons),
            'β':                  (-1,      2,     None,  2,      '%.2f',  update_electrons),
            'Auger ratio':        (0,       1,     None,  0,      None,    update_electrons),
            'Auger lifetime / s': (1e-15,   1e-14, None,  22e-16, None,    update_electrons),
            'Auger KE / eV':      (50,      1000,  None,  60,    '%.1f',   update_electrons),
        },
        'streaking laser 1': {
            'cross. angle / rad': (-np.pi,  np.pi, None,  0,      '%1.2f', update_streaking),
            'focal spot / m':     (100e-6,  2e-3,  None,  5e-4,   None,    update_streaking),
            'wavelen. / m':       (1e-7,    10e-6, None,  10e-6,  None,    update_streaking),
            'width / s':          (1e-14,   1e-12, None,  3e-13,  None,    update_streaking),
            'delay / s':          (-1e-12,  1e-12, None,  0,      None,    update_streaking),
            'energy / J':         (0,       1e-3,  None,  30e-6,  None,    update_streaking),
            'CEP':                (0,       2*π,   None,  0,     '%1.2f',  update_streaking),
            'ellipticity':        (0,       1,     None,  1,     '%1.2f',  update_streaking),
            'tilt':               (0,       π,     None,  0,     '%1.2f',  update_streaking),
        },
        # 'streaking laser harmonics': {
        #     'harmonic':           (0,       3,     1,     0,      '%1d',   update_streaking),
        #     'focal spot / m':     (100e-6,  2e-3,  None,  5e-4,   None,    update_streaking),
        #     'wavelen. / m':       (1e-7,    10e-6, None,  10e-6,  None,    update_streaking),
        #     'delay / s':          (-1e-12,  1e-12, None,  0,      None,    update_streaking),
        #     'energy / J':         (0,       1e-3,  None,  0,      None,    update_streaking),
        # },
        # 'simulation': {
        #     'time / s':           (0,       1e-11, None,  1e-12,  None,    update_streaking),
        #     'stepsize / s':       (5e-15,   2e-14, None,  1e-14,  None,    update_streaking),
        # },
        'detector': {
            r'ϑ accept. / rad':   (0.01,    np.pi,  None,  0.25,  '%1.2f', update_detector),
            r'ϑ center / rad':    (0,       np.pi,  None,  np.pi/2,'%1.2f',update_detector),
            r'φ bins':            (8,       64,     8,     32,    '%1d',   update_detector),
            r'ϑ bins':            (8,       64,     8,     32,    '%1d',   update_detector),
            'z offset / m':       (-0.249,  0.249,  None,  0,     '%1.2f', update_detector),
            'Y rotation / rad':   (-np.pi/2,np.pi/2,None,  0,     '%1.2f', update_detector),
            'X rotation / rad':   (-np.pi/2,np.pi/2,None,  0,     '%1.2f', update_detector),
        }

    #        Name                  min      max    step   start   fmt       update function
    }

    gs_widgets = gs[:, 0].subgridspec(41, 1)
    stats_ax2 = fig.add_subplot(gs_widgets[39])
    stats_ax2.axis('off')
    stats_text2 = stats_ax2.text(0,0, "")
    stats_ax1 = fig.add_subplot(gs_widgets[40])
    stats_ax1.axis('off')
    stats_text1 = stats_ax1.text(0,0, "")
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

    frames = 600
    anirange = np.linspace(0, np.pi, frames)
    def animate(frame):#1e-17,     5e-15
        #sliders['streaking laser 1']['CEP'].set_val(frame / frames * 2 * np.pi)
        sliders['streaking laser 1']['tilt'].set_val(anirange[frame])
        #sliders['streaking laser 1']['focal spot / m'].set_val(frame / frames * (5e-4-1e-4) + 1e-4)
        return im1, im2, st1, st2, st3

    update_electrons(None)
    im1.autoscale()
    im2.autoscale()
    #plt.tight_layout(pad=1)

    figManager = plt.get_current_fig_manager()
    figManager.window.move(0, 0)
    figManager.window.showMaximized()
    figManager.window.setFocus()

    
    # set some parameters for anim
    #sliders['streaking laser 1']['focal spot / m'].set_val(1e-4)
    #sliders['streaking laser 1']['energy / J'].set_val(5e-4)
    #kebins2 = np.linspace(0, 200, 50)
    #kebins1 = np.linspace(48, 52, 50)
    plt.show()
    #anim = matplotlib.animation.FuncAnimation(fig, animate,
    #                           frames=frames, blit=True)
    #anim.save('simulations/build/anim_tilt_track.mp4', fps=60, dpi=100)

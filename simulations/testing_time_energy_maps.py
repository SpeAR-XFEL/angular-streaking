import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from angular_streaking.streaking.time_energy_map import Time_Energy_Map
from angular_streaking.streaking.stats import rejection_sampling_nD
import scipy.interpolate as si
from scipy.constants import c

N_G = 10

mu_t = np.random.normal(0, 5e-15, N_G)
mu_E = np.random.normal(1200, 3, N_G)
sigma_t = np.abs(np.random.normal(3e-15, 1e-15, N_G))
sigma_E = np.abs(np.random.normal(3, 1, N_G))
corr_list = np.random.uniform(-1, 1, N_G)
I_list = np.abs(np.random.normal(10, 3, N_G))
stepsizes = (1e-16, 0.1)


TEmap = Time_Energy_Map(
    mu_list=np.stack((mu_t, mu_E)),
    sigma_list=np.stack((sigma_t, sigma_E)),
    corr_list=corr_list,
    I_list=I_list,
    stepsizes=stepsizes,
)


gs = GridSpec(
    3, 2, width_ratios=[5, 1], height_ratios=[5, 2, 5], wspace=0.04, hspace=0.06
)

fig = plt.figure(1, figsize=(8, 8))
plt.clf()
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[4], sharex=ax1)
ax1.pcolormesh(TEmap.time_list * 1e15, TEmap.Ekin_list, TEmap.time_energy_map)
ax1.set_ylabel("photon energy in eV")
ax1.tick_params(axis="x", which="both", bottom=False)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.plot(TEmap.projection_Ekin / max(TEmap.projection_Ekin), TEmap.Ekin_list)
ax2.set_ylim(min(TEmap.Ekin_list), max(TEmap.Ekin_list))
ax2.set_xticks([0, 0.5, 1])
ax2.tick_params(axis="y", which="both", left=False)
ax2.grid(1)
plt.setp(ax2.get_yticklabels(), visible=False)

ax3.plot(TEmap.time_list * 1e15, TEmap.projection_time / max(TEmap.projection_time))
ax3.set_xlim(min(TEmap.time_list * 1e15), max(TEmap.time_list * 1e15))
ax3.set_yticks([0, 0.5, 1])
ax3.grid(1)
plt.setp(ax3.get_xticklabels(), visible=False)

plt.xlabel("time in fs")

fig.suptitle(f"Time-Energy Map of {N_G} randomly Gaussian-distributed 2D-Gaussians")

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

"""
Generate Electrons from the Map
"""

N_e = 100000
Neon_z_width = 1e-3
t_xray = TEmap.t1 - TEmap.t0
t_overlap = Neon_z_width / c + 2 * t_xray

spline = si.RectBivariateSpline(
    TEmap.time_list, TEmap.Ekin_list, TEmap.time_energy_map.T
)
pdf = spline.ev

t_elecs, E_photon = rejection_sampling_nD(
    pdf,
    [
        [TEmap.time_list.min(), TEmap.time_list.max()],
        [TEmap.Ekin_list.min(), TEmap.Ekin_list.max()],
    ],
    N_e,
)

ax4.plot(t_elecs * 1e15, E_photon, ".", ms=0.1, color="tab:red")
ax4.set_ylim(min(TEmap.Ekin_list), max(TEmap.Ekin_list))
ax4.grid(1)
ax4.set_ylabel("photon energy in eV")

hist_t, bins_t = np.histogram(t_elecs, bins=TEmap.time_list)
ax3.plot(
    (bins_t[1:] - (bins_t[1] - bins_t[0]) / 2) * 1e15,
    hist_t / max(hist_t),
    color="tab:red",
)

hist_E, bins_E = np.histogram(E_photon, bins=TEmap.Ekin_list)
ax2.plot(
    hist_E / max(hist_E), bins_E[1:] - (bins_E[1] - bins_E[0]) / 2, color="tab:red"
)

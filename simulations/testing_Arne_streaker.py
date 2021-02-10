import numpy as np
import matplotlib.pyplot as plt
from streaking.ionization import ionizer_Sauter
from streaking.time_energy_map import Time_Energy_Map
import scipy.constants as const
from streaking.streak import Arne_streaker
from streaking.gaussian_beam import SimpleGaussianBeam
from matplotlib.gridspec import GridSpec

### generate time-energy map
N_G = 10

mu_t = np.random.normal(0, 5e-15, N_G) # s
mu_E = np.random.normal(1200, 5, N_G) # eV
sigma_t = np.abs(np.random.normal(.4e-15, 0.2e-15, N_G))
sigma_E = np.abs(np.random.normal(2, 1, N_G))
corr_list = np.random.normal(0, 0, N_G)
I_list = np.abs(np.random.normal(10, 1, N_G))
stepsizes = (1e-16, 0.1)

TEmap = Time_Energy_Map(
    mu_list=np.stack((mu_t, mu_E)),
    sigma_list=np.stack((sigma_t, sigma_E)),
    corr_list=corr_list,
    I_list=I_list,
    stepsizes=stepsizes,
)

### generate electrons according to time-energy map
N_e=1000
E_ionize=900 # eV

elecs = ionizer_Sauter(TEmap,E_ionize,N_e)
angles_start=np.arctan2(elecs.p[:,1],elecs.p[:,0])
Ekin_start=np.copy(elecs.Ekin())


streaked_elecs=Arne_streaker(elecs, SimpleGaussianBeam(energy=30e-6,cep=0),TEmap.t0,1e-16,10000)

streaked_angles=(np.arctan2(streaked_elecs.p[:,1],streaked_elecs.p[:,0])-np.pi/2)%(2*np.pi)

#bins = [np.linspace(0, 2 * np.pi, 51), 51]
#plt.hist2d((angles + np.pi / 2) % (2 * np.pi), Ekins / const.e, bins=bins)

gs = GridSpec(
    4, 2, width_ratios=[5, 1], height_ratios=[3, 2, 5, 2], wspace=0.1, hspace=0.2
)

fig=plt.figure(1,figsize=(8,8))
plt.clf()
ax1=fig.add_subplot(gs[0])
ax1.pcolormesh(TEmap.time_list * 1e15, TEmap.Ekin_list, TEmap.time_energy_map)
ax1.set_ylabel("photon energy in eV")
ax1.tick_params(axis="x", which="both", bottom=False)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_xlabel('xray pulse time axis in fs')    
ax1.xaxis.set_label_position('top')

ax2=fig.add_subplot(gs[1],sharey=ax1)
ax2.plot(TEmap.projection_Ekin / max(TEmap.projection_Ekin), TEmap.Ekin_list)
ax2.set_ylim(min(TEmap.Ekin_list), max(TEmap.Ekin_list))
ax2.set_xticks([0, 0.5, 1])
ax2.tick_params(axis="y", which="both", left=False)
ax2.grid(1)
plt.setp(ax2.get_yticklabels(), visible=False)

ax3=fig.add_subplot(gs[2],sharex=ax1)
ax3.plot(TEmap.time_list * 1e15, TEmap.projection_time / max(TEmap.projection_time))
ax3.set_xlim(min(TEmap.time_list * 1e15), max(TEmap.time_list * 1e15))
ax3.set_yticks([0, 0.5, 1])
ax3.grid(1)

ax5=fig.add_subplot(gs[4])
angle_edges=np.linspace(0, 2*np.pi, 17)
Ekin_edges=np.linspace(250,350, 51)
H, xedges, yedges = np.histogram2d(
    streaked_angles,
    streaked_elecs.Ekin()/const.e,
    [angle_edges, Ekin_edges],
)

ax5.pcolormesh(angle_edges[:-1],Ekin_edges[:-1],H.T)
ax5.set_ylabel("electron kinetic energy in eV")
ax5.tick_params(axis="x", which="both", bottom=False)
plt.setp(ax5.get_xticklabels(), visible=False)

ax6=fig.add_subplot(gs[5],sharey=ax5)
Ekin_hist,bins=np.histogram(streaked_elecs.Ekin()/const.e,bins=Ekin_edges)
ax6.tick_params(axis="y", which="both", left=False)
plt.plot(Ekin_hist/max(Ekin_hist),Ekin_edges[:-1])
plt.setp(ax6.get_yticklabels(), visible=False)
ax6.set_xticks([0, 0.5, 1])
ax6.grid(1)
ax6.set_ylim([Ekin_edges[0],Ekin_edges[-2]])

ax7=fig.add_subplot(gs[6],sharex=ax5)
angles_hist,bins=np.histogram(streaked_angles,bins=angle_edges)
plt.plot(angle_edges[:-1],angles_hist/max(angles_hist))
ax7.set_xlabel('streaked electron angle in rad')
ax7.set_yticks([0, 0.5, 1])
ax7.grid(1)
ax7.set_xlim([angle_edges[0],angle_edges[-2]])
# ax5.scatter(streaked_angles,streaked_elecs.Ekin()/const.e)

# plt.subplot(121)
# bins = [np.linspace(0, 2 * np.pi, 51), 51]
# plt.hist2d((angles_start + np.pi / 2) % (2 * np.pi), Ekin_start / const.e, bins=bins)
# plt.title("Unstreaked")
# plt.subplot(122)
# plt.title("Streaked")
# plt.hist2d((streaked_angles + np.pi / 2) % (2 * np.pi), streaked_elecs.Ekin() / const.e, bins=bins)
# plt.show()

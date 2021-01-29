import functions as func
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
### Inputs ###

    # Laser Parameter
# wavelength in m
wavelength=10e-6    
# pulse energy in J
E=30e-6             
# laser phase in rad at t=0
phi=0     
# temporal distance in s between XFEL and laser pulse
t_offset=0e-15     

# generate time-energy map of the photoelectrons
Ekin_mu_list=[300,300]
Ekin_sigma_list=[2,0.6]
times_mu_list=[-2e-15,2e-15]
times_sigma_list=[0.2e-15,.8e-15]
corr_list=[0,0]

time_energy_map,times_list,Ekin_list=func.generate_time_energy_map(Ekin_mu_list, Ekin_sigma_list, times_mu_list, times_sigma_list, corr_list)
time_energy_map=time_energy_map/np.sum(time_energy_map) # "normalize"
N_tsteps_emission=time_energy_map.shape[1]

N=200
elecs_per_timestep=np.rint(np.sum(time_energy_map,axis=0)*N).astype(int) 
N_elecs=np.sum(elecs_per_timestep) # due to the rounding, usually slightly less than the defined N

phase=phi+t_offset*2*np.pi*func.c/wavelength
phi_deg=np.rad2deg(phase)
z_start=t_offset*func.c
laserfield=func.get_laser(
      wavelength=10e-6,
      x0=1000e-6/2.3548, # sigma width of horizontal focus in m
      y0=1000e-6/2.3548,
      E=E,
      phi=phase,
      z_start=z_start)#30e-6)


#x_coords=np.ones((N,N_tsteps+1))
#x_coords[:,0]=elecs[0,:]
#
#y_coords=np.ones((N,N_tsteps+1))
#y_coords[:,0]=elecs[1,:]
t=np.copy(times_list[0])

N_tsteps=10000
timestep=1e-16

elecs=np.zeros((6,N_elecs))
birthtimes=np.zeros(N_elecs)
birth_angles=np.zeros(N_elecs)
N_now=0

## 
for i in range(N_tsteps_emission):
   num_new=elecs_per_timestep[i]
   if num_new != 0:
      times_generated=timestep*np.random.rand(num_new)
      birthtimes[N_now:N_now+num_new]=t+times_generated
      new_elecs=func.generate_electron(num_new,
                                       pdf_Ekin= lambda E: np.interp(E,Ekin_list,time_energy_map[:,i])/np.max(time_energy_map[:,i]))
      birth_angles[N_now:N_now+num_new]=np.arctan2(new_elecs[4,:],new_elecs[3,:])
      elecs[:,N_now:N_now+num_new],dummy=func.interaction_step(laserfield,new_elecs,t,tstep=times_generated)

      
   elecs[:,:N_now],t=func.interaction_step(laserfield,elecs[:,:N_now],t,tstep=timestep)
   N_now+=num_new
   
for i in range(N_tsteps):
   elecs,t=func.interaction_step(laserfield,elecs,t,tstep=timestep)
#   x_coords[:,i+1]=np.copy(elecs[0,:])
#   y_coords[:,i+1]=np.copy(elecs[1,:])
   
angles_spatial=np.arctan2(elecs[1,:],elecs[0,:])
angles_momentum=np.arctan2(elecs[4,:],elecs[3,:])
p_abs_list=np.sqrt(np.sum(elecs[3:,:]**2,axis=0))
Ekin_final_list=p_abs_list**2/2/func.e_m/func.e_charge
#plt.plot(x_coords.T*1e6,y_coords.T*1e6,'-',color='tab:blue')
#plt.plot(x_coords[:,-1]*1e6,y_coords[:,-1]*1e6,'o',color='tab:blue')

Ekin_min=270
Ekin_max=330
H,xedges,yedges=np.histogram2d(angles_spatial,Ekin_final_list,[np.linspace(-np.pi,np.pi*15/16,16),np.linspace(Ekin_min,Ekin_max,30)])

X,Y=np.meshgrid(xedges,yedges)


plt.figure(1,figsize=(10,6))
plt.clf()

plt.subplot(2,1,1)
plt.imshow(time_energy_map,extent=[min(times_list)*1e15,max(times_list)*1e15,min(Ekin_list)+800,max(Ekin_list)+800],aspect='auto',origin='lower')
plt.xlabel('time in fs')
plt.ylabel('$E_{kin}$ in eV')
plt.title('Photoelectrons, energy vs. time')

ax=plt.subplot(2,2,3)
ax.set_axisbelow(True) # for the grid lines to be below the scatter points (pure cosmetic)
plt.grid(1,ls='--')
plt.scatter(angles_spatial,Ekin_final_list,c=birthtimes*1e15,cmap=cm.copper, marker='.', label=f'dummy')
plt.colorbar(label='electron birth time in fs')

plt.xlabel('angles in rad')
plt.ylabel('$E_{kin}$ in eV')
plt.xlim(-np.pi,np.pi)
plt.ylim(Ekin_min,Ekin_max)
plt.title('Photoelectrons after 1ps')

plt.subplot(2,2,4)
plt.title('Photoelectrons after 1ps')
plt.imshow(H.T,extent=[-np.pi,np.pi*15/16,Ekin_min,Ekin_max],aspect='auto',origin='lower')
plt.xlabel('angles in rad')

plt.tight_layout()

#plt.legend()
#plt.xlabel('x in $\mu$m')
#plt.ylabel('y in $\mu$m')
#   
   
   
   
   
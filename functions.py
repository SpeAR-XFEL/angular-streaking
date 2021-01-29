import numpy as np
##### natural constants #####
c=299792458        # speed of light
e_charge=1.6021766208e-19 # electron charge
e_m=9.10938356e-31      # electron mass in eV/c^2
Z0=376.73          # impedance of free space in Ohm
epsilon_0=8.854e-12 # vacuum permittivity

def get_laser(
      focal_point_x=1e-15,
      focal_point_y=1e-15,
      z_start=0,
      phi=0,
      wavelength=10e-6, # wavelength in m
      M2=1.,
      E=30*1e-6, # pulse energy in J
      duration=300e-15, # FWHM pulse length in s
      x0=500e-6/2.3548, # sigma width of horizontal focus in m
      y0=500e-6/2.3548): # sigma width of vertical focus in m

   sigma_t=duration/(2*np.sqrt(2*np.log(2))) #FWHM -> sigma for Gaussian distribution
   w0_x=2*x0
   w0_y=2*y0
   k=2*np.pi/wavelength # wavenumber in 1/m
   omega=2*np.pi*c/wavelength # angular frequency in rad/s
   sigma_z=duration*c/2.3548 # sigma width of pulse length in m
   zRx=np.pi*w0_x**2/(M2*wavelength) # horizontal Rayleigh length in m
   zRy=np.pi*w0_y**2/(M2*wavelength) # vertical Rayleigh length in m
   w_x=lambda z: w0_x*(np.sqrt(1+z**2/(zRx**2))) # horizontal beam size w_x at position z in m
   w_y=lambda z: w0_y*(np.sqrt(1+z**2/(zRy**2))) # vertical beam size w_y at position z in m
      
   E0=2**-0.25*np.pi**-0.75*np.sqrt(Z0*E/(x0*y0*sigma_t))
   
   def Efield(X,Y,Z,T):
      Zdif_x = Z-focal_point_x # Distance of electron to focus (mod1_center)
      Zdif_y = Z-focal_point_y
      Zlas=z_start+c*T  # Position of the laser pulse center
      R_x=Zdif_x*(1+(zRx/Zdif_x)**2)
      R_y=Zdif_y*(1+(zRy/Zdif_y)**2)
      
      central_E_field=E0*w0_x/(w_x(Zdif_x))
      offaxis_pulsed_factor=np.exp(-(Y/w_y(Zdif_y))**2-(X/w_x(Zdif_x))**2-((Z-Zlas)/(2*sigma_z))**2)
      phase=k*(Z-z_start)-omega*T+k/2*X**2/R_x+k/2*Y**2/R_y-0.5*np.arctan(Zdif_x/zRx)-0.5*np.arctan(Zdif_y/zRy)+phi
      polarization_vec=np.vstack((np.cos(phase),np.sin(phase),np.zeros(len(phase))))
      return central_E_field*offaxis_pulsed_factor*polarization_vec
   return Efield


def interaction_step(
      laser_field,
      elec,
      t,
      tstep=1e-12):
   p_vec_list=elec[3:,:]
   p_abs_list=np.sqrt(np.sum(p_vec_list**2,axis=0))
   gamma_list=np.sqrt((p_abs_list/e_m/c)**2+1)
   
   laserE=laser_field(elec[0,:],elec[1,:],elec[2,:],t)
   laserB=np.cross([0,0,1],laserE.T)/c
   F_lor=-e_charge*(laserE+np.cross(p_vec_list.T,laserB).T/e_m/gamma_list)
   dp_vec_list=F_lor*tstep
   p_vec_list_new=p_vec_list+dp_vec_list

   p_abs_list_new=np.sqrt(np.sum(p_vec_list_new**2,axis=0))
   gamma_list_new=np.sqrt((p_abs_list_new/e_m/c)**2+1)        
                      
   spatial_new=elec[:3,:]+p_vec_list_new/e_m/gamma_list_new*tstep   
   elec[0:3,:]=np.copy(spatial_new)
   elec[3:,:]=np.copy(p_vec_list_new)
       
   return elec,t+tstep

def generate_electron(
      num,
      nu=1100, #photon energy in eV
      ionization=800, #ionization energy in eV
      pdf_Ekin= lambda E: np.exp(-(E-300)**2/(2*1**2)),   # gaussian distribution at mu=300eV and sigma=5eV
      Ekin_limits=[280,320]
      ):
   elecs=np.zeros((6,num))
   beta=2
   pdf_theta=lambda theta: 1+beta/2 * (3*np.cos(theta)**2-1)
   for i in range(num):
      Ekin=rejection_sampling(pdf_Ekin,1,Ekin_limits)*e_charge
      p_abs=np.sqrt(2*e_m*Ekin)  # classical mechanics
      
      k=1+beta
      limits=[0,2*np.pi]
      theta=rejection_sampling(pdf_theta,k,limits)
      elecs[:,i]=np.array([0,0,0,np.sin(theta)*p_abs,np.cos(theta)*p_abs,0])
   return elecs

def generate_time_energy_map(
        Ekin_mu_list=[303,300],
        Ekin_sigma_list=[3,0.6],
        times_mu_list=[-2e-15,0],
        times_sigma_list=[0.2e-15,2e-15],
        corr_list=[-0.9,0.3],
        Ekin_list=np.linspace(280,320,100),
        times_list=np.arange(-10e-15,10e-15,1e-16)):
    
    time_energy_map=np.zeros((len(Ekin_list),len(times_list)))
    N_pulses=len(Ekin_mu_list)
    X,Y=np.meshgrid(times_list,Ekin_list)
    for i in range(N_pulses):
        mu1=Ekin_mu_list[i]
        sigma1=Ekin_sigma_list[i]
        mu2=times_mu_list[i]
        sigma2=times_sigma_list[i]
        corr=corr_list[i]
        time_energy_map=time_energy_map+1/(2*np.pi*sigma1*sigma2*np.sqrt(1-corr**2))*np.exp(-1/(2-2*corr**2)*((X-mu2)**2/sigma2**2+(Y-mu1)**2/sigma1**2-2*corr*(X-mu2)*(Y-mu1)/sigma1/sigma2))
        # time_energy_map2=1/(2*np.pi*sigma_Ekin*sigma_times*np.sqrt(1-correlation**2))*np.exp(-1/(2-2*correlation**2)*((X-mu_times)**2/sigma_times**2+(Y-mu_Ekin)**2/sigma_Ekin-2*correlation*(X-mu_times)*(Y-mu_Ekin)/sigma_Ekin/sigma_times))
    return time_energy_map,times_list,Ekin_list

def rejection_sampling(
      F_pdf,
      k,
      limits):
   p=np.random.rand()
   x=np.random.uniform(limits[0],limits[1],None)
   while p*k>F_pdf(x):
      x=np.random.uniform(limits[0],limits[1],None)
   return x




































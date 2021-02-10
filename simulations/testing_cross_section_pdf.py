import numpy as np
import matplotlib.pyplot as plt
from streaking.ionization import diff_cross_section_dipole,diff_cross_section_Sauter_lowEnergy,diff_cross_section_Sauter
from scipy.constants import c,e,m_e

#electron kinetic energy
Ekin=1*e
gamma=1+Ekin/(m_e*c**2)

theta=np.linspace(0,np.pi,1000)
phi=np.linspace(-np.pi,np.pi,1000)

x,y=np.meshgrid(theta,phi)

y1=diff_cross_section_dipole(phi,2)
y2=diff_cross_section_Sauter_lowEnergy(np.pi/2,phi)
z1=diff_cross_section_Sauter_lowEnergy(x,y)
z2=diff_cross_section_Sauter(x,y,gamma)
#
plt.figure()
plt.pcolormesh(x,y,z1/np.max(z1))
plt.colorbar()
plt.figure()
plt.pcolormesh(x,y,z2/np.max(z2))
plt.colorbar()
plt.figure()
plt.pcolormesh(x,y,z1/np.max(z1)-z2/np.max(z2))
plt.colorbar()
#plt.xlabel(r'polar angle $\theta$')
#plt.ylabel(r'azimuthal angle $\varphi$')
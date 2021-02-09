import numpy as np
import matplotlib.pyplot as plt
from streaking.ionization import diff_cross_section_dipole,diff_cross_section_Sauter2
from scipy.constants import c,e,m_e

#electron kinetic energy
Ekin=300*e
gamma=1+Ekin/(m_e*c**2)
beta=np.sqrt(1-1/gamma**2)

theta=np.linspace(-np.pi,np.pi,10000)
y1=diff_cross_section_dipole(theta,2)
y2=diff_cross_section_Sauter2(theta,1,beta,gamma)

plt.plot(theta,y1/max(y1))
plt.plot(theta,y2/max(y2))
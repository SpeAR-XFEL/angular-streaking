import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.interpolate as si


class Time_Energy_Map:
    """
   generates a 2D map of intensity vs photon energy and time
   inputs are either
      - a already existing .png (for comical purposes)
      - a set of 2D Gaussian peaks given via mu, sigma, corr, I
   """

    def __init__(
        self,
        image_path=None,
        Ekin_range=None,
        time_range=None,
        mu_list=[[-2e-15, 0], [1203, 1200]],
        sigma_list=[[0.8e-15, 1e-15], [3, 0.6]],
        I_list=[1, 1],
        corr_list=[-0.9, 0.3],
        stepsizes=(1e-16, 0.1),
    ):
        """
        Parameters
        ----------
        image_path : string, optional
            directory of a greyscale .png, describing energy vs time of the x-ray pulse
        Ekin_range : tuple of scalar, optional (required if image_path is given)
            minimum and maximum values of the kinetic energy range given in the .png in eV
        time_range : tuple of scalar, optional (required if image_path is given)
            minimum and maximum values of the time range given in the .png in s
        mu_list : (N,2) array_like
            center of gaussian peaks, in (s,eV)
        sigma_list : (N,2) array_like
            sigma widths of Gaussian peaks, in (s,eV)
        I_list : (N) array_like
            list of (relative) peak intensities
        corr_list : (N) array_like
            list of 2D-Gaussian correlation coefficients
        stepsizes : tupel
            resolution of the time-energy map given in (s,eV)
        """
        if image_path is not None:
            time_energy_map = gaussian_filter(
                np.flipud(-1 * plt.imread(image_path)[:, :, 0] + 1), sigma=3
            )

            self.Ekin_list = np.linspace(
                Ekin_range[0], Ekin_range[1], self.time_energy_map.shape[0]
            )
            self.time_list = np.arange(
                time_range[0], time_range[1], self.time_energy_map.shape[1]
            )
        else:

            mu_list = np.array(mu_list)
            sigma_list = np.array(sigma_list)
            corr_list = np.array(corr_list)

            # the map should include at least a distance of 4 sigma to each Gaussian peak
            time_min = np.min(mu_list[0, :] - 4 * sigma_list[0, :])
            time_max = np.max(mu_list[0, :] + 4 * sigma_list[0, :])
            Ekin_min = np.min(mu_list[1, :] - 4 * sigma_list[1, :])
            Ekin_max = np.max(mu_list[1, :] + 4 * sigma_list[1, :])

            self.time_list = np.linspace(
                time_min,
                time_max,
                np.rint((time_max - time_min) / stepsizes[0]).astype(int) + 1,
            )
            self.Ekin_list = np.linspace(
                Ekin_min,
                Ekin_max,
                np.rint((Ekin_max - Ekin_min) / stepsizes[1]).astype(int) + 1,
            )

            time_energy_map = np.zeros((len(self.Ekin_list), len(self.time_list)))
            N_pulses = mu_list.shape[1]
            X, Y = np.meshgrid(self.time_list, self.Ekin_list)
            for i in range(N_pulses):
                mu1 = mu_list[0, i]
                sigma1 = sigma_list[0, i]
                mu2 = mu_list[1, i]
                sigma2 = sigma_list[1, i]
                corr = corr_list[i]
                time_energy_map = time_energy_map + I_list[i] / (
                    2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - corr ** 2)
                ) * np.exp(
                    -1
                    / (2 - 2 * corr ** 2)
                    * (
                        (X - mu1) ** 2 / sigma1 ** 2
                        + (Y - mu2) ** 2 / sigma2 ** 2
                        - 2 * corr * (X - mu1) * (Y - mu2) / sigma1 / sigma2
                    )
                )
            time_energy_map = time_energy_map / np.max(time_energy_map)
        self.time_energy_map = time_energy_map
        self.t0 = self.time_list[0]
        self.t1 = self.time_list[-1]
        self.E0 = self.Ekin_list[0]
        self.E1 = self.Ekin_list[-1]
        self.projection_time = np.sum(self.time_energy_map, axis=0)
        self.projection_Ekin = np.sum(self.time_energy_map, axis=1)
        self.spline = si.RectBivariateSpline(
            self.time_list, self.Ekin_list, self.time_energy_map.T
        )

    def pdf(self, t, Ekin):
        return self.spline.ev(t, Ekin)

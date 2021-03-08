import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si


class MultivariateMapInterpolator:
    """
    Holds a multivariate discrete map in defined parameter range.
    Allows adding something to the map and calculating interpolated values anywhere
    in the domain of definition and some other nice features.
   """

    def __init__(
        self,
        domain,
        resolution,
        normalized=True
    ):
        """
        Parameters
        ----------
        domain : array_like, shape (N, 2)
            Domain of defintion for all N dimensions.
        resolution : array_like, shape (N,)
            Size of the discrete map in every dimesion
        normalized : boolean, optional
            Whether to keep the map normalized to a maximum value of 1.
        """
        self.domain = np.asarray(domain)
        self.resolution = resolution
        self.map = np.zeros(resolution)
        self.normalized = normalized
        self.ranges = [np.linspace(start, stop, size) for start, stop, size in zip(*self.domain.T, resolution)]
        self.grid = np.moveaxis(np.meshgrid(*self.ranges, indexing='ij'), 0, -1)
        self.interp = si.RegularGridInterpolator(self.ranges, self.map, bounds_error=False, method='nearest')

    @classmethod
    def from_image(cls, path, domain):
        im = np.rot90(plt.imread(path), k=-1)
        interpolator = cls(domain, im.shape)
        interpolator.add_arbitrary(im)
        return interpolator

    @classmethod
    def from_gauss_blob_list(cls, means, covs, scales, resolution=(200, 200), sigma=3):
        diagonals = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1))
        domain = np.array((np.min(means - sigma * diagonals, axis=0), np.max(means + sigma * diagonals, axis=0))).T
        interpolator = cls(domain, resolution)
        for mean, cov, scale in zip(means, covs, scales):
            interpolator.add_gauss_blob(mean, cov, scale)
        interpolator.normalize()
        return interpolator

    def normalize(self):
        if self.normalized:
            m = self.map.max()
            if m != 0:
                self.map /= self.map.max()

    def add_gauss_blob(self, means, cov, scale=1):
        Σinv = np.linalg.inv(cov)
        xmu = means - self.grid
        self.map += scale * np.exp(-0.5 * np.einsum('...i,ij,...j', xmu, Σinv, xmu))

    def add_arbitrary(self, map_):
        self.map += map_
        self.normalize()

    def eval(self, points):
        return self.interp(points)

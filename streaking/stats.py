import numpy as np


def rejection_sampling(pdf, parameter_range, samples, params=()):
    """
    Rejection sampling on the (optionally multivariate) probability density function `pdf`.
    As a fixed number of samples is provided, the runtime drastically depends on  how evenly
    the pdf is distributed. The PDF is expected to be normalized to its maximum.

    Parameters
    ----------
    pdf : function
       Multivariate probability density function (N variables), normalized to the maximum
    parameter_range : array_like, shape (N, 2) or (2,)
       Parameter ranges for each dimension
    samples : int
       Number of samples
    params : iterable
       Additional parameters passed to `pdf`, optional

    Returns
    -------
    rand : ndarray, shape (N, samples)
       Drawn random samples
    """
    parameter_range = np.atleast_2d(parameter_range)
    ndim = parameter_range.shape[0]

    rejected = np.full(samples, True)
    rand = np.empty((samples, ndim))
    rejection = np.empty(samples)
    sum_rejected = samples
    while sum_rejected > 0:
        rand[rejected] = np.random.uniform(*parameter_range.T, (sum_rejected, ndim))
        rejection[rejected] = np.random.rand(sum_rejected)
        rejected[rejected] = rejection[rejected] > pdf(*rand[rejected].T, *params)
        sum_rejected = np.sum(rejected)

    return np.squeeze(rand.T)

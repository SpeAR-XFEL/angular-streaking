import numpy as np


def covariance_from_correlation_2d(stds, cor):
    return np.array(
        (
            (stds[0] ** 2, cor * stds[0] * stds[1]),
            (cor * stds[0] * stds[1], stds[1] ** 2),
        )
    )


def rejection_sampling(pdf, parameter_range, samples, oversampling=2, params=()):
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
    oversampling : scalar
        Initial sampling factor to reduce number if required iterations, optional
    params : iterable
        Additional parameters passed to `pdf`, optional

    Returns
    -------
    rand : ndarray, shape (N, samples)
        Drawn random samples
    """
    parameter_range = np.atleast_2d(parameter_range)
    ndim = parameter_range.shape[0]

    osamples = int(samples * oversampling)

    rejected = np.full(osamples, True)
    rand = np.empty((osamples, ndim))
    rejection = np.empty(osamples)
    sum_rejected = osamples
    while sum_rejected > osamples - samples:
        rand[rejected] = np.random.uniform(*parameter_range.T, (sum_rejected, ndim))
        rejection[rejected] = np.random.rand(sum_rejected)
        rejected[rejected] = rejection[rejected] > pdf(rand[rejected], *params)
        sum_rejected = np.sum(rejected)
    rand = rand[~rejected][:samples]
    return np.squeeze(rand.T)

import numpy as np
import scipy


def rejection_sampling(pdf, parameter_range, samples, params=()):
    # Funny solution I guess...
    # Worst case run-time is infinite, but that’s improbable.

    # Find global maximum of target PDF to scale the uniform distribution.
    M = -scipy.optimize.shgo(
        lambda x: -pdf(x, *params), (parameter_range,), iters=8
    ).fun

    rejected = np.full(samples, True)
    rand = np.empty(samples)
    rejection = np.empty(samples)
    sum_rejected = samples
    while sum_rejected > 0:
        rand[rejected] = np.random.uniform(*parameter_range, sum_rejected)
        rejection[rejected] = M * np.random.rand(sum_rejected)
        rejected[rejected] = rejection[rejected] > pdf(rand[rejected], *params)
        sum_rejected = np.sum(rejected)
    return rand


def rejection_sampling_nD(pdf, parameter_range, samples, params=()):

    """
    draws (x1,...,xN) distributed according to pdf(x1,...,xN)
    Runtime drastically depends on how even the pdf is distributed. 
    
    Parameters:
    pdf : function
       N-D probability density function, normalized to the maximum
    parameter_range : (N,2) array_like
       each row consists of the min and max values for the dimensions
    samples : int
       number of pairs to be drawn
    params : 
       whatever else you need for the pdf
    
    Return:
    rand : (N,samples) array_like
       each column is a set of (x1,...,xN) 
    """
    Ndim = len(parameter_range)

    rejected = np.full(samples, True)
    rand = np.empty((Ndim, samples))
    rejection = np.empty(samples)
    sum_rejected = samples
    while sum_rejected > 0:
        for i in range(Ndim):
            rand[i, :][rejected] = np.random.uniform(*parameter_range[i], sum_rejected)
        rejection[rejected] = np.random.rand(sum_rejected)
        rand_rejected = rand[:, rejected]
        rejected[rejected] = rejection[rejected] > pdf(*rand_rejected, *params)
        sum_rejected = np.sum(rejected)
    return rand

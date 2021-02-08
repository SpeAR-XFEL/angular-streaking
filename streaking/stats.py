import numpy as np
import scipy


def rejection_sampling(pdf, parameter_range, samples, params=()):
    # Funny solution I guess...
    # Worst case run-time is infinite, but that’s improbable.

    # Find global maximum of target PDF to scale the uniform distribution.
    M = -scipy.optimize.shgo(lambda x: -pdf(x, *params), (parameter_range,), iters=8).fun

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

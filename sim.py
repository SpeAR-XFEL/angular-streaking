import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize

def rejection_sampling(pdf, parameter_range, samples):
    # Funny solution I guess...
    # Worst case run-time is infinite.

    # Find global maximum of target PDF to scale the uniform distribution.
    M = -1.01 * scipy.optimize.shgo(lambda x: -pdf(x), (parameter_range,), iters=8).fun

    rejected = np.full(samples, True)
    rand = np.empty(samples)
    rejection = np.empty(samples)
    sum_rejected = samples
    while sum_rejected > 0:
        rand[rejected] = np.random.uniform(*parameter_range, sum_rejected)
        rejection[rejected] = M * np.random.rand(sum_rejected)
        rejected[rejected] = rejection[rejected] > pdf(rand[rejected])
        sum_rejected = np.sum(rejected)
    return rand


def ionize(β, IX, EX, EB, E_range, t_range, electrons):
    ϑpdf = lambda ϑ: (1 + β * 1/2 * (3 * np.cos(ϑ)**2 - 1))
    ϑ = rejection_sampling(ϑpdf, (-np.pi, np.pi), electrons)
    ψ = np.random.uniform(-np.pi, np.pi, electrons)
    t0 = rejection_sampling(IX, t_range, electrons) # birth time
    # not time-dependent for now. needs to be to account for chirp.
    E = rejection_sampling(EX, E_range, electrons) - EB
    return ϑ, ψ, t0, E


def histogram(samples, pdf=None):
    x = np.linspace(samples.min(), samples.max(), 100)
    plt.hist(samples, density=True, bins=50, alpha=0.5)
    if pdf:
        plt.plot(x, pdf(x), 'C0-')
    plt.xlim((samples.min(), samples.max()))
    plt.tight_layout(pad=0)
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Double Gauss pulse to check the rejection sampling
    XFEL_intensity = lambda t: (0.6 * scipy.stats.norm(0, 1e-15).pdf(t) + 
                                0.4 * scipy.stats.norm(3e-15, 1e-15).pdf(t))
    
    XFEL_photon_energy = scipy.stats.norm(914, 2).pdf

    ϑ, ψ, t0, E = ionize(
        2, # β
        XFEL_intensity, 
        XFEL_photon_energy, 
        870.2, # binding energy
        (900, 960), # considered energy range
        (-5e-15, 7e-15), # considered time range
        100000 # number of electrons to generate (not yet based on cross section)
    )
    # histogram(t0, XFEL_intensity)
    # histogram(E)

    # Plotting
    
    ax = plt.subplot(111, projection='polar')
    H, xedges, yedges = np.histogram2d(ϑ, E, bins=(np.linspace(-np.pi, np.pi, 50+1), 50))
    ax.pcolormesh(*np.meshgrid(xedges, yedges), H.T, snap=True)
    hist1d = H.T.sum(axis=0)
    hist1d /= hist1d.max()
    ax.plot(xedges[:-1] + (xedges[1]-xedges[0])/2, hist1d * (yedges[-1]-yedges[0]) + yedges[0], 'C3.')
    #ax.set_xlabel('ϑ / rad')
    #ax.set_ylabel('E / eV')
    ax.set_ylim(yedges[0], yedges[-1])
    ax.tick_params(axis='y', colors='white')
    plt.setp(ax.get_yticklabels(), alpha=0.7)
    #ax.spines['end']  .set_color('white')
    #ax.spines['start'].set_color('white')
    ax.grid(True, alpha=0.25)
    #plt.tight_layout(pad=0.5)
    plt.show()
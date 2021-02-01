from streaking.components.gaussian_beam import SimpleGaussianBeam
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    beam = SimpleGaussianBeam()
    z = np.linspace(-2e-4, 2e-4, 1000)
    x = y = np.zeros_like(z) + 0.001
    E, B = beam.fields(x, y, z, 0)
    plt.plot(z, E[1])
    plt.show()

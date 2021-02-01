import streaking
from streaking.components import gaussian_beam
import numpy as np

if __name__ == "__main__":
    beam = gaussian_beam.GaussianBeam(1, 2, 3)
    print(beam.electric_field(1, 2))

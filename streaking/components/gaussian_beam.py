import numpy as np

class GaussianBeam:
    """Add documentation
    
    Methods
    -------
    electric_field(r, t)
        Calculates the electric field of the gaussian beam
        at every given position and time.
    """

    def __init__(self, a, b, c):
        """Add more documentation"""
        self.a = a
        self.b = b
        self.c = c
    
    def electric_field(self, r, t):
        """Calculates the electric field of the gaussian beam
        at every given position and time.

        Parameters
        ----------
        r : (N, 3) array_like
            Position array.
        t : (...) array_like
            Time array.

        Returns
        -------
        E : (..., N, 3) array_like
            Electric field vectors.
        """
        return 1 + 2 + 3
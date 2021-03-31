import numpy as np
import scipy.constants as const


class SimpleGaussianBeam:
    """Calculates electric and magnetic field of a simple
    gaussian laser beam on the z axis.

    Methods
    -------
    fields(self, x, y, z, t)
        Calculates the electric and magnetic fields of the gaussian beam
        at every given position and time.

    __add__(self, other)
        Allows you to add two beams, the resulting beam object will calculate
        the fields for the superposition of both beam objects.
    """

    def __init__(
        self,
        focal_point=(0, 0),
        focal_size=(500e-6 / 2.3548, 500e-6 / 2.3548),
        envelope_offset=0,
        cep=0,
        wavelength=10e-6,
        M2=1.0,
        energy=30 * 1e-6,
        duration=300e-15,
        polarization=(np.sqrt(2), 1j * np.sqrt(2)),
    ):
        """
        Parameters
        ----------
        focal_point : tuple of scalar
            Longitudinal coordinates of hor. and ver. focal point.
        focal_size : tuple of scalar
            Focal sizes (standard deviaton) in the transverse plane.
        envelope_offset : scalar
            Time offset of the envelope. An offset of zero means the
            envelope’s maximum is at the origin.
        cep : scalar
            Carrier-envelope phase. A phase of zero means the electric
            field’s maximum is at the maximum of the envelope.
        wavelength : scalar
            Beam wavelength.
        M2 : scalar
            Beam propagation ratio. One means ideal Gaussian beam.
        energy : scalar
            Pulse energy in Joules.
        duration : scalar
            FWHM pulse duration in seconds.
        polarization : tuple of scalar
            Normalized Jones vector, defaults to right-hand circular polarization.
        """
        assert len(focal_point) == 2
        assert len(focal_size) == 2
        assert len(polarization) == 2

        self.focal_point = focal_point
        self.cep = cep
        # FWHM -> sigma for Gaussian distribution
        self.sigma_t = duration / (2 * np.sqrt(2 * np.log(2)))
        self.w0_x = 2 * focal_size[0]
        self.w0_y = 2 * focal_size[1]
        self.k = 2 * np.pi / wavelength  # wavenumber in 1/m
        self.omega = 2 * np.pi * const.c / wavelength  # angular frequency in rad/s
        self.sigma_z = duration * const.c / 2.3548  # sigma width of pulse length in m
        # horizontal Rayleigh length in m
        self.zRx = np.pi * self.w0_x ** 2 / (M2 * wavelength)
        # vertical Rayleigh length in m
        self.zRy = np.pi * self.w0_y ** 2 / (M2 * wavelength)
        self.E0 = (
            2 ** -0.25
            * np.pi ** -0.75
            * np.sqrt(
                const.mu_0
                * const.c
                * energy
                / (focal_size[0] * focal_size[1] * self.sigma_t)
            )
        )
        self.envelope_offset = envelope_offset
        self.polarization = polarization
        self.other_beams_list = []

    def field(self, x, y, z, t):
        """Calculates the electric field of the gaussian beam
        at every given position and time.

        Parameters
        ----------
        x : array_like
            x coordinates.
        y : array_like
            y coordinates.
        z : array_like
            z coordinates.
        t : array_like or scalar
            Time. Either defined for each set of x,y,z or scalar.

        Returns
        -------
        E : (..., 3) array_like
            Electric field vectors.
        """
        E0, phase = self._E0_and_phase(x, y, z, t)
        E_field = np.zeros((phase.shape[0], 3))
        # Correct for phase shift introduced by Jones vector
        # => For t = 0, E _always_ points to +x
        phase -= np.angle(self.polarization[1]) - np.pi / 2

        E_field[:, (0, 1)] = E0[:, None] * np.real(self.polarization * np.exp(1j * phase[:, None]))
        for otherbeam in self.other_beams_list:
            E = otherbeam.field(x, y, z, t)
            E_field += E
        return E_field

    def vector_potential(self, x, y, z, t):
        E0, phase = self._E0_and_phase(x, y, z, t)
        phase -= np.pi / 2

        E_field = np.zeros((phase.shape[0], 3))
        E_field[:, (0, 1)] = E0[:, None] * np.real(self.polarization * np.exp(1j * phase[:, None]))

        A = (- E_field / self.omega).T

        for otherbeam in self.other_beams_list:
            A += otherbeam.vector_potential(x, y, z, t)
        return A

    def vector_potential_Arne(self, x, y, z, t):
        dt = 1e-15
        ranges = t + np.arange(0, 2e-12, dt)[:, None]
        E = self.field(x, y, z, ranges)
        A = -np.trapz(E, dx=dt, axis=1)
        return A

    def _E0_and_phase(self, x, y, z, t):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        assert x.shape == y.shape == z.shape
        t = np.asarray(t)
        #assert t.shape == () or t.shape == x.shape

        # Distance of electron to focus (mod1_center)
        Zdif_x = z - self.focal_point[0]
        Zdif_y = z - self.focal_point[1]

        # horizontal beam sizes w_{x,y} at position z in m
        w_x = self.w0_x * np.sqrt(1 + Zdif_x ** 2 / (self.zRx ** 2))
        w_y = self.w0_y * np.sqrt(1 + Zdif_y ** 2 / (self.zRy ** 2))

        # Position of the laser pulse center
        Zlas = const.c * (t + self.envelope_offset)
        R_x = Zdif_x + (self.zRx ** 2 / Zdif_x)
        R_y = Zdif_y + (self.zRy ** 2 / Zdif_y)

        central_E_field = self.E0 * self.w0_x / w_x
        offaxis_pulsed_factor = np.exp(
            -((y / w_y) ** 2) - (x / w_x) ** 2 - ((z - Zlas) / (2 * self.sigma_z)) ** 2
        )
        phase = (
            self.k * (z - self.envelope_offset * const.c)
            - self.omega * t
            + self.k / 2 * x ** 2 / R_x
            + self.k / 2 * y ** 2 / R_y
            - 0.5 * np.arctan(Zdif_x / self.zRx)
            - 0.5 * np.arctan(Zdif_y / self.zRy)
            + self.cep
        )

        return central_E_field * offaxis_pulsed_factor, phase

    def __iadd__(self, other):
        """
        Allows you to add another beam to this one, from then on the fields method will
        calculate the superposition of both. Cascading works.

        Parameters
        ----------
        other : SimpleGaussianBeam
            Other beam object for superposition

        Returns
        -------
        newbeam : SimpleGaussianBeam
            New beam object that provides the superposition of the passed beams.
        """
        self.other_beams_list.append(other)
        return self

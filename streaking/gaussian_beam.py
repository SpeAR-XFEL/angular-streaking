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
        focal_size=(200e-6, 200e-6),
        envelope_offset=0,
        cep=0,
        wavelength=800e-9,
        M2=1.0,
        energy=1e-3,
        duration=50e-15,
        polarization=(np.sqrt(2), 1j * np.sqrt(2)),
        origin=(0, 0, 0),
        rotation=None,
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
            beam quality factor. One means ideal Gaussian beam.
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
        self.origin = origin
        self.rotation = rotation
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
        E_field = self._E_this_beam(x, y, z, t)

        for otherbeam in self.other_beams_list:
            E = otherbeam.field(x, y, z, t)
            E_field += E
        return E_field

    def vector_potential(self, x, y, z, t):
        E_field = self._E_this_beam(x, y, z, t, phase_offset=-np.pi / 2)
        A = -E_field / self.omega

        for otherbeam in self.other_beams_list:
            A += otherbeam.vector_potential(x, y, z, t)
        return A

    def _E_this_beam(self, x, y, z, t, phase_offset=0):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        assert x.shape == y.shape == z.shape
        t = np.asarray(t)
        assert t.shape == () or t.shape == x.shape

        x -= self.origin[0]
        y -= self.origin[1]
        z -= self.origin[2]

        if self.rotation is not None:
            x, y, z = self.rotation.apply(np.array((x, y, z)).T).T

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
            + phase_offset
        )

        #print(np.arctan(Zdif_x / self.zRx).min(), np.arctan(Zdif_x / self.zRx).max())
        #print(self.zRx)

        E0 = central_E_field * offaxis_pulsed_factor

        E_field = np.zeros((phase.shape[0], 3))
        # Correct for phase shift introduced by Jones vector
        # => For t = 0, E _always_ points to +x
        phase -= np.angle(self.polarization[1]) - np.pi / 2

        E_field[:, (0, 1)] = E0[:, None] * np.real(
            self.polarization * np.exp(1j * phase[:, None])
        )

        if self.rotation is not None:
            E_field = self.rotation.apply(E_field)

        return E_field

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


class RoundGaussianBeam:
    """Calculates electric and magnetic field of a simple round
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
        focal_point=0,
        focal_size=200e-6,
        envelope_offset=0,
        cep=0,
        wavelength=800e-9,
        M2=1.0,
        energy=1e-3,
        duration=50e-15,
        polarization=0,
        transverse_offset=(0,0)
    ):
        """
        Parameters
        ----------
        focal_point : scalar
            Longitudinal coordinate of focal point.
        focal_size : scalar
            Focal size (standard deviaton) in the transverse plane.
        envelope_offset : scalar
            Time offset of the envelope. An offset of zero means the
            envelope’s maximum is at the origin.
        cep : scalar
            Carrier-envelope phase. A phase of zero means the electric
            field’s maximum is at the maximum of the envelope.
        wavelength : scalar
            Beam wavelength.
        M2 : scalar
            beam quality factor. One means ideal Gaussian beam.
        energy : scalar
            Pulse energy in Joules.
        duration : scalar
            FWHM pulse duration in seconds.
        polarization : float
            angle of linear polarization axis, 0 = horizontal polarization
        transverse_offset : float
            tuple of horizontal and vertical parallel offset to the beam axis
        """

        self.focal_point = focal_point
        self.cep = cep
        self.wavelength = wavelength
        self.energy = energy
        # FWHM -> sigma for Gaussian distribution
        self.sigma_t = duration / 2.3548
        self.w0 = 2 * focal_size
        self.k = 2 * np.pi / wavelength  # wavenumber in 1/m
        self.omega = 2 * np.pi * const.c / wavelength  # angular frequency in rad/s
        self.sigma_z = self.sigma_t * const.c # sigma width of pulse length in m
        # horizontal Rayleigh length in m
        self.zR = np.pi * self.w0 ** 2 / (M2 * wavelength)

        self.P0 = self.energy * const.c/self.sigma_z / np.sqrt(2*np.pi)

        self.envelope_offset = envelope_offset
        self.polarization = polarization
        self.transverse_offset = transverse_offset
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
        E0, phase = self._E0_and_phase(x-self.transverse_offset[0], y-self.transverse_offset[1], z, t)

        # TODO: Implement Stokes vector.
        polarization_vec = np.array(
            (np.cos(self.polarization), np.sin(self.polarization), 0)
            )
        

        E_field = (E0 * np.cos(phase) * polarization_vec[:,None]).T

        for otherbeam in self.other_beams_list:
            E = otherbeam.field(x, y, z, t)
            E_field += E
        return E_field

    def _E0_and_phase(self, x, y, z, t):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        assert x.shape == y.shape == z.shape
        t = np.asarray(t)
        #assert t.shape == () or t.shape == x.shape

        # Distance of electron to focus (mod1_center)
        Zdif = z - self.focal_point

        # horizontal beam sizes w_{x,y} at position z in m
        w = self.w0 * np.sqrt(1 + Zdif ** 2 / (self.zR ** 2))


        # Position of the laser pulse center
        Zlas = const.c * (t + self.envelope_offset)
        P = self.P0*np.exp(-1/2 * ((z-Zlas)/self.sigma_z)**2)
        
        R = Zdif + (self.zR ** 2 / Zdif)
        gouy = np.arctan(Zdif / self.zR)
        
        factor=np.sqrt(4 * P / (const.epsilon_0 * const.c * np.pi)) / w
        exponent = -(x**2 + y**2)/w**2 + 1j * ( self.k * (z - self.envelope_offset * const.c) - 
                                               self.omega * t - gouy + self.k * (x**2 + y**2)/(2*R))

        
        
        E_amplitude = factor * np.exp(np.real(exponent))
        E_phase = np.imag(exponent)

        return E_amplitude, E_phase

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


class RoundGaussianBeamCircular:
    """Calculates electric and magnetic field of a simple round
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
        focal_point=0,
        focal_size=200e-6,
        envelope_offset=0,
        cep=0,
        wavelength=800e-9,
        M2=1.0,
        energy=1e-3,
        duration=50e-15,
        polarization=(np.sqrt(2), 1j * np.sqrt(2)),
        transverse_offset=(0,0)
    ):
        """
        Parameters
        ----------
        focal_point : scalar
            Longitudinal coordinate of focal point.
        focal_size : scalar
            Focal size (standard deviaton) in the transverse plane.
        envelope_offset : scalar
            Time offset of the envelope. An offset of zero means the
            envelope’s maximum is at the origin.
        cep : scalar
            Carrier-envelope phase. A phase of zero means the electric
            field’s maximum is at the maximum of the envelope.
        wavelength : scalar
            Beam wavelength.
        M2 : scalar
            beam quality factor. One means ideal Gaussian beam.
        energy : scalar
            Pulse energy in Joules.
        duration : scalar
            FWHM pulse duration in seconds.
        polarization : float
            angle of linear polarization axis, 0 = horizontal polarization
        transverse_offset : float
            tuple of horizontal and vertical parallel offset to the beam axis
        """

        self.focal_point = focal_point
        self.cep = cep
        self.wavelength = wavelength
        self.energy = energy
        # FWHM -> sigma for Gaussian distribution
        self.sigma_t = duration / 2.3548
        self.w0 = 2 * focal_size
        self.k = 2 * np.pi / wavelength  # wavenumber in 1/m
        self.omega = 2 * np.pi * const.c / wavelength  # angular frequency in rad/s
        self.sigma_z = self.sigma_t * const.c # sigma width of pulse length in m
        # horizontal Rayleigh length in m
        self.zR = np.pi * self.w0 ** 2 / (M2 * wavelength)

        self.P0 = self.energy * const.c/self.sigma_z / np.sqrt(2*np.pi)

        self.envelope_offset = envelope_offset
        self.polarization = polarization
        self.transverse_offset = transverse_offset
        self.other_beams_list = []

    def field(self, x, y, z, t, phase_offset=0):
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
        E0, phase = self._E0_and_phase(x-self.transverse_offset[0], y-self.transverse_offset[1], z, t, phase_offset)

        # TODO: Implement Stokes vector.
        
        E_field = np.zeros((phase.shape[0], 3))
        # Correct for phase shift introduced by Jones vector
        # => For t = 0, E _always_ points to +x
        phase -= np.angle(self.polarization[1]) - np.pi / 2

        E_field[:, (0, 1)] = E0[:, None] * np.real(
            self.polarization * np.exp(1j * phase[:, None])
        )        

        for otherbeam in self.other_beams_list:
            E = otherbeam.field(x, y, z, t)
            E_field += E
        return E_field

    def _E0_and_phase(self, x, y, z, t, phase_offset):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        assert x.shape == y.shape == z.shape
        t = np.asarray(t)
        #assert t.shape == () or t.shape == x.shape

        # Distance of electron to focus (mod1_center)
        Zdif = z - self.focal_point

        # horizontal beam sizes w_{x,y} at position z in m
        w = self.w0 * np.sqrt(1 + Zdif ** 2 / (self.zR ** 2))


        # Position of the laser pulse center
        Zlas = const.c * (t + self.envelope_offset)
        P = self.P0*np.exp(-1/2 * ((z-Zlas)/self.sigma_z)**2)
        
        R = Zdif + (self.zR ** 2 / Zdif)
        gouy = np.arctan(Zdif / self.zR)
        
        factor=np.sqrt(4 * P / (const.epsilon_0 * const.c * np.pi)) / w
        exponent = -(x**2 + y**2)/w**2 + 1j * ( self.k * (z - self.envelope_offset * const.c) - 
                                               self.omega * t - gouy + self.k * (x**2 + y**2)/(2*R))

        
        
        E_amplitude = factor * np.exp(np.real(exponent))
        E_phase = np.imag(exponent) + phase_offset

        return E_amplitude, E_phase

    def vector_potential(self, x, y, z, t):
        E_field = self.field(x, y, z, t, phase_offset=-np.pi / 2)
        A = -E_field / self.omega
    
        for otherbeam in self.other_beams_list:
            A += otherbeam.vector_potential(x, y, z, t)
        return A
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


class AstigmaticGaussianBeam:
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
        center_point = 0, 
        astigmatism = 0, 
        theta = 0, 
        focal_size=(200e-6, 200e-6),
        envelope_offset=0,
        cep=0,
        wavelength=800e-9,
        M2=(1.0 , 1.0),
        energy=1e-3,
        duration=50e-15,
        polarization=0,
        offset=(0,0)
    ):
        """
        Parameters
        ----------
        center_point : scalar
            Longitudinal coordinate of point centerered between the focal points.
        astigmatism : scalar
            distance between the focal points
        theta : scalar
            angle between lab and principal coordinate system. 
            Zero means, the first focus is the horizontal focus and the second one is vertical.
        focal_size : tuple of scalar
            Focal sizes (standard deviaton).
        envelope_offset : scalar
            Time offset of the envelope. For t=0, the pulse maximum is at this value.
        wavelength : scalar
            Beam wavelength.
        M2 : tuple of scalar
            Hor. and ver. beam quality factor. One means ideal Gaussian beam.
        energy : scalar
            Pulse energy in Joules.
        duration : scalar
            FWHM pulse duration in seconds.
        polarization : float
            angle of linear polarization axis, 0 = horizontal polarization
        offset : float
            tuple of horizontal and vertical parallel offset to the beam axis
        """
        assert len(focal_size) == 2

        self.center_point = center_point
        self.astigmatism = astigmatism
        self.theta = theta
        self.cep = cep
        self.wavelength = wavelength
        self.energy = energy
        # FWHM -> sigma for Gaussian distribution
        self.sigma_t = duration / 2.3548
        self.w0_1 = 2 * focal_size[0]
        self.w0_2 = 2 * focal_size[1]
        self.k = 2 * np.pi / wavelength  # wavenumber in 1/m
        self.omega = 2 * np.pi * const.c / wavelength  # angular frequency in rad/s
        self.sigma_z = self.sigma_t * const.c  # sigma width of pulse length in m
        # horizontal Rayleigh length in m
        self.zR1 = np.pi * self.w0_1 ** 2 / (M2[0] * wavelength)
        # vertical Rayleigh length in m
        self.zR2 = np.pi * self.w0_2 ** 2 / (M2[1] * wavelength)
        self.P0 = self.energy * const.c/self.sigma_z / np.sqrt(2*np.pi)

        self.envelope_offset = envelope_offset
        self.polarization = polarization
        self.offset = offset
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
        E0, phase = self._E0_and_phase(x-self.offset[0], y-self.offset[1], z, t)

        # TODO: Implement Stokes vector.
        polarization_vec = np.array(
            (np.cos(self.polarization), np.sin(self.polarization), 0)
            )
        

        E_field = (E0 * np.cos(phase) * polarization_vec[:,None]).T

        for otherbeam in self.other_beams_list:
            E = otherbeam.field(x, y, z, t)
            E_field += E
        return E_field

    def _E0_and_phase(self, x, y, z, t):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        assert x.shape == y.shape == z.shape
        t = np.asarray(t)
        #assert t.shape == () or t.shape == x.shape

        # Distance of electron to focus (mod1_center)
        Zdif_1 = z - (self.center_point-self.astigmatism/2)
        Zdif_2 = z - (self.center_point+self.astigmatism/2)

        # Position of the laser pulse center
        Zlas = const.c * (t + self.envelope_offset)

        q1 = Zdif_1 + 1j * self.zR1
        q2 = Zdif_2 + 1j * self.zR2
        
        P = self.P0*np.exp(-1/2 * ((z-Zlas)/self.sigma_z)**2)
        
        gouy = - (np.arctan(Zdif_1 / self.zR1) + np.arctan(Zdif_2 / self.zR2)) / 2
        
        factor = np.sqrt(2 * P * self.k * np.sqrt(self.zR1*self.zR2) / (const.epsilon_0 * const.c * np.pi * np.abs(q1*q2)))
        exponent = 1j * (gouy + self.cep + self.k * (z - self.envelope_offset * const.c) - self.omega * t +
                               self.k / 2 * (x**2 * (np.cos(self.theta)**2 / q1 + np.sin(self.theta)**2 / q2) + 
                                y**2 * (np.sin(self.theta)**2 / q1 + np.cos(self.theta)**2 / q2) + 
                                np.sin(2*self.theta) * (1/q1 + 1/q2)*x*y))
        
        E_amplitude = factor * np.exp(np.real(exponent))
        E_phase = np.imag(exponent)
                           

        return E_amplitude, E_phase

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
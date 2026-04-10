from __future__ import annotations

import numpy as np
import miepython
from .base import ScatteringBackend, ParticleSize, SphereSize


class MiePythonBackend(ScatteringBackend):
    name = "miepython"

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, mu: np.ndarray, size: ParticleSize) -> np.ndarray:
        """
        Compute unpolarized scattering intensity for Mie theory.
        
        Args:
            m: Complex refractive index
            wavelength_nm: Wavelength in nanometers
            mu: Cosines of scattering angles
            size: ParticleSize (must be SphereSize for Mie)
        
        Returns:
            Unpolarized intensity values
        """
        if not isinstance(size, SphereSize):
            raise TypeError(f"MiePythonBackend only accepts SphereSize, got {type(size).__name__}")
        
        # Calculate size parameter x
        wavelength_um = wavelength_nm / 1000.0
        x = 2.0 * np.pi * size.r_um / max(wavelength_um, 1e-12)
        return miepython.i_unpolarized(m, x, mu)
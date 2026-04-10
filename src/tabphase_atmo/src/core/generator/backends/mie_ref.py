from __future__ import annotations

import numpy as np
from .base import ParticleSize, SphereSize
from ..mie_backend import mie_unpolarized_intensity


class MieReferenceBackend:
    """
    Intentional copy of the miepython backend with a different name.

    Purpose:
      - validate wrapper/backend plumbing and cache separation
      - provide a stable "reference" label later when we add hybrid backends
    """
    name = "mie_ref"

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, mu: np.ndarray, size: ParticleSize) -> np.ndarray:
        """
        Compute unpolarized scattering intensity using Mie reference implementation.
        
        Args:
            m: Complex refractive index
            wavelength_nm: Wavelength in nanometers
            mu: Cosines of scattering angles
            size: ParticleSize (must be SphereSize for Mie)
        
        Returns:
            Unpolarized intensity values
        """
        if not isinstance(size, SphereSize):
            raise TypeError(f"MieReferenceBackend only accepts SphereSize, got {type(size).__name__}")
        
        # Calculate size parameter x
        wavelength_um = wavelength_nm / 1000.0
        x = 2.0 * np.pi * size.r_um / max(wavelength_um, 1e-12)
        return mie_unpolarized_intensity(m, x, mu)
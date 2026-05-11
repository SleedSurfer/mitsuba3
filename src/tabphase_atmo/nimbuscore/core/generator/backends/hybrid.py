from __future__ import annotations

import numpy as np
from .base import ScatteringBackend, ParticleSize, SphereSize


class HybridBackend:
    """
    Hard-switch hybrid: evaluates entirely on Mie or entirely on GO
    based strictly on droplet radius to prevent spectral tearing.
    """
    name = "hybrid"

    def __init__(
            self,
            low_backend: ScatteringBackend,
            high_backend: ScatteringBackend,
            radius_threshold_um: float = 500.0,
    ):
        self.low_backend = low_backend
        self.high_backend = high_backend
        self.radius_threshold_um = float(radius_threshold_um)

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, mu: np.ndarray, size: ParticleSize) -> np.ndarray:
        """
        Routes the calculation to the appropriate backend based on physical radius.
        """
        if not isinstance(size, SphereSize):
            raise TypeError(f"HybridBackend only works with SphereSize, got {type(size).__name__}")

        if size.r_um < self.radius_threshold_um:
            return self.low_backend.intensity_unpolarized(m, wavelength_nm, mu, size)
        else:
            return self.high_backend.intensity_unpolarized(m, wavelength_nm, mu, size)

    def get_geometric_threshold(self, wavelength_nm: float) -> float:
        """
        Returns the fixed radius where Geometric Optics takes over.
        Wavelength is ignored to maintain spectral consistency.
        """
        return self.radius_threshold_um
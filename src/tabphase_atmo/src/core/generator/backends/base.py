from __future__ import annotations

from typing import Protocol
import numpy as np


class ScatteringBackend(Protocol):
    """
    Backends compute unpolarized single-sphere scattering intensity I(mu)
    for a given refractive index m and size parameter x.
    """

    name: str

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, radius_um: float, mu: np.ndarray) -> np.ndarray:
        """
        Calculates the unpolarized scattering phase function.
        m: Complex index of refraction
        wavelength_nm: True wavelength in nanometers
        radius_um: True particle radius in micrometers
        mu: Cosine of scattering angles
        """
        ...

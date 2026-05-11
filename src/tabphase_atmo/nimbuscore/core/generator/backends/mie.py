from __future__ import annotations

import numpy as np
import miepython
from .base import ScatteringBackend, ParticleSize, SphereSize


class MiePythonBackend(ScatteringBackend):
    name = "miepython"

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, theta: np.ndarray,
                              size: ParticleSize) -> np.ndarray:
        if not isinstance(size, SphereSize):
            raise TypeError(f"MiePythonBackend only accepts SphereSize, got {type(size).__name__}")

        wavelength_um = wavelength_nm / 1000.0
        x = 2.0 * np.pi * size.r_um / max(wavelength_um, 1e-12)

        mu_quarantined = np.cos(theta)
        return miepython.i_unpolarized(m, x, mu_quarantined)
from __future__ import annotations

import numpy as np
import miepython

from ..mie_backend import mie_unpolarized_intensity


class MiePythonBackend:
    name = "miepython"

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, radius_um: float, mu: np.ndarray) -> np.ndarray:
        # Calculate x because Mie analytical formulas actually need it
        wavelength_um = wavelength_nm / 1000.0
        x = 2.0 * np.pi * radius_um / max(wavelength_um, 1e-12)
        return miepython.i_unpolarized(m, x, mu)
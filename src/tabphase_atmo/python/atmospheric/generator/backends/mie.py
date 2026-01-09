from __future__ import annotations

import numpy as np
import miepython

from ..mie_backend import mie_unpolarized_intensity


class MiePythonBackend:
    name = "miepython"

    def intensity_unpolarized(self, m: complex, x: float, mu: np.ndarray) -> np.ndarray:
        return miepython.i_unpolarized(m, x, mu)
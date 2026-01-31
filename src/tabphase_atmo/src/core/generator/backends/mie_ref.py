from __future__ import annotations

import numpy as np
from ..mie_backend import mie_unpolarized_intensity


class MieReferenceBackend:
    """
    Intentional copy of the miepython backend with a different name.

    Purpose:
      - validate wrapper/backend plumbing and cache separation
      - provide a stable "reference" label later when we add hybrid backends
    """
    name = "mie_ref"

    def intensity_unpolarized(self, m: complex, x: float, mu: np.ndarray) -> np.ndarray:
        return mie_unpolarized_intensity(m, x, mu)
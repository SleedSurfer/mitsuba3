from __future__ import annotations

import numpy as np
from .base import ScatteringBackend


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


class HybridBackend:
    """
    True hybrid: switches/blends per-call based on x.
    Called once per (λ, r) inside the LUT generator.
    """
    name = "hybrid"

    def __init__(
        self,
        low_backend: ScatteringBackend,
        high_backend: ScatteringBackend,
        x0: float = 800.0,
        x1: float = 1400.0,
    ):
        if x1 <= x0:
            raise ValueError("HybridBackend requires x1 > x0")
        self.low_backend = low_backend
        self.high_backend = high_backend
        self.x0 = float(x0)
        self.x1 = float(x1)

    def intensity_unpolarized(self, m: complex, x: float, mu: np.ndarray) -> np.ndarray:
        w = _smoothstep(self.x0, self.x1, x)

        if w <= 0.0:
            return self.low_backend.intensity_unpolarized(m, x, mu)
        if w >= 1.0:
            return self.high_backend.intensity_unpolarized(m, x, mu)

        lo = self.low_backend.intensity_unpolarized(m, x, mu)
        hi = self.high_backend.intensity_unpolarized(m, x, mu)
        return (1.0 - w) * lo + w * hi
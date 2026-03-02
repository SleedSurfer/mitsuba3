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

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, radius_um: float, mu: np.ndarray) -> np.ndarray:
        # Calculate x internally just for the blending weight
        wavelength_um = wavelength_nm / 1000.0
        x = 2.0 * np.pi * radius_um / max(wavelength_um, 1e-12)

        w = _smoothstep(self.x0, self.x1, x)

        if w <= 0.0:
            return self.low_backend.intensity_unpolarized(m, wavelength_nm, radius_um, mu)
        if w >= 1.0:
            return self.high_backend.intensity_unpolarized(m, wavelength_nm, radius_um, mu)

        lo = self.low_backend.intensity_unpolarized(m, wavelength_nm, radius_um, mu)
        hi = self.high_backend.intensity_unpolarized(m, wavelength_nm, radius_um, mu)
        return (1.0 - w) * lo + w * hi

    def get_geometric_threshold(self, wavelength_nm: float) -> float:
        """
        Returns the radius (in microns) where Geometric Optics becomes active.
        This is the radius where size parameter x >= x1 (full GO regime).

        Args:
            wavelength_nm: Wavelength in nanometers

        Returns:
            radius_um: Threshold radius in microns
        """
        wavelength_um = wavelength_nm / 1000.0
        # x = 2π * r / λ → r = x * λ / (2π)
        return self.x1 * wavelength_um / (2.0 * np.pi)
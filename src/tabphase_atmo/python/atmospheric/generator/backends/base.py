from __future__ import annotations

from typing import Protocol
import numpy as np


class ScatteringBackend(Protocol):
    """
    Backends compute unpolarized single-sphere scattering intensity I(mu)
    for a given refractive index m and size parameter x.
    """

    name: str

    def intensity_unpolarized(self, m: complex, x: float, mu: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        m : complex
            Relative refractive index.
        x : float
            Size parameter, x = 2*pi*r / lambda.
        mu : np.ndarray
            Cosine of scattering angle.

        Returns
        -------
        np.ndarray
            Unpolarized intensity aligned with mu.
        """
        ...
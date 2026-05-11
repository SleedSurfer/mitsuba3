"""
mie_backend.py
Mie scattering backend (single-sphere unpolarized intensity) used by the LUT generator.

This file exists to make the generator modular: other backends (GO/Airy/hybrid, etc.)
can later implement the same API.
"""
from __future__ import annotations

import numpy as np
import miepython


def mie_unpolarized_intensity(m: complex, x: float, mu: np.ndarray) -> np.ndarray:
    """
    Returns unpolarized scattering intensity I(mu) for a sphere.

    Parameters
    ----------
    m : complex
        Relative refractive index (particle / medium). For water droplets in air, ~1.33+0j.
    x : float
        Size parameter x = 2*pi*r/lambda (with r and lambda in same units).
    mu : np.ndarray
        Cosine of scattering angle, typically linspace(1, -1, N).

    Returns
    -------
    np.ndarray
        Intensity values aligned with mu (same shape).
    """
    return miepython.i_unpolarized(m, x, mu)
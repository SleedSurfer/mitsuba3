from __future__ import annotations

import numpy as np
from scipy.special import airy


def _fresnel_R_unpolarized(cos_i: np.ndarray, n: float) -> np.ndarray:
    """Unpolarized Fresnel reflectance for air->sphere (n>1)."""
    cos_i = np.clip(cos_i, 0.0, 1.0)
    sin2_i = np.maximum(0.0, 1.0 - cos_i * cos_i)

    sin2_t = sin2_i / (n * n)
    # air->water: no TIR, but keep robust
    tir = sin2_t > 1.0
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))

    rs = ((cos_i - n * cos_t) / (cos_i + n * cos_t + 1e-12)) ** 2
    rp = ((n * cos_i - cos_t) / (n * cos_i + cos_t + 1e-12)) ** 2
    R = 0.5 * (rs + rp)
    R[tir] = 1.0
    return R


def _snell_r(i: np.ndarray, n: float) -> np.ndarray:
    """Refraction angle r = asin(sin(i)/n)."""
    return np.arcsin(np.sin(i) / max(n, 1e-9))


def _deviation(theta_order: int, i: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    GO deviation/scattering angle θ_p(i) for order p:

      p=0 (external reflection): θ = π - 2i
      p>=1 (p-1 internal reflections): θ = pπ + 2i - 2(p+1)r
    """
    p = theta_order
    if p == 0:
        return np.pi - 2.0 * i
    return p * np.pi + 2.0 * i - 2.0 * (p + 1.0) * r


def _dtheta_di(p: int, i: np.ndarray, n: float, r: np.ndarray) -> np.ndarray:
    """
    Derivative dθ/di for GO mapping.
    r depends on i via Snell, with dr/di = cos(i)/(n*cos(r)).
    """
    if p == 0:
        return -2.0 * np.ones_like(i)

    cos_i = np.cos(i)
    cos_r = np.cos(r)
    dr_di = cos_i / (max(n, 1e-9) * (cos_r + 1e-12))
    return 2.0 - 2.0 * (p + 1.0) * dr_di


def _invert_monotone(theta_of_i: np.ndarray, i_grid: np.ndarray, theta_query: np.ndarray) -> np.ndarray:
    """
    Invert theta(i) approximately via interpolation assuming monotone arrays.
    Returns i(theta_query). Out-of-range queries are clipped.
    """
    # theta_of_i must be increasing for np.interp; enforce by possibly reversing
    if theta_of_i[0] > theta_of_i[-1]:
        theta_of_i = theta_of_i[::-1]
        i_grid = i_grid[::-1]

    theta_min = theta_of_i[0]
    theta_max = theta_of_i[-1]
    tq = np.clip(theta_query, theta_min, theta_max)
    return np.interp(tq, theta_of_i, i_grid)


def _forward_diffraction(mu: np.ndarray, x: float, strength: float) -> np.ndarray:
    theta = np.arccos(np.clip(mu, -1.0, 1.0))
    sigma = max(1e-4, 2.0 / max(float(x), 1.0))  # radians
    return strength * np.exp(-(theta * theta) / (2.0 * sigma * sigma))


def _primary_rainbow_incidence(n: float) -> float:
    """
    Incidence angle i_r where primary (p=2) deviation is stationary:
      cos^2(i_r) = (n^2 - 1)/3
    """
    n2 = n * n
    cos2i = (n2 - 1.0) / 3.0
    cos2i = float(np.clip(cos2i, 0.0, 1.0))
    return float(np.arccos(np.sqrt(cos2i)))


class GOAiryBackend:
    """
    Large-x backend: GO ray orders + Airy uniform approximation near primary rainbow.

    This is intended to be used via HybridBackend, not as a perfect Mie replacement.
    """
    name = "go_airy"

    def __init__(
        self,
        i_samples: int = 4096,
        p_orders: tuple[int, ...] = (0, 1, 2),
        include_diffraction: bool = True,
        diffraction_strength: float = 1.0,

        enable_airy: bool = True,
        airy_strength: float = 1.0,
        airy_width_scale: float = 0.02,   # radians * x^(2/3) (tunable)
        airy_window_deg: float = 10.0,
    ):
        self.i_samples = i_samples
        self.p_orders = p_orders
        self.include_diffraction = include_diffraction
        self.diffraction_strength = diffraction_strength

        self.enable_airy = enable_airy
        self.airy_strength = airy_strength
        self.airy_width_scale = airy_width_scale
        self.airy_window_deg = airy_window_deg

        # Fixed incidence grid
        self._i_grid = np.linspace(0.0, 0.5 * np.pi - 1e-6, i_samples)

    def intensity_unpolarized(self, m: complex, x: float, mu: np.ndarray) -> np.ndarray:
        n = float(np.real(m))
        theta = np.arccos(np.clip(mu, -1.0, 1.0))

        i_grid = self._i_grid
        r_grid = _snell_r(i_grid, n)
        cos_i_grid = np.cos(i_grid)

        R_grid = _fresnel_R_unpolarized(cos_i_grid, n)
        T_grid = np.maximum(0.0, 1.0 - R_grid)

        I_total = np.zeros_like(theta, dtype=np.float64)

        # --- Sum GO orders ---
        for p in self.p_orders:
            th_map = _deviation(p, i_grid, r_grid)
            dth = _dtheta_di(p, i_grid, n, r_grid)
            J = np.maximum(np.abs(dth), 1e-6)

            # energy-like weight for the order
            # p=0: single reflection
            # p>=1: transmission in, (internal reflections)^(p-1), transmission out
            if p == 0:
                W = R_grid
            else:
                W = T_grid * (R_grid ** (p - 1)) * T_grid

            # Invert theta(i) approximately to i(theta)
            # For p=2, theta(i) is not globally monotone due to the rainbow stationary point;
            # We will treat it with Airy in a window around the rainbow and use a monotone branch otherwise.
            if p == 2 and self.enable_airy:
                # Rainbow location
                i_r = _primary_rainbow_incidence(n)
                r_r = float(_snell_r(np.array([i_r]), n)[0])
                theta_r = float(_deviation(2, np.array([i_r]), np.array([r_r]))[0])

                window = np.deg2rad(self.airy_window_deg)
                near = np.abs(theta - theta_r) <= window

                # --- Airy in the rainbow window ---
                if np.any(near):
                    # z scaling ~ (theta - theta_r) / (c * x^{-2/3})
                    airy_width = max(self.airy_width_scale * (max(float(x), 1.0) ** (-2.0 / 3.0)), 1e-6)
                    z = (theta[near] - theta_r) / airy_width
                    Ai = airy(z)[0]
                    airy_I = (Ai * Ai)

                    # Scale by a representative GO weight at i_r
                    # (Not exact amplitude; generator normalization fixes energy.)
                    # Evaluate W/J at closest grid index to i_r:
                    ir_idx = int(np.clip(round((i_r / (0.5 * np.pi)) * (self.i_samples - 1)), 0, self.i_samples - 1))
                    scale = float(W[ir_idx] / J[ir_idx])
                    I_total[near] += self.airy_strength * scale * airy_I

                # --- Outside window: use a monotone branch (post-rainbow) ---
                far = ~near
                if np.any(far):
                    # Choose a monotone segment. Using i in [i_r, pi/2) gives post-rainbow branch.
                    mask = i_grid >= i_r
                    th_seg = th_map[mask]
                    i_seg = i_grid[mask]
                    W_seg = W[mask]
                    J_seg = J[mask]

                    i_eval = _invert_monotone(th_seg, i_seg, theta[far])
                    # Interpolate W/J over i eval
                    val = np.interp(i_eval, i_seg, W_seg / J_seg)
                    I_total[far] += val

            else:
                # For p=0 and p=1 monotonicity is OK over i in [0, pi/2)
                i_eval = _invert_monotone(th_map, i_grid, theta)
                val = np.interp(i_eval, i_grid, W / J)
                I_total += val

        # --- Forward diffraction (separate lobe) ---
        if self.include_diffraction:
            I_total += _forward_diffraction(mu, x, self.diffraction_strength)

        # Numerical floor for safety
        return np.maximum(I_total, 0.0).astype(np.float64)
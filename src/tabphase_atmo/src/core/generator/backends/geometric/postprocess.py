import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import j1

# Table II from Sadeghi et al. (2012)
# Standard Deviation of the Gaussian Filter Diffraction Approximation
_RADIUS_MM = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_SIGMA_DEG = np.array([0.70, 0.45, 0.30, 0.25, 0.22, 0.20, 0.18, 0.17, 0.16, 0.15])


import numpy as np
from scipy.ndimage import gaussian_filter1d

import numpy as np
from scipy.ndimage import gaussian_filter1d


def apply_diffraction_smoothing(
        theta_rad: np.ndarray,
        intensity: np.ndarray,
        radius_mm: float
) -> np.ndarray:
    """
    Applies localized Gaussian blurs to the primary and secondary rainbow caustics,
    leaving forward scattering and supernumerary fringes pristine.
    """
    radius_clamped = np.clip(radius_mm, _RADIUS_MM[0], _RADIUS_MM[-1])

    # Primary sigma
    base_sigma_deg = np.interp(radius_clamped, _RADIUS_MM, _SIGMA_DEG)
    sigma_rad_pri = np.radians(base_sigma_deg)

    # Secondary sigma (Sadeghi doubles it)
    sigma_rad_sec = np.radians(base_sigma_deg * 2.0)

    num_bins = len(theta_rad)
    if num_bins < 2:
        return intensity

    dtheta = abs(theta_rad[1] - theta_rad[0])
    sigma_bins_pri = sigma_rad_pri / dtheta
    sigma_bins_sec = sigma_rad_sec / dtheta

    # 1. Generate the fully blurred signals for both primary and secondary
    smoothed_pri = gaussian_filter1d(intensity, sigma=sigma_bins_pri, mode='nearest')
    smoothed_sec = gaussian_filter1d(intensity, sigma=sigma_bins_sec, mode='nearest')

    # 2. Box in the search area to stop np.argmax from finding the 0° sun nuke.
    # Primary rainbow is typically around 137° - 142°
    mask_pri = (theta_rad >= np.radians(135.0)) & (theta_rad <= np.radians(145.0))
    # Secondary rainbow is typically around 125° - 130°
    mask_sec = (theta_rad >= np.radians(120.0)) & (theta_rad <= np.radians(132.0))

    # Find the exact geometric cliffs within those bounds
    peak_idx_pri = np.argmax(np.where(mask_pri, intensity, 0.0))
    peak_idx_sec = np.argmax(np.where(mask_sec, intensity, 0.0))

    # 3. Build soft Gaussian windows centered exactly on the singularities.
    window_pri = np.exp(-0.5 * ((np.arange(num_bins) - peak_idx_pri) / (sigma_bins_pri * 3.0)) ** 2)
    window_sec = np.exp(-0.5 * ((np.arange(num_bins) - peak_idx_sec) / (sigma_bins_sec * 3.0)) ** 2)

    # 4. Composite: blend the primary blur, the secondary blur, and keep the rest raw.
    final_intensity = (intensity * (1.0 - window_pri - window_sec)) + \
                      (smoothed_pri * window_pri) + \
                      (smoothed_sec * window_sec)

    return np.maximum(final_intensity, 0.0)


def compute_fraunhofer_diffraction(theta_rad: np.ndarray, x: float) -> np.ndarray:
    """
    Computes the forward Fraunhofer diffraction lobe.
    Includes a strict Gaussian window to kill the unphysical Bessel ringing
    in the tail that otherwise destroys the geometric raytraced data.
    """
    diffraction = np.zeros_like(theta_rad)
    sin_theta = np.sin(theta_rad)

    mask = theta_rad < 1e-7
    diffraction[mask] = (x ** 4) / 4.0

    valid = ~mask
    u_val = x * sin_theta[valid]

    obliquity = ((1.0 + np.cos(theta_rad[valid])) / 2.0) ** 2

    # Raw Bessel calculation
    raw_diffraction = ((x * j1(u_val)) / sin_theta[valid]) ** 2 * obliquity

    # THE FIX: Aggressive Gaussian Windowing
    # 0.05 radians is about 2.8 degrees. This smoothly but brutally
    # executes the Fraunhofer tail before it touches the geometric data.
    sigma_rad = 10.0 / x
    window = np.exp(-0.5 * (theta_rad[valid] / sigma_rad) ** 2)

    diffraction[valid] = raw_diffraction * window

    return diffraction
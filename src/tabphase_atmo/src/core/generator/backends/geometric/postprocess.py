import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import j1

# Table II from Sadeghi et al. (2012)
# Standard Deviation of the Gaussian Filter Diffraction Approximation
_RADIUS_MM = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_SIGMA_DEG = np.array([0.70, 0.45, 0.30, 0.25, 0.22, 0.20, 0.18, 0.17, 0.16, 0.15])


def apply_diffraction_smoothing(
        theta_rad: np.ndarray,
        intensity: np.ndarray,
        radius_mm: float,
        is_secondary: bool = False
) -> np.ndarray:
    """
    Applies a physically-based Gaussian blur to approximate wave diffraction,
    softening the infinite peaks predicted by pure geometric optics.
    """
    # 1. Interpolate the required sigma from the paper's empirical table
    # We clip the radius to stay within the bounds of the provided data
    radius_clamped = np.clip(radius_mm, _RADIUS_MM[0], _RADIUS_MM[-1])
    sigma_deg = np.interp(radius_clamped, _RADIUS_MM, _SIGMA_DEG)

    # The paper doubles the blur radius for the secondary rainbow (p=3)
    # to account for the extra internal reflection scattering.
    if is_secondary:
        sigma_deg *= 2.0

    sigma_rad = np.radians(sigma_deg)

    # 2. Convert the spatial standard deviation (radians) into array bins
    # We need to know how many radians each bin in our array represents
    num_bins = len(theta_rad)
    if num_bins < 2:
        return intensity  # Cannot filter a point

    # Assuming theta_rad is uniformly spaced (which it is from our linspace)
    dtheta = abs(theta_rad[1] - theta_rad[0])

    # Calculate sigma in terms of array indices
    sigma_bins = sigma_rad / dtheta

    # 3. Apply the 1D Gaussian convolution
    # mode='nearest' prevents the edges of the polar plot from dropping to zero
    smoothed_intensity = gaussian_filter1d(intensity, sigma=sigma_bins, mode='nearest')

    return smoothed_intensity


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
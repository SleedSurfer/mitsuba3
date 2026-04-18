import numpy as np
from scipy.special import j1

def get_airy_diffraction(theta_rad, radius_um, wavelength_nm):
    """
    Computes Fraunhofer diffraction forward peak for a specific effective radius.
    """
    wavelength_um = wavelength_nm / 1000.0
    size_param = (2.0 * np.pi * radius_um) / wavelength_um

    theta_safe = np.maximum(theta_rad, 1e-7)
    u = size_param * np.sin(theta_safe)

    airy_intensity = (2.0 * j1(u) / u) ** 2
    return airy_intensity * (size_param ** 2) / (4.0 * np.pi)


def _build_blur_matrix(theta_rad, variance):
    """
    Constructs a universal (N x N) Gaussian blur transformation matrix.
    Doing this once saves billions of redundant calculations.
    """
    N = len(theta_rad)
    d_theta = theta_rad[1] - theta_rad[0]
    theta_rbw = np.radians(138.0)

    spread_fwd = theta_rad * variance
    spread_rbw = (2.0 / 3.0) * (np.abs(theta_rad - theta_rbw) + 0.1) * variance
    blend = 0.5 * (1.0 + np.tanh((theta_rad - np.pi / 2) * 4.0))
    spread_rad = (1.0 - blend) * spread_fwd + blend * spread_rbw + 1e-5

    sigma_idx = np.maximum(0.5, spread_rad / d_theta)

    i_grid = np.arange(N)[:, None]
    j_grid = np.arange(N)[None, :]
    sigma_grid = sigma_idx[:, None]

    W = np.exp(-0.5 * ((j_grid - i_grid) / sigma_grid) ** 2)

    row_sums = np.sum(W, axis=1, keepdims=True)
    W /= row_sums

    return W


def apply_polydispersity_filter(phase_table, num_angles, variance):
    theta_rad = np.linspace(0, np.pi, num_angles)
    sin_theta = np.sin(theta_rad)

    print(f"[Filter] Constructing {num_angles}x{num_angles} Blur Matrix...")
    W = _build_blur_matrix(theta_rad, variance)
    original_shape = phase_table.shape
    flat_phase = phase_table.reshape(-1, num_angles)

    original_integrals = np.trapezoid(flat_phase * sin_theta, theta_rad, axis=1)

    log_phase = np.log(np.maximum(flat_phase, 1e-12))
    print(f"[Filter] Applying Tensor Convolution...")
    result_log = log_phase @ W.T

    smoothed_phase = np.exp(result_log)

    new_integrals = np.trapezoid(smoothed_phase * sin_theta, theta_rad, axis=1)
    safe_new = np.maximum(new_integrals, 1e-12)
    correction = original_integrals / safe_new

    smoothed_phase *= correction[:, None]

    return smoothed_phase.reshape(original_shape)
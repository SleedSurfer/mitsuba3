import numpy as np
from scipy.special import j1
import scipy.sparse as sp


def _build_blur_matrix(theta_rad, variance):
    """
    Constructs a dense (N x N) Gaussian blur transformation matrix.
    Truncates beyond 4-sigma to prevent mathematical ghosting.
    Returns a raw numpy array. No SciPy CSR overhead.
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

    # Dynamic 4-sigma cutoff: kill the useless tail weights
    W[np.abs(j_grid - i_grid) > 4.0 * sigma_grid] = 0.0

    row_sums = np.sum(W, axis=1, keepdims=True)

    # Safe divide to prevent NaN, return pure dense numpy array
    return np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums != 0)


def apply_polydispersity_filter(phase_table, num_angles, variance):
    theta_rad = np.linspace(0, np.pi, num_angles)

    # Grab the raw dense matrix
    W = _build_blur_matrix(theta_rad, variance)

    original_shape = phase_table.shape
    num_wavelengths = original_shape[0]

    smoothed_phase = np.zeros_like(phase_table)

    print(f"[Filter] Processing {num_wavelengths} wavelength chunks (Dense Log-Space)...")

    for w in range(num_wavelengths):
        # Extract 2D slice: (Azimuths, Angles)
        chunk = phase_table[w, :, :]

        log_chunk = np.log(np.maximum(chunk, 1e-12))

        # Apply the dense matrix convolution using standard numpy matmul
        result_log = (W @ log_chunk.T).T

        # Exponentiate and immediately assign.
        # Energy conservation is handled globally by normalize_macroscopic_phase.
        smoothed_phase[w, :, :] = np.exp(result_log)

    return smoothed_phase
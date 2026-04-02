import numpy as np
import struct

from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.hermite import hermgauss
import multiprocessing

from .lobe_analyzer import analyze_lut_for_mis, write_mis_metadata
from ..config import MieConfig
from ..materials import get_material_ior
from .backends.base import ScatteringBackend
from .backends.mie import MiePythonBackend


def _compute_lognormal_params(radius_eff_um, cv):
    """
    Calculates the mu and sigma for a log-normal distribution
    anchored to the OPTICAL effective radius, not the number mean.
    """
    if cv <= 0:
        return np.log(radius_eff_um), 1e-10

    # In atmospheric optics, effective variance (v_eff) is exactly CV^2
    v_eff = cv ** 2

    # Calculate the log-normal shape parameter sigma
    sigma_squared = np.log(1.0 + v_eff)
    sigma = np.sqrt(sigma_squared)

    mu = np.log(radius_eff_um) - 2.5 * sigma_squared

    return (mu, sigma)


def _normalize_phase(intensity_linear, theta_linear):
    """Helper to ensure energy conservation."""
    raw_integral = np.trapezoid(intensity_linear * np.sin(theta_linear), theta_linear) * 2.0 * np.pi
    return (intensity_linear / (raw_integral + 1e-12)).astype(np.float32), raw_integral


def _process_wavelength(w_nm, index, radii, weights, backend, config):
    N = config.num_angles
    theta_cheb = 0.5 * np.pi * (1.0 - np.cos(np.linspace(0, np.pi, N)))
    mu_cheb = np.cos(theta_cheb)
    theta_linear = np.linspace(0, np.pi, N)

    w_clamp = max(config.min_wavelength, min(config.max_wavelength, w_nm))
    m = get_material_ior(config.material, w_clamp)

    # Initialize with correct dimensions
    if config.num_phi_bins > 1:
        intensity_cheb = np.zeros((config.num_phi_bins, N))
    else:
        intensity_cheb = np.zeros(N)

    for j, r_um in enumerate(radii):
        intensity_cheb += weights[j] * backend.intensity_unpolarized(m, w_nm, r_um, mu_cheb)

    # Interpolation and Normalization
    if config.num_phi_bins > 1:
        intensity_linear = np.zeros((config.num_phi_bins, N))
        normalized_phase = np.zeros_like(intensity_linear, dtype=np.float32)
        raw_integral_sum = 0.0

        for p in range(config.num_phi_bins):
            intensity_linear[p, :] = np.interp(theta_linear, theta_cheb, intensity_cheb[p, :])
            norm_phase, raw_int = _normalize_phase(intensity_linear[p, :], theta_linear)
            normalized_phase[p, :] = norm_phase
            raw_integral_sum += raw_int

        raw_integral = raw_integral_sum / config.num_phi_bins
    else:
        intensity_linear = np.interp(theta_linear, theta_cheb, intensity_cheb)
        normalized_phase, raw_integral = _normalize_phase(intensity_linear, theta_linear)

    print(f"  [Sim] {w_nm:.1f}nm | Integral: {raw_integral:.4f}", flush=True)

    return index, normalized_phase


def generate_phase_table(config: MieConfig, backend: ScatteringBackend):
    theta = np.linspace(0, np.pi, config.num_angles)
    mu = np.cos(theta)
    wavelengths = np.linspace(config.min_wavelength, config.max_wavelength, config.num_wavelengths)

    num_cores = min(6, max(1, multiprocessing.cpu_count() - 2))

    print(f"[Gen] Generating Phase Table on {num_cores} CORES...")
    print(f"[Gen]   Config: {config.output_filename}")

    print(f"[Gen] Pre-warming JIT Cache for Reff={config.radius_mean_um:.2f}...")

    num_nodes = max(config.num_samples, 16) if config.num_samples > 1 else 1
    print(f"[Gen]   Mode: MULTI-PARTICLE INTEGRATION ({num_nodes} Quadrature Nodes)")

    # Setup Quadrature
    mu_log, sigma_log = _compute_lognormal_params(config.radius_mean_um, config.variance)
    x_nodes, w_nodes = hermgauss(num_nodes)
    radii = np.exp(np.sqrt(2.0) * sigma_log * x_nodes + mu_log)
    weights = (w_nodes / np.sqrt(np.pi)) * (radii ** 2)
    weights /= np.sum(weights)

    # 1. THE PRE-WARM (Single Thread)
    prewarm_results = []
    first_res = _process_wavelength(wavelengths[0], 0, radii, weights, backend, config)
    prewarm_results.append(first_res)

    # 2. THE GUILLOTINE POOL (Multi Thread)
    print("[Gen] Cache pre-warm complete. Unleashing the pool.")
    args_list = [(w, i, radii, weights, backend, config) for i, w in enumerate(wavelengths) if i > 0]

    with multiprocessing.Pool(processes=num_cores, maxtasksperchild=1) as pool:
        pool_results = pool.starmap(_process_wavelength, args_list)

    # Merge and sort
    results = prewarm_results + pool_results
    results.sort(key=lambda x: x[0])

    # Dynamic allocation based on anisotropy
    if config.num_phi_bins > 1:
        phase_table = np.zeros((config.num_wavelengths, config.num_phi_bins, config.num_angles), dtype=np.float32)
    else:
        phase_table = np.zeros((config.num_wavelengths, config.num_angles), dtype=np.float32)

    for idx, data in results:
        phase_table[idx, ...] = data

    return phase_table, mu, wavelengths


def generate_mie_table(config: MieConfig = MieConfig()):
    return generate_phase_table(config, backend=MiePythonBackend())


import struct
import numpy as np


def save_binary_file(filename, phase_table, mu_vals, wavelengths, config: MieConfig):
    # Incoming phase_table shape: (Wavelength, Phi, Theta)

    # We want memory layout: (Theta, Phi, Wavelength)
    # 1. Swap Wavelength (0) and Theta (2) -> (Theta, Phi, Wavelength)
    if config.num_phi_bins > 1:
        phase_interleaved = np.moveaxis(phase_table, 0, -1)
        phase_interleaved = np.swapaxes(phase_interleaved, 0, 1)  # Ensure Theta is axis 0
    else:
        # Fallback for old 1D spherical data (Wavelength, Theta) -> (Theta, Wavelength)
        phase_interleaved = phase_table.T

    data_flat = phase_interleaved.flatten().astype(np.float32)

    with open(filename, "wb") as f:
        f.write(b"ATMPHASE")
        f.write(struct.pack("<I", 2))  # VERSION 2 HEADER
        f.write(struct.pack("<I", config.num_angles))  # Theta count
        f.write(struct.pack("<I", config.num_phi_bins))  # Phi count
        f.write(struct.pack("<I", config.num_wavelengths))  # Wavelength count
        f.write(struct.pack("<f", float(config.min_wavelength)))
        f.write(struct.pack("<f", float(config.max_wavelength)))
        f.write(data_flat.tobytes())

    print(f"[GEN] Saved anisotropic v2 binary: {filename}")

    # (Your lobe_analyzer will need an update to handle 3D data,
    #  you might want to bypass it temporarily if num_phi_bins > 1)
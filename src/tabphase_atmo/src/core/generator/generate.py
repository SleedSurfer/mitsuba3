import numpy as np
import struct
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
import multiprocessing
from scipy import stats
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


# ==============================================================================
# INTEGRATION MODE 1: THE FAST PATH (1 Particle + Angular Gaussian Blur)
# ==============================================================================
def _process_wavelength_fast(w_nm, index, backend, config):
    N = config.num_angles
    theta_cheb = 0.5 * np.pi * (1.0 - np.cos(np.linspace(0, np.pi, N)))
    mu_cheb = np.cos(theta_cheb)
    theta_linear = np.linspace(0, np.pi, N)

    w_clamp = max(config.min_wavelength, min(config.max_wavelength, w_nm))
    m = get_material_ior(config.material, w_clamp)

    # 1. Evaluate only the mean radius
    intensity_cheb = backend.intensity_unpolarized(m, w_nm, config.radius_mean_um, mu_cheb)
    intensity_linear = np.interp(theta_linear, theta_cheb, intensity_cheb)

    # 2. Smear the high-frequency fringes based on the variance
    if config.variance > 0.0:
        dtheta = np.pi / config.num_angles
        sigma_rad = config.variance * 0.15  # Empirical tuning scalar
        sigma_bins = sigma_rad / dtheta
        intensity_linear = gaussian_filter1d(intensity_linear, sigma=sigma_bins, mode='nearest')

    normalized_phase, raw_integral = _normalize_phase(intensity_linear, theta_linear)
    print(f"  [Fast] {w_nm:.1f}nm | Integral: {raw_integral:.4f}", flush=True)

    return index, normalized_phase


# ==============================================================================
# INTEGRATION MODE 2: THE MASOCHIST PATH (Brute Force N-Particle Summation)
# ==============================================================================
def _process_wavelength_brute(w_nm, index, radii, weights, backend, config):
    N = config.num_angles
    theta_cheb = 0.5 * np.pi * (1.0 - np.cos(np.linspace(0, np.pi, N)))
    mu_cheb = np.cos(theta_cheb)
    theta_linear = np.linspace(0, np.pi, N)

    w_clamp = max(config.min_wavelength, min(config.max_wavelength, w_nm))
    m = get_material_ior(config.material, w_clamp)

    intensity_cheb = np.zeros_like(mu_cheb)

    for j, r_um in enumerate(radii):
        intensity_cheb += weights[j] * backend.intensity_unpolarized(m, w_nm, r_um, mu_cheb)

    intensity_linear = np.interp(theta_linear, theta_cheb, intensity_cheb)

    normalized_phase, raw_integral = _normalize_phase(intensity_linear, theta_linear)
    print(f"  [Brute] {w_nm:.1f}nm | Integral: {raw_integral:.4f}", flush=True)

    return index, normalized_phase


def generate_phase_table(config: MieConfig, backend: ScatteringBackend):
    theta = np.linspace(0, np.pi, config.num_angles)
    mu = np.cos(theta)
    wavelengths = np.linspace(config.min_wavelength, config.max_wavelength, config.num_wavelengths)

    num_cores = min(6, max(1, multiprocessing.cpu_count() - 2))

    print(f"[Gen] Generating Phase Table on {num_cores} CORES...")
    print(f"[Gen]   Config: {config.output_filename}")

    # ==========================================
    # 1. THE PRE-WARM (Single Thread)
    # ==========================================
    # Compiles the AST kernel and writes to the SSD safely so the pool workers don't fight. TODO dont touch
    print(f"[Gen] Pre-warming JIT Cache for Reff={config.radius_mean_um:.2f}...")
    prewarm_results = []

    if config.brute_force_integration:
        num_nodes = max(config.num_samples, 16) if config.num_samples > 1 else 1
        print(f"[Gen]   Mode: SMART BRUTE FORCE ({num_nodes} Quadrature Nodes)")

        mu_log, sigma_log = _compute_lognormal_params(config.radius_mean_um, config.variance)
        x_nodes, w_nodes = hermgauss(num_nodes)
        radii = np.exp(np.sqrt(2.0) * sigma_log * x_nodes + mu_log)
        weights = (w_nodes / np.sqrt(np.pi)) * (radii ** 2)
        weights /= np.sum(weights)

        # Execute the first wavelength synchronously
        first_res = _process_wavelength_brute(wavelengths[0], 0, radii, weights, backend, config)
        prewarm_results.append(first_res)

        # Pack args for the rest (skip index 0)
        args_list = [(w, i, radii, weights, backend, config) for i, w in enumerate(wavelengths) if i > 0]
        target_func = _process_wavelength_brute

    else:
        print(f"[Gen]   Mode: FAST CONVOLUTION (1 Particle + Angular Blur)")

        first_res = _process_wavelength_fast(wavelengths[0], 0, backend, config)
        prewarm_results.append(first_res)

        args_list = [(w, i, backend, config) for i, w in enumerate(wavelengths) if i > 0]
        target_func = _process_wavelength_fast

    print("[Gen] Cache pre-warm complete. Unleashing the guillotine pool.")

    # ==========================================
    # 2. THE GUILLOTINE POOL (Multi Thread)
    # ==========================================
    # maxtasksperchild=1 forces the worker to die and release C++ memory after every task
    with multiprocessing.Pool(processes=num_cores, maxtasksperchild=1) as pool:
        pool_results = pool.starmap(target_func, args_list)

    # Merge and sort
    results = prewarm_results + pool_results
    results.sort(key=lambda x: x[0])

    phase_table = np.zeros((config.num_wavelengths, config.num_angles), dtype=np.float32)
    for idx, data in results:
        phase_table[idx, :] = data

    return phase_table, mu, wavelengths


def generate_mie_table(config: MieConfig = MieConfig()):
    return generate_phase_table(config, backend=MiePythonBackend())


def save_binary_file(filename, phase_table, mu_vals, wavelengths, config: MieConfig):
    phase_interleaved = phase_table.T
    data_flat = phase_interleaved.flatten().astype(np.float32)

    with open(filename, "wb") as f:
        f.write(b"ATMPHASE")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", config.num_angles))
        f.write(struct.pack("<I", config.num_wavelengths))
        f.write(struct.pack("<f", float(config.min_wavelength)))
        f.write(struct.pack("<f", float(config.max_wavelength)))
        f.write(data_flat.tobytes())

    print(f"[GEN] Saved binary: {filename}")

    lobes, weights = analyze_lut_for_mis(phase_table, mu_vals, wavelengths)
    with open(filename, "ab") as f:
        write_mis_metadata(f, lobes, weights)
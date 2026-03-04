import numpy as np
import struct
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
import multiprocessing
from scipy import stats
from numpy.polynomial.hermite import hermgauss

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

    # Your input "variance" percentage is actually the Coefficient of Variation (CV)
    # In atmospheric optics, effective variance (v_eff) is exactly CV^2
    v_eff = cv ** 2

    # Calculate the log-normal shape parameter sigma
    sigma_squared = np.log(1.0 + v_eff)
    sigma = np.sqrt(sigma_squared)

    # THE FIX: Anchor the log-normal location parameter mu to the
    # EFFECTIVE optical radius, not the arithmetic number mean.
    # r_eff = exp(mu + 2.5 * sigma^2)  -->  mu = ln(r_eff) - 2.5 * sigma^2
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

    # 1. Brutalize the CPU over N particles
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
    num_cores = multiprocessing.cpu_count()

    print(f"[Gen] Generating Phase Table on {num_cores} CORES...")
    print(f"[Gen]   Config: {config.output_filename}")

    # --- ROUTER ---
    if config.brute_force_integration:
        # 32 nodes with Gauss-Hermite is usually equivalent to like 1000 linear samples
        num_nodes = max(config.num_samples, 32)
        print(f"[Gen]   Mode: SMART BRUTE FORCE ({num_nodes} Quadrature Nodes)")

        mu_log, sigma_log = _compute_lognormal_params(config.radius_mean_um, config.variance)

        # Get optimal Gaussian nodes and weights
        x_nodes, w_nodes = hermgauss(num_nodes)

        # Transform nodes from standard normal space to our specific log-normal radii
        radii = np.exp(np.sqrt(2.0) * sigma_log * x_nodes + mu_log)

        # The probability distribution is inherently baked into w_nodes!
        # We just multiply by r^2 to account for the physical scattering cross-section area
        weights = (w_nodes / np.sqrt(np.pi)) * (radii ** 2)

        # Normalize weights so we don't blow up the energy
        weights /= np.sum(weights)

        results = Parallel(n_jobs=-1)(
            delayed(_process_wavelength_brute)(w, i, radii, weights, backend, config)
            for i, w in enumerate(wavelengths)
        )
    else:
        print(f"[Gen]   Mode: FAST CONVOLUTION (1 Particle + Angular Blur)")
        results = Parallel(n_jobs=-1)(
            delayed(_process_wavelength_fast)(w, i, backend, config)
            for i, w in enumerate(wavelengths)
        )

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
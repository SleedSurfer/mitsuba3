"""
generate.py
"""
import numpy as np
import struct
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing

# Use relative import for the analyzer and internal config
from .lobe_analyzer import analyze_lut_for_mis, write_mis_metadata
from ..config import MieConfig
from ..materials import get_material_ior

from .backends.base import ScatteringBackend
from .backends.mie import MiePythonBackend


def _compute_lognormal_params(radius_mean_um, radius_std_um):
    if radius_std_um <= 0:
        return np.log(radius_mean_um), 1e-10
    cv_squared = (radius_std_um / radius_mean_um) ** 2
    sigma_squared = np.log(1 + cv_squared)
    sigma = np.sqrt(sigma_squared)
    mu = np.log(radius_mean_um) - sigma_squared / 2.0
    return (mu, sigma)


def _process_wavelength(
        w_nm: float,
        index: int,
        radii: np.ndarray,
        weights: np.ndarray,
        # We don't pass a grid anymore, we construct the optimal one internally
        unused_grid_arg,
        backend: ScatteringBackend,
        config: MieConfig
):
    # --- 1. CHEBYSHEV GRID (Optimal Distribution) ---
    # Same N as output, just moved to where they matter.
    N = config.num_angles

    # Warped Phase: 0 -> Pi
    k = np.arange(N)
    theta_cheb = np.arccos(np.cos(k * np.pi / (N - 1)))  # Simplified Chebyshev nodes (0 to Pi)
    # Actually, simpler form for 0->Pi mapping:
    theta_cheb = 0.5 * np.pi * (1.0 - np.cos(np.linspace(0, np.pi, N)))

    mu_cheb = np.cos(theta_cheb)

    # --- 2. PHYSICS (Same Cost) ---
    w_clamp = max(config.min_wavelength, min(config.max_wavelength, w_nm))
    m = get_material_ior(config.material, w_clamp)
    w_um = w_clamp / 1000.0

    intensity_cheb = np.zeros_like(mu_cheb)

    for j, r_um in enumerate(radii):
        x = 2 * np.pi * r_um / w_um
        physical_weight = weights[j] * (r_um ** 2)
        intensity_cheb += physical_weight * backend.intensity_unpolarized(m, w_nm, r_um, mu_cheb)

    # --- 3. RE-GRIDDING (Linear Interpolation) ---
    # We must save to a linear grid for the Mitsuba plugin.
    theta_linear = np.linspace(0, np.pi, N)

    # Interpolate Intensity from Cheb -> Linear
    # This is safe because Cheb is dense where intensity changes fast (Peak)
    # and sparse where intensity changes slow (Side lobes).
    intensity_linear = np.interp(theta_linear, theta_cheb, intensity_cheb)

    # --- 4. NORMALIZATION ---
    # Standard integration on the output grid
    raw_integral = np.trapezoid(intensity_linear * np.sin(theta_linear), theta_linear) * 2 * np.pi

    normalized_phase = intensity_linear / (raw_integral + 1e-12)

    # Debug: Check if we captured the energy this time
    print(f"  [Gen] {w_nm:.1f}nm | Integral Capture: {raw_integral:.4f}", flush=True)

    return index, normalized_phase.astype(np.float32)


def generate_phase_table(config: MieConfig, backend: ScatteringBackend):
    # --- ANGLE SPACE GRID ---
    # Linear in Theta (Radians)
    # 0 = Forward (mu=1), Pi = Backward (mu=-1)
    theta = np.linspace(0, np.pi, config.num_angles)
    mu = np.cos(theta)

    mu_log, sigma_log = _compute_lognormal_params(config.radius_mean_um, config.radius_std_um)
    lognorm_dist = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))

    lower_bound = np.exp(mu_log - 3.0 * sigma_log)
    upper_bound = np.exp(mu_log + 3.0 * sigma_log)

    radii = np.logspace(np.log10(lower_bound), np.log10(upper_bound), config.num_samples)
    weights = lognorm_dist.pdf(radii)
    weights /= np.sum(weights)

    wavelengths = np.linspace(config.min_wavelength, config.max_wavelength, config.num_wavelengths)

    num_cores = multiprocessing.cpu_count()
    print(f"Generating Phase Table on {num_cores} CORES...")
    print(f"  Config: {config.output_filename}")
    print(f"  Grid: Linear Angle Space (Theta 0->Pi)")
    print(f"  Mode: ZERO-PROCESS (Raw)")

    results = Parallel(n_jobs=-1)(
        delayed(_process_wavelength)(w, i, radii, weights, mu, backend, config)
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
    """
    Save phase table to binary .bin file with ATMPHASE format.
    Handles both LUT data and MIS metadata.

    Args:
        filename: Output file path
        phase_table: Phase function data [num_wavelengths, num_angles]
        mu_vals: Cosine of scattering angles (for MIS analysis)
        wavelengths: Wavelength array in nm
        config: MieConfig for metadata
    """
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

    print(f"[GEN] Analyzing LUT for MIS metadata...")
    lobes, weights = analyze_lut_for_mis(phase_table, mu_vals, wavelengths)

    with open(filename, "ab") as f:
        write_mis_metadata(f, lobes, weights)
    print(f"[GEN] MIS metadata appended.")


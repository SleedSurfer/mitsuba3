"""
generate.py
Parallelized Mie scattering generator using MieConfig.
"""
import numpy as np
import struct
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing

from .lobe_analyzer import analyze_lut_for_mis, write_mis_metadata
from .mie_backend import mie_unpolarized_intensity
from python.atmospheric.config import MieConfig


def get_water_ior(wavelength_nm):  # Sellmeier
    w_um = wavelength_nm / 1000.0
    A = [5.666959820e-1, 1.731900098e-1, 2.026271125e-2, 1.139365864e-1]
    B = [5.084151894e-3, 1.818488474e-2, 2.625439472e-2, 1.073842352e1]  # daimon & masamura 2007
    n_squared = 1.0
    for i in range(4):
        n_squared += (A[i] * w_um**2) / (w_um**2 - B[i])
    return complex(np.sqrt(n_squared), 0.0)  # albedo is zero rn


def _compute_lognormal_params(radius_mean_um, radius_std_um):
    if radius_std_um <= 0:
        return np.log(radius_mean_um), 1e-10
    cv_squared = (radius_std_um / radius_mean_um) ** 2
    sigma_squared = np.log(1 + cv_squared)
    sigma = np.sqrt(sigma_squared)
    mu = np.log(radius_mean_um) - sigma_squared / 2.0
    return (mu, sigma)


# --- Core Logic ---
def process_wavelength(w_nm, index, radii, weights, mu, d_mu):
    m = get_water_ior(w_nm)
    weighted_phase = np.zeros_like(mu)

    for j, r in enumerate(radii):
        x = 2 * np.pi * r / (w_nm / 1000.0)
        intensity = mie_unpolarized_intensity(m, x, mu)
        weighted_phase += weights[j] * intensity

    # Normalization (Energy Conservation)
    raw_integral = np.sum(weighted_phase) * d_mu * 2 * np.pi
    normalized_phase = weighted_phase / (raw_integral + 1e-12)  # Safety epsilon

    print(f"  [Core] Finished {w_nm:.1f}nm | Raw Int: {raw_integral:.2f}")
    return index, normalized_phase.astype(np.float32)


def generate_mie_table(config: MieConfig = MieConfig()):
    """
    Main entry point. Uses config object for all parameters.
    """
    # 1. Setup Ranges
    wavelengths = np.linspace(config.min_wavelength, config.max_wavelength, config.num_wavelengths)
    mu = np.linspace(1, -1, config.num_angles)
    d_mu = 2.0 / (config.num_angles - 1)

    # 2. Physics Distribution (Grid Method)
    mu_log, sigma_log = _compute_lognormal_params(config.radius_mean_um, config.radius_std_um)
    lognorm_dist = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))

    lower_bound = np.exp(mu_log - 3.0 * sigma_log)
    upper_bound = np.exp(mu_log + 3.0 * sigma_log)

    # Fixed Grid Sampling (Deterministic)
    radii = np.logspace(np.log10(lower_bound), np.log10(upper_bound), config.num_samples)
    weights = lognorm_dist.pdf(radii)
    weights /= np.sum(weights)

    # 3. Execution
    num_cores = multiprocessing.cpu_count()
    print(f"Generating Mie Table on {num_cores} CORES...")
    print(f"  Config: {config.output_filename}")
    print(f"  Range: {lower_bound:.2f}um to {upper_bound:.2f}um")

    results = Parallel(n_jobs=-1)(
        delayed(process_wavelength)(w, i, radii, weights, mu, d_mu)
        for i, w in enumerate(wavelengths)
    )
    results.sort(key=lambda x: x[0])

    # 4. Assembly
    phase_table = np.zeros((config.num_wavelengths, config.num_angles), dtype=np.float32)
    for idx, data in results:
        phase_table[idx, :] = data

    return phase_table


def save_binary_file(filename, phase_table, config: MieConfig):
    """
    Saves binary using config parameters for header.
    """
    phase_interleaved = phase_table.T
    data_flat = phase_interleaved.flatten().astype(np.float32)

    with open(filename, 'wb') as f:
        f.write(b'ATMPHASE')
        f.write(struct.pack('<I', 1))  # Version
        f.write(struct.pack('<I', config.num_angles))
        f.write(struct.pack('<I', config.num_wavelengths))
        f.write(struct.pack('<f', float(config.min_wavelength)))
        f.write(struct.pack('<f', float(config.max_wavelength)))

        f.write(data_flat.tobytes())

    print(f"[GEN] Saved binary: {filename}")

    print(f"[GEN] Analyzing LUT for MIS metadata...")
    mu_vals = np.linspace(1, -1, config.num_angles)
    wavelengths = np.linspace(config.min_wavelength, config.max_wavelength, config.num_wavelengths)

    # Analyze (phase_table is [num_wavelengths, num_angles])
    lobes, weights = analyze_lut_for_mis(phase_table, mu_vals, wavelengths)

    # Print detected lobes
    print(f"[GEN] Detected {len(lobes)} lobes:")
    for lobe in lobes:
        lobe_type = ['Forward', 'Rainbow', 'Residual', 'Glory'][lobe['type']]
        wl_dep = " (λ-dependent)" if lobe.get('wavelength_dependent', False) else ""
        print(
            f"[GEN]  - {lobe_type}{wl_dep}: mu={lobe['mu_center']:.3f}, κ={lobe['kappa']:.1f}, amp={lobe['amplitude']:.3e}"
        )

    print(
        f"[GEN] Mixture weights: forward={weights['forward']:.3f}, rainbow={weights['rainbow']:.3f}, residual={weights['residual']:.3f}"
    )

    # Append MIS metadata to file
    with open(filename, 'ab') as f:  # Append mode
        write_mis_metadata(f, lobes, weights)

    print(f"[GEN] MIS metadata appended to {filename}")
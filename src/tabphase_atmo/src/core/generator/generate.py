import struct
import multiprocessing
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import j1

# Import your new configs
from src.core.config import BaseParticleConfig, DropletConfig, HexagonalConfig

from .backends.base import ScatteringBackend,HexSize,SphereSize


def get_airy_diffraction(theta_rad, radius_um, wavelength_nm):
    """Computes Fraunhofer diffraction forward peak for a perfect sphere."""
    wavelength_um = wavelength_nm / 1000.0
    size_param = (2.0 * np.pi * radius_um) / wavelength_um

    theta_safe = np.maximum(theta_rad, 1e-7)
    u = size_param * np.sin(theta_safe)

    airy_intensity = (2.0 * j1(u) / u) ** 2
    return airy_intensity * (size_param ** 2) / (4.0 * np.pi)


def apply_faux_polydispersity(intensity_array, theta_rad, variance):
    """
    Log-Space Faux Polydispersity.
    Applies a spatially expanding blur without breaking physics/energy conservation.
    """
    if variance <= 0.001:
        return intensity_array

    N = len(theta_rad)
    result_log = np.zeros_like(intensity_array)
    d_theta = theta_rad[1] - theta_rad[0]
    sin_theta = np.sin(theta_rad)

    original_integral = np.trapezoid(intensity_array * sin_theta, theta_rad)
    log_intensity = np.log(np.maximum(intensity_array, 1e-12))
    x_indices = np.arange(N)

    # Physical anchor for water's primary rainbow
    theta_rbw = np.radians(138.0)

    for i, theta in enumerate(theta_rad):
        # 1. Forward Region Physics: Rings expand from 0 degrees (scales linearly)
        spread_fwd = theta * variance

        # 2. Backward Region Physics: Fringes expand from the rainbow anchor.
        # Spacing scales as R^(-2/3) per Airy theory. We add 0.1 rad (~5 deg) of
        # baseline distance to simulate the minor shift of the main geometric peak.
        spread_rbw = (2.0 / 3.0) * (np.abs(theta - theta_rbw) + 0.1) * variance

        blend = 0.5 * (1.0 + np.tanh((theta - np.pi / 2) * 4.0))

        spread_rad = (1.0 - blend) * spread_fwd + blend * spread_rbw + 1e-5

        sigma_idx = max(0.5, spread_rad / d_theta)

        weights = np.exp(-0.5 * ((x_indices - i) / sigma_idx) ** 2)
        weights /= np.sum(weights)

        result_log[i] = np.sum(log_intensity * weights)

    smoothed_intensity = np.exp(result_log)
    new_integral = np.trapezoid(smoothed_intensity * sin_theta, theta_rad)

    if new_integral > 1e-12:
        smoothed_intensity *= (original_integral / new_integral)

    return smoothed_intensity

def _compute_lognormal_params(radius_eff_um, cv):
    if cv <= 0:
        return np.log(radius_eff_um), 1e-10
    v_eff = cv ** 2
    sigma_squared = np.log(1.0 + v_eff)
    sigma = np.sqrt(sigma_squared)
    mu = np.log(radius_eff_um) - 2.5 * sigma_squared
    return (mu, sigma)


def _normalize_phase(intensity_linear, theta_linear):
    raw_integral = np.trapezoid(intensity_linear * np.sin(theta_linear), theta_linear) * 2.0 * np.pi
    return (intensity_linear / (raw_integral + 1e-12)).astype(np.float32), raw_integral


def _process_wavelength(w_nm, index, size_params, weights, backend, config, habit_params):
    N = config.num_angles
    theta_cheb = 0.5 * np.pi * (1.0 - np.cos(np.linspace(0, np.pi, N)))
    mu_cheb = np.cos(theta_cheb)
    theta_linear = np.linspace(0, np.pi, N)

    # USE THE PRECOMPUTED IOR FROM THE CONFIG
    m = config.computed_iors[index]

    if config.num_phi_bins > 1:
        intensity_cheb = np.zeros((config.num_phi_bins, N))
    else:
        intensity_cheb = np.zeros(N)

    for j, current_size in enumerate(size_params):
        intensity_cheb += weights[j] * backend.intensity_unpolarized(m, w_nm, mu_cheb, current_size)

    if config.num_phi_bins > 1:
        intensity_linear = np.zeros((config.num_phi_bins, N))

        # 1. Interpolate all rows FIRST without touching their magnitudes
        for p in range(config.num_phi_bins):
            intensity_linear[p, :] = np.interp(theta_linear, theta_cheb, intensity_cheb[p, :])
    else:
        intensity_linear = np.interp(theta_linear, theta_cheb, intensity_cheb)

    # --- WAVE OPTICS & POLYDISPERSITY APPLIED IN THE WORKER ---
    needs_diffraction = isinstance(config, DropletConfig)
    needs_polydispersity = "variance" in habit_params and habit_params["variance"] > 0.0

    if needs_diffraction:
        diffraction_peak = get_airy_diffraction(theta_linear, habit_params["radius"], w_nm)
        if config.num_phi_bins > 1:
            for p in range(config.num_phi_bins):
                intensity_linear[p, :] += diffraction_peak
        else:
            intensity_linear += diffraction_peak

    if needs_polydispersity:
        var = habit_params["variance"]
        if config.num_phi_bins > 1:
            for p in range(config.num_phi_bins):
                intensity_linear[p, :] = apply_faux_polydispersity(intensity_linear[p, :], theta_linear, var)
        else:
            intensity_linear = apply_faux_polydispersity(intensity_linear, theta_linear, var)

    # --- FINALLY: NORMALIZE AFTER ALL ENERGY IS ADDED ---
    if config.num_phi_bins > 1:
        # Calculate the global 2D integral across the sphere
        theta_integrals = np.trapezoid(intensity_linear * np.sin(theta_linear), theta_linear, axis=1)
        d_phi = (2.0 * np.pi) / config.num_phi_bins
        global_integral = np.sum(theta_integrals * d_phi)

        normalized_phase = (intensity_linear / (global_integral + 1e-12)).astype(np.float32)
        raw_integral = global_integral / (2.0 * np.pi)
    else:
        normalized_phase, raw_integral = _normalize_phase(intensity_linear, theta_linear)

    print(f"  [Sim] {w_nm:.1f}nm | IOR: {m:.4f} | Integral: {raw_integral:.4f}", flush=True)

    return index, normalized_phase


def generate_phase_table(config: BaseParticleConfig, backend: ScatteringBackend, habit_params: dict):
    wavelengths = np.array(config.wavelengths_nm)
    num_cores = min(6, max(1, multiprocessing.cpu_count() - 2))
    print(f"[Gen] Generating Phase Table on {num_cores} CORES...")

    # Set up the single geometry trace (no more quadrature arrays!)
    if isinstance(config, DropletConfig):
        r_mean = habit_params["radius"]
        print(f"[Gen] Mode: DROPLET (radius={r_mean}um)")
        size_params = [SphereSize(r_um=r_mean)]
        weights = np.array([1.0])

    elif isinstance(config, HexagonalConfig):
        c_ax = habit_params["c_axis"]
        a_ax = habit_params["a_axis"]
        print(f"[Gen] Mode: CRYSTAL HABIT (c={c_ax}um, a={a_ax}um)")
        size_params = [HexSize(c_axis_um=c_ax, a_axis_um=a_ax)]
        weights = np.array([1.0])
    else:
        raise ValueError("Unknown Config Type in Generator")

    print("[Gen] Unleashing the pool.")
    args_list = [(w, i, size_params, weights, backend, config, habit_params) for i, w in enumerate(wavelengths)]

    if args_list:
        with multiprocessing.Pool(processes=num_cores, maxtasksperchild=1) as pool:
            results = pool.starmap(_process_wavelength, args_list)
    else:
        results = []

    results.sort(key=lambda x: x[0])

    if config.num_phi_bins > 1:
        phase_table = np.zeros((len(wavelengths), config.num_phi_bins, config.num_angles), dtype=np.float32)
    else:
        phase_table = np.zeros((len(wavelengths), config.num_angles), dtype=np.float32)

    for idx, data in results:
        phase_table[idx, ...] = data

    theta = np.linspace(0, np.pi, config.num_angles)

    print("[Gen] Raytracing and Wave Optics complete. Exiting Generator.")

    return phase_table, np.cos(theta), wavelengths




def save_binary_file(filename, phase_table, mu_vals, wavelengths, config: BaseParticleConfig):
    # Same logic as before, just using BaseParticleConfig types
    if config.num_phi_bins > 1:
        phase_interleaved = np.moveaxis(phase_table, 0, -1)
        phase_interleaved = np.swapaxes(phase_interleaved, 0, 1)
    else:
        phase_interleaved = phase_table.T

    data_flat = phase_interleaved.flatten().astype(np.float32)

    with open(filename, "wb") as f:
        f.write(b"ATMPHASE")
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<I", config.num_angles))
        f.write(struct.pack("<I", config.num_phi_bins))
        f.write(struct.pack("<I", len(wavelengths)))
        f.write(struct.pack("<f", float(wavelengths[0])))
        f.write(struct.pack("<f", float(wavelengths[-1])))
        f.write(data_flat.tobytes())

    print(f"[GEN] Saved anisotropic v2 binary: {filename}")
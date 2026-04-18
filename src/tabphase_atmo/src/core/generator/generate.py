import struct
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import os
from scipy.special import j1
from scipy.ndimage import gaussian_filter1d
from src.core.config import BaseParticleConfig, DropletConfig, HexagonalConfig
from .backends.base import ScatteringBackend, HexSize, SphereSize
from .backends.tracer_dispatcher import DrJitRaytracerBackend


def apply_caustic_diffraction_blur(intensity_array, theta_rad, radius_um):
    """
    Caustic Regularizer.
    Only smooths the mathematical singularities of the primary and secondary rainbows,
    leaving the geometric supernumerary interference fringes untouched.
    """
    base_sigma_deg = 0.7 * (100.0 / max(radius_um, 1.0)) ** (2.0 / 3.0)
    sigma_rad = np.radians(base_sigma_deg)

    d_theta = theta_rad[1] - theta_rad[0]
    sigma_bins = max(0.5, sigma_rad / d_theta)

    smoothed = gaussian_filter1d(intensity_array, sigma=sigma_bins, mode='nearest')
    mask_pri = (theta_rad >= np.radians(135.0)) & (theta_rad <= np.radians(142.0))
    mask_sec = (theta_rad >= np.radians(125.0)) & (theta_rad <= np.radians(132.0))

    peak_idx_pri = np.argmax(np.where(mask_pri, intensity_array, 0.0))
    peak_idx_sec = np.argmax(np.where(mask_sec, intensity_array, 0.0))

    window_pri = np.exp(-0.5 * ((np.arange(len(theta_rad)) - peak_idx_pri) / (sigma_bins * 2.0)) ** 2)
    window_sec = np.exp(-0.5 * ((np.arange(len(theta_rad)) - peak_idx_sec) / (sigma_bins * 2.0)) ** 2)

    combined_window = np.clip(window_pri + window_sec, 0.0, 1.0)

    final_intensity = (intensity_array * (1.0 - combined_window)) + (smoothed * combined_window)

    return np.maximum(final_intensity, 0.0)

def _get_effective_radius(size_params) -> float:
    """ Calculates R_eff using the Cauchy Average Projected Area theorem. """
    param = size_params[0]
    if isinstance(param, SphereSize):
        return param.r_um
    elif isinstance(param, HexSize):
        a = param.a_axis_um
        h = param.c_axis_um
        S_total = 3.0 * np.sqrt(3.0) * (a ** 2) + 6.0 * a * h
        r_eff = np.sqrt(S_total / (4.0 * np.pi))
        return r_eff
    else:
        raise ValueError("Unknown size parameter for D_eff calculation.")

def _normalize_phase(intensity_linear, theta_linear):
    raw_integral = np.trapezoid(intensity_linear * np.sin(theta_linear), theta_linear) * 2.0 * np.pi
    return (intensity_linear / (raw_integral + 1e-12)).astype(np.float32), raw_integral

def _process_wavelength(w_nm, index, size_params, backend, config):
    N = config.num_angles
    theta_linear = np.linspace(0, np.pi, N)
    m = config.computed_iors[index]

    if config.num_phi_bins > 1:
        intensity_linear = np.zeros((config.num_phi_bins, N))
    else:
        intensity_linear = np.zeros(N)

    current_size = size_params[0]
    intensity_linear += backend.intensity_unpolarized(m, w_nm, theta_linear, current_size)

    if isinstance(backend, DrJitRaytracerBackend) and isinstance(current_size, SphereSize):
        r_eff = _get_effective_radius(size_params)
        if config.num_phi_bins > 1:
            for p in range(config.num_phi_bins):
                intensity_linear[p, :] = apply_caustic_diffraction_blur(intensity_linear[p, :], theta_linear, r_eff)
        else:
            intensity_linear = apply_caustic_diffraction_blur(intensity_linear, theta_linear, r_eff)


    return index, intensity_linear.astype(np.float32)


def generate_phase_table(config: BaseParticleConfig, backend: ScatteringBackend, habit_params: dict):
    wavelengths = np.array(config.wavelengths_nm)
    num_cores = min(6, max(1, multiprocessing.cpu_count() - 2))
    print(f"[Gen] Generating Phase Table on {num_cores} CORES...")

    if isinstance(config, DropletConfig):
        r_mean = habit_params["radius"]
        print(f"[Gen] Mode: SPHERE (radius={r_mean}um)")
        size_params = [SphereSize(r_um=r_mean)]
    elif isinstance(config, HexagonalConfig):
        c_ax = habit_params["c_axis"]
        a_ax = habit_params["a_axis"]
        print(f"[Gen] Mode: PRISM (c={c_ax}um, a={a_ax}um)")
        size_params = [HexSize(c_axis_um=c_ax, a_axis_um=a_ax)]
    else:
        raise ValueError("Unknown Config Type in Generator")
    args_list = [(w, i, size_params, backend, config) for i, w in enumerate(wavelengths)]

    results = []
    total_wavelengths = len(args_list)
    if args_list:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_cores, mp_context=ctx) as ex:
            futures = [
                ex.submit(_process_wavelength, w, i, size_params, backend, config)
                for i, w in enumerate(wavelengths)
            ]

            for done, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                print(f"\r[Gen] Generating wavelengths ({done}/{total_wavelengths})", end="", flush=True)

        print()
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

    print("[Gen] LUT shape generated, exiting...")

    return phase_table, np.cos(theta), wavelengths


def save_binary_file(filename, phase_table, mu_vals, wavelengths, config: BaseParticleConfig):
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

    print(f"[GEN] Saved binary: {os.path.basename(filename)}")
import struct
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import os
from scipy.special import j1
from scipy.ndimage import gaussian_filter1d
from nimbuscore.core.config import BaseParticleConfig, DropletConfig, HexagonalConfig, CuboctahedralConfig
from .backends.base import ScatteringBackend, HexSize, SphereSize, CuboctaSize
from .backends.tracer_dispatcher import DrJitRaytracerBackend


import numpy as np
from scipy.ndimage import convolve1d
from scipy.stats import skewnorm

import numpy as np
from scipy.ndimage import gaussian_filter1d

import numpy as np
from scipy.ndimage import gaussian_filter1d


def apply_caustic_diffraction_blur(intensity_array, theta_rad, radius_um, w_nm):
    wavelength_um = w_nm / 1000.0
    x = (2.0 * np.pi * max(radius_um, 1e-6)) / max(wavelength_um, 1e-6)

    # Generalized analytical formula
    base_sigma_deg = 60.3 * (x ** (-2.0 / 3.0))
    sigma_rad = np.radians(base_sigma_deg)

    d_theta = np.abs(theta_rad[1] - theta_rad[0])
    sigma_bins = max(0.5, sigma_rad / d_theta)

    # The raw blurs
    smoothed_pri = gaussian_filter1d(intensity_array, sigma=sigma_bins, mode='nearest')
    smoothed_sec = gaussian_filter1d(intensity_array, sigma=sigma_bins * 2.0, mode='nearest')

    mask_pri = (theta_rad >= np.radians(135.0)) & (theta_rad <= np.radians(142.0))
    mask_sec = (theta_rad >= np.radians(125.0)) & (theta_rad <= np.radians(132.0))

    peak_idx_pri = np.argmax(np.where(mask_pri, intensity_array, 0.0))
    peak_idx_sec = np.argmax(np.where(mask_sec, intensity_array, 0.0))

    idx = np.arange(len(theta_rad))
    mid_idx = (peak_idx_pri + peak_idx_sec) // 2

    # --- ASYMMETRIC WINDOW FOR PRIMARY BOW ---
    window_pri = np.zeros_like(theta_rad)
    # 1. Shadow side (Alexander's band): Keep blur at 100% to thicken the band
    mask_pri_shadow = (idx >= mid_idx) & (idx <= peak_idx_pri)
    window_pri[mask_pri_shadow] = 1.0
    # 2. Supernumerary side: Fade out the blur to protect the interference fringes
    mask_pri_super = idx > peak_idx_pri
    window_pri[mask_pri_super] = np.exp(-0.5 * ((idx[mask_pri_super] - peak_idx_pri) / (sigma_bins * 1.0)) ** 2)

    # --- ASYMMETRIC WINDOW FOR SECONDARY BOW ---
    window_sec = np.zeros_like(theta_rad)
    # 1. Shadow side (Alexander's band): Keep blur at 100%
    mask_sec_shadow = (idx <= mid_idx) & (idx >= peak_idx_sec)
    window_sec[mask_sec_shadow] = 1.0
    # 2. Supernumerary side: Fade out the blur to protect fringes
    mask_sec_super = idx < peak_idx_sec
    window_sec[mask_sec_super] = np.exp(-0.5 * ((idx[mask_sec_super] - peak_idx_sec) / (sigma_bins * 2.0)) ** 2)

    final_intensity = intensity_array.copy()

    # Apply the asymmetric blends
    final_intensity = (final_intensity * (1.0 - window_pri)) + (smoothed_pri * window_pri)
    final_intensity = (final_intensity * (1.0 - window_sec)) + (smoothed_sec * window_sec)

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
    elif isinstance(param, CuboctaSize):
        a = param.a_axis_um
        S_total = (12.0 + 4.0 * np.sqrt(3.0)) * (a ** 2)
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

    active_engine = backend
    if type(backend).__name__ == "HybridBackend":
        if current_size.r_um < backend.radius_threshold_um:
            active_engine = backend.low_backend
        else:
            active_engine = backend.high_backend

    # Run the type check on the unpacked engine
    if type(active_engine).__name__ == "DrJitRaytracerBackend" and type(current_size).__name__ == "SphereSize":
        r_eff = current_size.r_um
        if config.num_phi_bins > 1:
            for p in range(config.num_phi_bins):
                intensity_linear[p, :] = apply_caustic_diffraction_blur(intensity_linear[p, :], theta_linear, r_eff,
                                                                        w_nm)
        else:
            intensity_linear = apply_caustic_diffraction_blur(intensity_linear, theta_linear, r_eff, w_nm)

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
    elif isinstance(config, CuboctahedralConfig):
        a_ax = habit_params["a_axis"]
        print(f"[Gen] Mode: CUBOCTAHEDRON (a={a_ax}um)")
        size_params = [CuboctaSize(a_axis_um=a_ax)]
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


def save_binary_file(filename, phase_table, mu_vals, wavelengths, config: BaseParticleConfig, hg_weight: float,
                     g_val: float):
    if config.num_phi_bins > 1:
        phase_interleaved = np.moveaxis(phase_table, 0, -1)
        phase_interleaved = np.swapaxes(phase_interleaved, 0, 1)
    else:
        phase_interleaved = phase_table.T

    data_flat = phase_interleaved.flatten().astype(np.float32)

    with open(filename, "wb") as f:
        f.write(b"ATMPHASE")
        f.write(struct.pack("<I", 3))  # Version bumped to 3!
        f.write(struct.pack("<I", config.num_angles))
        f.write(struct.pack("<I", config.num_phi_bins))
        f.write(struct.pack("<I", len(wavelengths)))
        f.write(struct.pack("<f", float(wavelengths[0])))
        f.write(struct.pack("<f", float(wavelengths[-1])))

        f.write(struct.pack("<f", float(hg_weight)))
        f.write(struct.pack("<f", float(g_val)))

        f.write(data_flat.tobytes())

    print(f"[GEN] Saved binary v3: {os.path.basename(filename)}")
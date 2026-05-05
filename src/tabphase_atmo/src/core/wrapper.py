import os
import numpy as np

from filters import apply_polydispersity_filter
from utils.wrapper_math import compute_optical_weight, normalize_macroscopic_phase, apply_similarity_truncation
from .generator.backends import (
    MiePythonBackend,
    MieReferenceBackend,
    DrJitRaytracerBackend,
    HybridBackend,
)
from .generator.generate import generate_phase_table, save_binary_file
from .generator.visualize import visualize_anisotropic

from .config import BaseParticleConfig, DropletConfig, HexagonalConfig, BackendType

NUM_BATCHES=5
GRID_RESOLUTION=600

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _get_base_lut_info(config: BaseParticleConfig) -> str:
    base = f"{config.num_angles}ang-{config.num_phi_bins}az-{config.num_wavelengths}wl"
    if isinstance(config, HexagonalConfig):
        base += f"-{int(config.sun_elevation_deg)}sun"
        base += f"-{int(config.air_turbulence_factor*100)}turb"
    return base


def _generate_raw_filename(config: BaseParticleConfig, comp_item) -> str:
    lut_info = _get_base_lut_info(config)
    if isinstance(config, DropletConfig):
        shape, _, r, _ = comp_item
        return f"droplet_{lut_info}_{shape.value[:4]}{int(r)}um_raw"
    else:
        habit, _, c, a, _ = comp_item
        return f"hex_{lut_info}_{habit.value[:4]}{int(c)}c{int(a)}a_raw"


def _generate_mix_filename(config: BaseParticleConfig) -> str:
    lut_info = _get_base_lut_info(config)
    if isinstance(config, DropletConfig):
        comp_strs = [f"{shape.value[:4]}{int(r)}um-{int(w * 100)}w-{int(v * 1000)}pvar"
                     for shape, w, r, v in config.composition]
    else:
        comp_strs = [f"{habit.value[:4]}{int(c)}c{int(a)}a-{int(w * 100)}w-{int(v * 1000)}pvar"
                     for habit, w, c, a, v in config.composition]

    return f"{'droplet' if isinstance(config, DropletConfig) else 'hex'}_{lut_info}_" + "_".join(comp_strs)


def _get_paths(config: BaseParticleConfig, cache_dir: str = "cache"):
    root = os.path.join(cache_dir, config.backend.name.lower())
    _ensure_dir(root)

    raw_dir = os.path.join(root, "raw_habits")
    _ensure_dir(raw_dir)

    final_filename = _generate_mix_filename(config)
    bin_path = os.path.join(root, f"{final_filename}.bin")

    polar_dir = os.path.join(root, "polars")
    _ensure_dir(polar_dir)
    polar_path = os.path.join(polar_dir, f"{final_filename}.png")

    return raw_dir, bin_path, polar_path


def _resolve_backend(config: BaseParticleConfig, specific_habit: str = "sphere"):
    sun_elev = getattr(config, 'sun_elevation_deg', 0.0)
    turbulence = getattr(config, 'air_turbulence_factor', 1.0)

    if config.backend == BackendType.AUTO:
        return HybridBackend(
            MiePythonBackend(),
            DrJitRaytracerBackend(grid_res=GRID_RESOLUTION, num_batches=NUM_BATCHES, num_phi_bins=config.num_phi_bins,
                                  particle_shape=specific_habit, sun_elevation_deg=sun_elev),
            x0=800.0, x1=1400.0
        )
    if config.backend == BackendType.DRJIT:
        return DrJitRaytracerBackend(
            grid_res=GRID_RESOLUTION, num_batches=NUM_BATCHES, num_phi_bins=config.num_phi_bins,
            particle_shape=specific_habit, sun_elevation_deg=sun_elev, air_turbulence_factor=turbulence
        )
    if config.backend == BackendType.MIEPYTHON:
        return MiePythonBackend()
    if config.backend == BackendType.REFERENCE:
        return MieReferenceBackend()
    raise ValueError(f"Unknown backend enum '{config.backend}'.")


def _resolve_raw_habit(config: BaseParticleConfig, comp_item, raw_dir: str, force_regen: bool):
    """Fetches raw energy arrays from cache, or generates them if missing."""
    raw_filename = _generate_raw_filename(config, comp_item) + ".npz"
    raw_path = os.path.join(raw_dir, raw_filename)

    if not force_regen and os.path.exists(raw_path):
        print(f"[Main] Habit found in raw cache, loading {raw_filename}...")
        loaded = np.load(raw_path)
        return loaded['phase'], loaded['mu'], loaded['wl']

    shape_str = comp_item[0].value
    print(f"[Main] Baking Habit: {shape_str}, {raw_filename}...")

    if isinstance(config, DropletConfig):
        habit_params = {"radius": comp_item[2], "variance": comp_item[3]}
    elif isinstance(config, HexagonalConfig):
        habit_params = {
            "c_axis": comp_item[2], "a_axis": comp_item[3],
            "variance": comp_item[4], "sun_elevation_deg": config.sun_elevation_deg
        }
    else:
        raise ValueError("[Main] Unknown config type for habit generation.")

    be = _resolve_backend(config, specific_habit=shape_str)
    shape_table, shape_mu, shape_wl = generate_phase_table(config, backend=be, habit_params=habit_params)

    np.savez_compressed(raw_path, phase=shape_table, mu=shape_mu, wl=shape_wl)
    return shape_table, shape_mu, shape_wl


def create_atmospheric_phase(config: BaseParticleConfig, up_vector: tuple = (0.0, 0.0, 1.0),
                             force_regen=False,threshold = 10.0, generate_polar=True, cache_dir="cache"):
    raw_dir, bin_path, polar_path = _get_paths(config, cache_dir)

    # 1. Final Output Cache Check
    if not force_regen and os.path.exists(bin_path):
        print(f"[Main] Final Mix Cache Hit: {os.path.basename(bin_path)}")
        if generate_polar and not os.path.exists(polar_path):
            visualize_anisotropic(bin_path)
            pass
        return {"type": "atmosphericphase", "filename": bin_path, "up": up_vector}#,"forward_scatter_limit": forward_peak_limit}

    print(f"[Main] Assembling new Mix: {os.path.basename(bin_path)}...")
    master_phase_table, master_mu, master_wl = None, None, None

    for comp_item in config.composition:
        variance = comp_item[3] if isinstance(config, DropletConfig) else comp_item[4]
        optical_weight = compute_optical_weight(config, comp_item)

        shape_table, shape_mu, shape_wl = _resolve_raw_habit(config, comp_item, raw_dir, force_regen)

        if variance >= 0.001:
            print(f"[Main] Applying polydispersity sim (variance={variance:.2f})...")
            shape_table = apply_polydispersity_filter(shape_table, config.num_angles, variance)

        print(f"[Main] Blending habit with Optical Weight: {optical_weight:.4f}")
        if master_phase_table is None:
            master_phase_table = shape_table * optical_weight
            master_mu = shape_mu
            master_wl = shape_wl
        else:
            master_phase_table += shape_table * optical_weight

    # 3. Finalize and Apply Similarity Theory (Phase Truncation)
    print(f"[Main] Applying Similarity Theory (Truncating at 1.0)...")
    master_phase_table, f_fractions = apply_similarity_truncation(
        master_phase_table, master_mu, config.num_phi_bins, threshold=threshold
    )

    hg_weight_val = float(np.mean(f_fractions))
    g_val = 0.877

    print(f"[Main] Missing Energy Fraction (f): {hg_weight_val:.4f}")
    print(f"[Main] Saving Master Mix...")

    # Pass the new variables to the exporter
    save_binary_file(bin_path, master_phase_table, master_mu, master_wl, config, hg_weight_val, g_val)

    if generate_polar:
        try:
            visualize_anisotropic(bin_path)
            print(f"[Main] Polar plot saved to {polar_path}")
        except Exception as e:
            print(f"[Main] Polar plot generation failed: {e}")

    # Return exactly the same dict for both cache hits and fresh bakes
    return {
        "type": "atmosphericphase",
        "filename": bin_path,
        "up": up_vector
    }
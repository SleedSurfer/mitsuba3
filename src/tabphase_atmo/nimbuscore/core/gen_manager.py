import os
import numpy as np

from .filters import apply_polydispersity_filter
from .utils.wrapper_math import compute_optical_weight, normalize_macroscopic_phase, apply_similarity_truncation
from .generator.backends import (
    MiePythonBackend,
    MieReferenceBackend,
    DrJitRaytracerBackend,
    HybridBackend,
)
from .generator.generate import generate_phase_table, save_binary_file
from .generator.visualize import visualize_anisotropic
from pathlib import Path
from .config import BaseParticleConfig, DropletConfig, HexagonalConfig, BackendType, CuboctahedralConfig

NUM_BATCHES = 15
GRID_RESOLUTION = 300

def get_global_cache_dir() -> str:
    """
    Default cache location is in the home/.nimbus
    """
    cache_path = Path.home() / ".nimbus" / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path)


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _get_base_lut_info(config) -> str:
    base = f"{config.num_angles}ang-{config.num_phi_bins}az-{config.num_wavelengths}wl"
    if getattr(config, 'sun_elevation_deg', None) is not None:
        base += f"-{int(config.sun_elevation_deg)}sun"
    if getattr(config, 'air_turbulence_factor', None) is not None:
        base += f"-{int(config.air_turbulence_factor * 100)}turb"
    return base


def _generate_raw_filename(config, comp_item) -> str:
    lut_info = _get_base_lut_info(config)
    cfg_type = config.__class__.__name__

    if cfg_type == "DropletConfig":
        return f"droplet_{lut_info}_{comp_item.shape.value[:4]}{int(comp_item.radius_mean_um)}um_raw"

    elif cfg_type == "CuboctahedralConfig":
        return f"cubo_{lut_info}_{comp_item.habit.value[:4]}{int(comp_item.a_axis_um)}a_raw"

    elif cfg_type == "HexagonalConfig":
        return f"hex_{lut_info}_{comp_item.habit.value[:4]}{int(comp_item.length_axis_um)}c{int(comp_item.width_axis_um)}a_raw"

    raise ValueError(f"Unknown config type for filename generation: {cfg_type}")


def _generate_mix_filename(config) -> str:
    lut_info = _get_base_lut_info(config)
    cfg_type = config.__class__.__name__

    if cfg_type == "DropletConfig":
        comp_strs = [f"{c.shape.value[:4]}{int(c.radius_mean_um)}um-{int(c.weight * 100)}w-{int(c.variance * 1000)}pvar"
                     for c in config.composition]
        prefix = 'droplet'

    elif cfg_type == "CuboctahedralConfig":
        comp_strs = [f"{c.habit.value[:4]}{int(c.a_axis_um)}a-{int(c.weight * 100)}w-{int(c.variance * 1000)}pvar"
                     for c in config.composition]
        prefix = 'cubo'

    elif cfg_type == "HexagonalConfig":
        comp_strs = [
            f"{c.habit.value[:4]}{int(c.length_axis_um)}c{int(c.width_axis_um)}a-{int(c.weight * 100)}w-{int(c.variance * 1000)}pvar"
            for c in config.composition]
        prefix = 'hex'

    else:
        raise ValueError(f"Unknown config type for mix filename: {cfg_type} (Full type: {type(config)})")

    return f"{prefix}_{lut_info}_" + "_".join(comp_strs)


def _get_paths(config: BaseParticleConfig, cache_dir: str = None):
    # Dynamic fallback evaluation babyyyyy
    if cache_dir is None:
        cache_dir = get_global_cache_dir()

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
            radius_threshold_um=500
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

    # Extract shape string safely based on type
    shape_str = comp_item.shape.value if isinstance(config, DropletConfig) else comp_item.habit.value
    print(f"[Main] Baking Habit: {shape_str}, {raw_filename}...")

    # Pack parameters for the backend generator
    if isinstance(config, DropletConfig):
        habit_params = {
            "radius": comp_item.radius_mean_um,
            "variance": comp_item.variance
        }
    elif isinstance(config, CuboctahedralConfig):
        habit_params = {
            "a_axis": comp_item.a_axis_um,
            "variance": comp_item.variance,
            "sun_elevation_deg": getattr(config, 'sun_elevation_deg', 0.0)
        }
    elif isinstance(config, HexagonalConfig):
        habit_params = {
            "c_axis": comp_item.length_axis_um,
            "a_axis": comp_item.width_axis_um,
            "variance": comp_item.variance,
            "sun_elevation_deg": getattr(config, 'sun_elevation_deg', 0.0)
        }
    else:
        raise ValueError("[Main] Unknown config type for habit generation.")

    be = _resolve_backend(config, specific_habit=shape_str)
    shape_table, shape_mu, shape_wl = generate_phase_table(config, backend=be, habit_params=habit_params)

    np.savez_compressed(raw_path, phase=shape_table, mu=shape_mu, wl=shape_wl)
    return shape_table, shape_mu, shape_wl


def create_atmospheric_phase(config: BaseParticleConfig, up_vector: tuple = (0.0, 0.0, 1.0),
                             force_regen=False, threshold=10.0, generate_polar=True, cache_dir=None,
                             masochist_mode=False):
    raw_dir, bin_path, polar_path = _get_paths(config, cache_dir)

    # 1. Final Output Cache Check
    if not force_regen and os.path.exists(bin_path):
        print(f"[Main] Final Mix Cache Hit: {os.path.basename(bin_path)}")
        if generate_polar and not os.path.exists(polar_path):
            try:
                visualize_anisotropic(bin_path)
            except Exception as e:
                print(f"[Main] Failed to generate missing polar: {e}")
        return {"type": "atmosphericphase", "filename": bin_path, "up": up_vector}

    print(f"[Main] Assembling new Mix: {os.path.basename(bin_path)}...")
    master_phase_table, master_mu, master_wl = None, None, None

    for comp_item in config.composition:
        variance = comp_item.variance
        optical_weight = compute_optical_weight(config, comp_item)

        if masochist_mode and variance >= 0.001:
            if not isinstance(config, DropletConfig):
                print(f"\n[Main] MASOCHIST MODE DENIED ")
                print(f"[Main] Nice try. Geometric raytracing for crystals is scale-invariant.")
                print(f"[Main] Integrating 21 sizes of a prism just stacks the exact same phase table 21 times.")
                print(f"[Main] Falling back to a standard single-bake for this crystal...")
                shape_table, shape_mu, shape_wl = _resolve_raw_habit(config, comp_item, raw_dir, force_regen)
            else:
                print(f"\n[Main] MASOCHIST MODE ENGAGED")
                print(
                    f"[Main] Bypassing convolution blur. Physically integrating 21 discrete Droplet sizes for variance={variance:.3f}...")

                base_sz = comp_item.radius_mean_um

                # Math for Log-Normal Distribution
                sigma = np.sqrt(variance)
                mu = np.log(base_sz) - 0.5 * sigma ** 2

                # Generate 21 discrete steps from -2.5 sigma to +2.5 sigma
                min_sz = np.exp(mu - 2.5 * sigma)
                max_sz = np.exp(mu + 2.5 * sigma)
                size_steps = np.linspace(min_sz, max_sz, 21)

                # 1. Calculate Number Density (How many droplets actually exist at this size)
                number_weights = (1.0 / (size_steps * sigma * np.sqrt(2 * np.pi))) * np.exp(
                    - (np.log(size_steps) - mu) ** 2 / (2 * sigma ** 2))

                # 2. Convert to Optical Weights (Multiply by cross-sectional area scattering factor)
                optical_weights = number_weights * (size_steps ** 2)

                # 3. Normalize so the total energy sums exactly to 1.0
                optical_weights /= np.sum(optical_weights)

                integrated_table = None
                for i, (step_sz, w) in enumerate(zip(size_steps, optical_weights)):
                    print(f"[Main BRUTE] Baking sphere slice {i + 1}/21: radius={step_sz:.2f}um (weight={w:.4f})")

                    temp_comp = comp_item._replace(radius_mean_um=step_sz, variance=0.0)

                    # Fire the engine for this specific size
                    step_table, shape_mu, shape_wl = _resolve_raw_habit(config, temp_comp, raw_dir, force_regen)

                    if integrated_table is None:
                        integrated_table = step_table * w
                    else:
                        integrated_table += step_table * w

                shape_table = integrated_table
                print("[Main] Physical droplet integration complete. Your CPU deserves a written apology.")

        else:
            # --- NORMAL SANE MODE ---
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
    print(f"[Main] Applying Similarity Theory (Truncating at {threshold})...")
    master_phase_table, f_fractions = apply_similarity_truncation(
        master_phase_table, master_mu, config.num_phi_bins, threshold=threshold
    )

    hg_weight_val = float(np.mean(f_fractions))
    g_val = 0.877  # Assuming you compute this properly elsewhere later

    print(f"[Main] Missing Energy Fraction (f): {hg_weight_val:.4f}")
    print(f"[Main] Saving Master Mix...")

    save_binary_file(bin_path, master_phase_table, master_mu, master_wl, config, hg_weight_val, g_val)

    if generate_polar:
        try:
            visualize_anisotropic(bin_path)
            print(f"[Main] Polar plot saved to {polar_path}")
        except Exception as e:
            print(f"[Main] Polar plot generation failed: {e}")

    return {
        "type": "atmosphericphase",
        "filename": bin_path,
        "up": up_vector
    }


def generate_mitsuba_xml(phase_dict: dict) -> str:
    """
    Translates the python dictionary output of create_atmospheric_phase
    into a valid Mitsuba 3 XML snippet.
    """
    phase_type = phase_dict.get("type", "atmosphericphase")
    filename = phase_dict.get("filename", "")
    up = phase_dict.get("up", (0.0, 0.0, 1.0))

    xml = (
        f'<phase type="{phase_type}">\n'
        f'    <string name="filename" value="{filename}"/>\n'
        f'    <vector name="up" x="{up[0]}" y="{up[1]}" z="{up[2]}"/>\n'
        f'</phase>'
    )
    return xml
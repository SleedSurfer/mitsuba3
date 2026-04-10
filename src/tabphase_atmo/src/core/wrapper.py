import os

from .generator.backends import (
    MiePythonBackend,
    MieReferenceBackend,
    DrJitRaytracerBackend,
    HybridBackend,
)
from .generator.generate import generate_phase_table, save_binary_file
from .generator.visualize import visualize_anisotropic

from .config import BaseParticleConfig, DropletConfig, HexagonalConfig, BackendType


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _generate_cache_filename(config: BaseParticleConfig) -> str:
    mid_ior = config.computed_iors[len(config.computed_iors) // 2] if config.computed_iors else 1.31
    ior_str = f"ior{mid_ior:.3f}"

    if isinstance(config, DropletConfig):
        comp_strs = [f"{shape.value[:4]}{weight:.2f}_r{r}v{v}" for shape, weight, r, v in config.composition]
        comp_joined = "_".join(comp_strs)
        return f"droplet_{comp_joined}_{ior_str}"

    elif isinstance(config, HexagonalConfig):
        comp_strs = [f"{habit.value[:4]}{weight:.2f}_c{c}a{a}" for habit, weight, c, a in config.composition]
        comp_joined = "_".join(comp_strs)
        return f"hex_{comp_joined}_{ior_str}"
    else:
        raise ValueError("Unknown config class passed to cache generator.")


def _get_paths(config: BaseParticleConfig, cache_dir: str = "cache"):
    root = os.path.join(cache_dir, config.backend.name.lower())
    _ensure_dir(root)
    filename = _generate_cache_filename(config)
    bin_path = os.path.join(root, f"{filename}.bin")

    heat_dir = os.path.join(root, "heatmaps")
    _ensure_dir(heat_dir)
    heat_path = os.path.join(heat_dir, f"{filename}.png")

    polar_dir = os.path.join(root, "polars")
    _ensure_dir(polar_dir)
    polar_path = os.path.join(polar_dir, f"{filename}.png")
    return bin_path, heat_path, polar_path


def _resolve_backend(config: BaseParticleConfig, specific_habit: str = "sphere"):
    # Extract the sun elevation if it exists (defaults to 0.0 for droplets/spheres)
    sun_elev = getattr(config, 'sun_elevation_deg', 0.0)

    if config.backend == BackendType.AUTO:
        return HybridBackend(
            MiePythonBackend(),
            DrJitRaytracerBackend(grid_res=600, num_batches=4, num_phi_bins=config.num_phi_bins,
                                  particle_shape=specific_habit, sun_elevation_deg=sun_elev),
            x0=800.0, x1=1400.0
        )
    if config.backend == BackendType.DRJIT:
        return DrJitRaytracerBackend(
            grid_res=500, num_batches=20, num_phi_bins=config.num_phi_bins,
            particle_shape=specific_habit, sun_elevation_deg=sun_elev
        )
    if config.backend == BackendType.MIEPYTHON:
        return MiePythonBackend()
    if config.backend == BackendType.REFERENCE:
        return MieReferenceBackend()
    raise ValueError(f"Unknown backend enum '{config.backend}'.")


def create_atmospheric_phase(config: BaseParticleConfig, up_vector: tuple = (0.0, 1.0, 0.0),
                             force_regen=False, generate_heatmap=True, generate_polar=True, cache_dir="cache"):
    file_path, heatmap_path, polar_path = _get_paths(config, cache_dir)

    if not force_regen and os.path.exists(file_path):
        print(f"[Wrapper] Cache Hit: {file_path}. We take those.")
        if generate_polar and not os.path.exists(polar_path):
            try:
                visualize_anisotropic(file_path, polar_path)
            except Exception as e:
                print(f"[Wrapper] Polar plot regen failed: {e}")
        return {"type": "atmosphericphase", "filename": file_path, "up": up_vector}

    print(f"[Wrapper] Generating new LUT: {os.path.basename(file_path)}...")

    try:
        phase_table = None
        mu = None
        wavelengths = None

        for comp_item in config.composition:
            if isinstance(config, DropletConfig):
                shape_enum, weight, radius, variance = comp_item
                shape_str = shape_enum.value
                habit_params = {"radius": radius, "variance": variance}
            else:
                shape_enum, weight, c_axis, a_axis = comp_item
                shape_str = shape_enum.value
                habit_params = {"c_axis": c_axis, "a_axis": a_axis, "sun_elevation_deg": config.sun_elevation_deg}

            print(f"          -> Cooking shape: {shape_str} (Weight: {weight * 100:.1f}%) | Params: {habit_params}")

            be = _resolve_backend(config, specific_habit=shape_str)
            shape_table, shape_mu, shape_wl = generate_phase_table(config, backend=be, habit_params=habit_params)

            if phase_table is None:
                phase_table = shape_table * weight
                mu = shape_mu
                wavelengths = shape_wl
            else:
                phase_table += shape_table * weight

        save_binary_file(file_path, phase_table, mu, wavelengths, config)

        if generate_polar:
            visualize_anisotropic(file_path, polar_path)

    except Exception as e:
        print(f"[Wrapper] GENERATION FAILED: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e

    return {"type": "atmosphericphase", "filename": file_path, "up": up_vector}
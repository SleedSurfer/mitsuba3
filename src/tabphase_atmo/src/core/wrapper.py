import os
import numpy as np

from .generator.backends import (
    MiePythonBackend,
    MieReferenceBackend,
    DrJitRaytracerBackend,
    HybridBackend,
)
from .generator.generate import generate_phase_table, save_binary_file
from .generator.visualize import visualize_binary_file, visualize_polar_plot

from .config import MieConfig

DRJIT_GRID_SIZE = 3000


def _size_parameter(radius_um: float, wavelength_nm: float) -> float:
    wavelength_um = wavelength_nm / 1000.0
    return float(2.0 * np.pi * radius_um / max(wavelength_um, 1e-9))


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _resolve_backend(name: str):
    """
    Resolves a string name to a specific backend instance.
    'Auto' returns None to signal dynamic selection later.
    """
    name = (name or "auto").lower().strip()

    if name in ("auto",):
        return None

    if name in ("mie", "miepython"):
        return MiePythonBackend()
    if name in ("mie_ref", "reference"):
        return MieReferenceBackend()

    if name in ("jit_traced", "drjit", "raytracer"):
        return DrJitRaytracerBackend(grid_res=DRJIT_GRID_SIZE,particle_shape="sphere")  # Default sensible ray count
    if name in ("hybrid",):
        return HybridBackend(MiePythonBackend(), DrJitRaytracerBackend(grid_res=DRJIT_GRID_SIZE,particle_shape="sphere"))

    raise ValueError(f"Unknown backend '{name}'. Supported: auto, mie, jit_traced, go_airy, hybrid")


def _get_backend_cache_root(backend_name: str, cache_dir: str = "cache") -> str:
    root = os.path.join(cache_dir, backend_name)
    _ensure_dir(root)
    return root


def _get_paths(config: MieConfig, backend_name: str, cache_dir: str = "cache"):
    """
    Centralized path generation to stop repeating os.path.join
    """
    root = _get_backend_cache_root(backend_name, cache_dir=cache_dir)

    bin_path = os.path.join(root, f"{config.output_filename}.bin")

    heat_dir = os.path.join(root, "heatmaps")
    _ensure_dir(heat_dir)
    heat_path = os.path.join(heat_dir, f"{config.output_filename}.png")

    polar_dir = os.path.join(root, "polars")
    _ensure_dir(polar_dir)
    polar_path = os.path.join(polar_dir, f"{config.output_filename}.png")

    return bin_path, heat_path, polar_path


def create_atmospheric_phase(
        config: MieConfig,
        force_regen=False,
        generate_heatmap=True,
        generate_polar=True,
        backend="auto",
        cache_dir="cache",
        # Tuning params for the Hybrid switch
        x_mie_only=500.0,
        hybrid_x0=500.0,
        hybrid_x1=850.0,
        x_go_only=850.0
):
    be = _resolve_backend(backend)

    if be is None:  # "Auto" mode
        print(f"[Wrapper] Auto-select enabled. Using Hybrid (Mie + Raytracer) for safety.")
        be = HybridBackend(
            MiePythonBackend(),
            DrJitRaytracerBackend(grid_res=DRJIT_GRID_SIZE),
            x0=800.0,  # Transition point
            x1=1400.0  # Full Raytracer here
        )

    file_path, heatmap_path, polar_path = _get_paths(config, be.name, cache_dir)

    if force_regen or not os.path.exists(file_path):
        print(f"[Wrapper] Generating {config.output_filename}...")
        print(f"          Method: {be.name}")
        print(f"          Params: r={config.radius_mean_um}um, std={config.variance*100}%")

        try:
            phase_table, mu, wavelengths = generate_phase_table(config, backend=be)
            save_binary_file(file_path, phase_table, mu, wavelengths, config)

            if generate_heatmap:
                visualize_binary_file(file_path, heatmap_path)
            if generate_polar:
                visualize_polar_plot(file_path, polar_path)

        except Exception as e:
            print(f"[Wrapper] GENERATION FAILED: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    else:
        print(f"[Wrapper] Cache Hit: {file_path}")
        # Lazy regen of visualizations if missing
        if generate_heatmap and not os.path.exists(heatmap_path):
            print("[Wrapper] Regenerating missing heatmap...")
            try:
                visualize_binary_file(file_path, heatmap_path)
            except:
                pass

        if generate_polar and not os.path.exists(polar_path):
            print("[Wrapper] Regenerating missing polar plot...")
            try:
                visualize_polar_plot(file_path, polar_path)
            except:
                pass

    return {"type": "atmosphericphase", "filename": file_path}
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

from .config import MieConfig,BackendType

import os


DRJIT_GRID_SIZE = 3000


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _resolve_backend(backend_enum: BackendType):
    """
    Resolves the BackendType enum directly to an initialized backend instance.
    """
    if backend_enum == BackendType.AUTO:
        return None  # Let the wrapper decide the default fallback

    if backend_enum == BackendType.MIEPYTHON:
        return MiePythonBackend()

    if backend_enum == BackendType.REFERENCE:
        return MieReferenceBackend()

    if backend_enum == BackendType.DRJIT:
        return DrJitRaytracerBackend(grid_res=DRJIT_GRID_SIZE, particle_shape="sphere")

    if backend_enum == BackendType.HYBRID:
        return HybridBackend(
            MiePythonBackend(),
            DrJitRaytracerBackend(grid_res=DRJIT_GRID_SIZE, particle_shape="sphere")
        )

    raise ValueError(f"Unknown backend enum '{backend_enum}'.")


def _get_backend_cache_root(backend_enum: BackendType, cache_dir: str = "cache") -> str:
    # Folders are now strictly named after the enum
    root = os.path.join(cache_dir, backend_enum.name.lower())
    _ensure_dir(root)
    return root


def _get_paths(config: MieConfig, cache_dir: str = "cache"):
    """
    Path generation is now totally driven by the config state. No loose arguments.
    """
    root = _get_backend_cache_root(config.backend, cache_dir=cache_dir)

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
        cache_dir="cache",
        # Tuning params for the Hybrid switch
        x_mie_only=500.0,
        hybrid_x0=500.0,
        hybrid_x1=850.0,
        x_go_only=850.0
):
    # The config is the captain now.
    be = _resolve_backend(config.backend)
    print(f"[Wrapper] Selected backend: {config.backend.name} -> {be.name if be else 'Auto (Hybrid Fallback)'}")
    if be is None:  # Auto mode fallback
        print(f"[Wrapper] Auto-select enabled. Defaulting to Hybrid (Mie + Raytracer) to stay safe.")
        be = HybridBackend(
            MiePythonBackend(),
            DrJitRaytracerBackend(grid_res=DRJIT_GRID_SIZE),
            x0=800.0,
            x1=1400.0
        )

    # Get paths deterministically from the config
    file_path, heatmap_path, polar_path = _get_paths(config, cache_dir)

    if force_regen or not os.path.exists(file_path):
        print(f"[Wrapper] Generating {config.output_filename}...")
        print(f"          Method: {be.name}")
        print(f"          Params: r={config.radius_mean_um}um, std={config.variance * 100}%")

        try:
            phase_table, mu, wavelengths = generate_phase_table(config, backend=be)
            save_binary_file(file_path, phase_table, mu, wavelengths, config)

            if generate_heatmap:
                visualize_binary_file(file_path, heatmap_path)
            if generate_polar:
                visualize_polar_plot(file_path, polar_path)

        except Exception as e:
            # Clean up the corrupted/half-written garbage so next run isn't fucked
            print(f"[Wrapper] GENERATION FAILED: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    else:
        print(f"[Wrapper] Cache Hit: {file_path}. We take those.")

        # Lazy regen of missing visualizers
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
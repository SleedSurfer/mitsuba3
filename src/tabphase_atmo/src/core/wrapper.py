import os
import numpy as np

from .generator.backends import (
    MiePythonBackend,
    MieReferenceBackend,
    DrJitRaytracerBackend,
    HybridBackend,
)
from .generator.generate import generate_phase_table, save_binary_file
from .generator.visualize import visualize_anisotropic

from src.core.config import MieConfig, BackendType

DRJIT_GRID_SIZE = 950


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _resolve_backend(config: MieConfig):
    """
    Resolves the BackendType enum directly to an initialized backend instance.
    """
    backend_enum = config.backend

    if backend_enum == BackendType.AUTO:
        print("[Backend] Auto-select enabled. Defaulting to Hybrid (Mie + Raytracer).")
        return HybridBackend(
            MiePythonBackend(),
            DrJitRaytracerBackend(grid_res=950, num_batches=4, num_phi_bins=config.num_phi_bins,
                                  particle_shape=config.shape.name.lower()),
            x0=800.0,
            x1=1400.0
        )

    if backend_enum == BackendType.MIEPYTHON:
        return MiePythonBackend()

    if backend_enum == BackendType.REFERENCE:
        return MieReferenceBackend()

    if backend_enum == BackendType.DRJIT:
        # If you are forcing an oblate test via notes, you can hardcode "oblate" here for testing
       # shape_str = "oblate" if "oblate" in config.note else config.shape.name.lower()
        shape_str = config.shape.name.lower()
        print(f"[Backend] Dr.Jit Raytracer selected with shape '{shape_str}' and {config.num_phi_bins} phi bins.")
        return DrJitRaytracerBackend(
            grid_res=500,
            num_batches=40,
            num_phi_bins=config.num_phi_bins,
            particle_shape=shape_str
        )

    if backend_enum == BackendType.HYBRID:
        return HybridBackend(
            MiePythonBackend(),
            DrJitRaytracerBackend(grid_res=DRJIT_GRID_SIZE, particle_shape="sphere")
        )

    # If it's none of the above, throw hands immediately.
    raise ValueError(f"Unknown backend enum '{backend_enum}'.")


def _get_backend_cache_root(backend_enum: BackendType, cache_dir: str = "cache") -> str:
    root = os.path.join(cache_dir, backend_enum.name.lower())
    _ensure_dir(root)
    return root


def _get_paths(config: MieConfig, cache_dir: str = "cache"):
    """
    Path generation is now totally driven by the config state.
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
        up_vector: tuple = (0.0, 1.0, 0.0),  # <--- Added gravity alignment
        force_regen=False,
        generate_heatmap=True,
        generate_polar=True,
        cache_dir="cache"):
    be = _resolve_backend(config)

    # Failsafe: if we somehow get a null backend, crash loudly.
    if be is None:
        raise ValueError("Backend resolution failed completely. Check your config enum and resolver.")

    print(f"[Wrapper] Selected backend: {config.backend.name} -> {be.name}")

    file_path, heatmap_path, polar_path = _get_paths(config, cache_dir)

    if force_regen or not os.path.exists(file_path):
        print(f"[Wrapper] Generating {config.output_filename}...")
        print(f"          Method: {be.name}")
        print(f"          Params: r={config.radius_mean_um}um, std={config.variance * 100}%")

        try:
            phase_table, mu, wavelengths = generate_phase_table(config, backend=be)
            save_binary_file(file_path, phase_table, mu, wavelengths, config)

            if generate_polar:
                visualize_anisotropic(file_path, polar_path)

        except Exception as e:
            # Clean up the corrupted/half-written garbage so next run isn't fucked
            print(f"[Wrapper] GENERATION FAILED: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    else:
        print(f"[Wrapper] Cache Hit: {file_path}. We take those.")

        # Lazy regen of missing visualizer
        if generate_polar and not os.path.exists(polar_path):
            print("[Wrapper] Regenerating missing polar plot...")
            try:
                visualize_anisotropic(file_path, polar_path)
            except Exception as e:
                print(f"[Wrapper] Polar plot regen failed: {e}")

    # The dictionary now directly maps the Python tuple to the C++ 'up' Vector3f property
    return {
        "type": "atmosphericphase",
        "filename": file_path,
        "up": up_vector
    }
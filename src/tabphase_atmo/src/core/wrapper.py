import os
import numpy as np

# Use package-relative imports
from .generator.backends import (
    MiePythonBackend,
    MieReferenceBackend,
    DrJitRaytracerBackend,  # <--- NEW HOTNESS
    HybridBackend,
)
from .generator.generate import generate_phase_table, save_binary_file
from .generator.visualize import visualize_binary_file, visualize_polar_plot

from .config import MieConfig


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

    # Mie (Small / Exact)
    if name in ("mie", "miepython"):
        return MiePythonBackend()
    if name in ("mie_ref", "reference"):
        return MieReferenceBackend()

    # Geometric Optics (Large / Fast)
    if name in ("jit_traced", "drjit", "raytracer"):
        return DrJitRaytracerBackend(num_rays=10_000_000)  # Default sensible ray count
    # Hybrid (The Best of Both Worlds)
    if name in ("hybrid",):
        # Default hybrid config; usually overridden by auto logic
        return HybridBackend(MiePythonBackend(), DrJitRaytracerBackend())

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

    # Visualizations live in subfolders
    heat_dir = os.path.join(root, "heatmaps")
    _ensure_dir(heat_dir)
    heat_path = os.path.join(heat_dir, f"{config.output_filename}.png")

    polar_dir = os.path.join(root, "polars")
    _ensure_dir(polar_dir)
    polar_path = os.path.join(polar_dir, f"{config.output_filename}.png")

    return bin_path, heat_path, polar_path


def create_atmospheric_phase(
        radius_mean_um=2.0,
        radius_std_um=0.5,
        num_angles=4096,
        num_wavelengths=64,
        note="mist",
        force_regen=False,
        generate_heatmap=True,
        generate_polar=True,
        backend="auto",  # Default to smart selection
        cache_dir="cache",
        # Tuning params for the Hybrid switch
        x_mie_only=700.0,
        x_go_only=1400.0,
        hybrid_x0=800.0,
        hybrid_x1=1400.0
):
    config = MieConfig(
        radius_mean_um=radius_mean_um,
        radius_std_um=radius_std_um,
        num_angles=num_angles,
        num_wavelengths=num_wavelengths,
        note=note,
    )

    # 1. Resolve Backend
    be = _resolve_backend(backend)

    # 2. Handle "Auto" Logic
    if be is None:
        lambda_mid = 0.5 * (config.min_wavelength + config.max_wavelength)
        x_mean = _size_parameter(config.radius_mean_um, lambda_mid)

        if x_mean <= x_mie_only:
            print(f"[Wrapper] Auto-select: Small drop (x={x_mean:.1f}) -> Mie")
            be = MiePythonBackend()
        elif x_mean >= x_go_only:
            print(f"[Wrapper] Auto-select: Large drop (x={x_mean:.1f}) -> Dr.Jit Raytracer")
            be = DrJitRaytracerBackend()
        else:
            print(f"[Wrapper] Auto-select: Transition zone (x={x_mean:.1f}) -> Hybrid (Mie + Raytracer)")
            # We blend Mie (Low) with Dr.Jit (High)
            be = HybridBackend(
                MiePythonBackend(),
                DrJitRaytracerBackend(),
                x0=hybrid_x0,
                x1=hybrid_x1
            )

    # 3. Pathing
    file_path, heatmap_path, polar_path = _get_paths(config, be.name, cache_dir)

    # 4. Generation / Caching
    if force_regen or not os.path.exists(file_path):
        print(f"[Wrapper] Generating {config.output_filename}...")
        print(f"          Method: {be.name}")
        print(f"          Params: r={radius_mean_um}um, std={radius_std_um}um")

        try:
            table = generate_phase_table(config, backend=be)
            save_binary_file(file_path, table, config)

            # Generate visualizations only on success
            if generate_heatmap:
                visualize_binary_file(file_path, heatmap_path)
            if generate_polar:
                visualize_polar_plot(file_path, polar_path)

        except Exception as e:
            print(f"[Wrapper] GENERATION FAILED: {e}")
            # Optional: Clean up partial file
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
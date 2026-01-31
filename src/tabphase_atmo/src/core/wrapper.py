import os
import numpy as np

# Use package-relative imports
from .generator.backends import (
    MiePythonBackend,
    MieReferenceBackend,
    GOAiryBackend,
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
    name = (name or "auto").lower()
    if name in ("mie", "miepython"):
        return MiePythonBackend()
    if name in ("mie_ref", "reference"):
        return MieReferenceBackend()
    if name in ("go", "go_airy", "large"):
        return GOAiryBackend()
    if name in ("hybrid",):
        return HybridBackend(MiePythonBackend(), GOAiryBackend())
    if name in ("auto",):
        return None
    raise ValueError("Unknown backend. Supported: auto, mie, mie_ref, go_airy, hybrid")


def _get_backend_cache_root(backend_name: str, cache_dir: str = "cache") -> str:
    """
    Returns cache/<backend_name> and ensures it exists.
    """
    root = os.path.join(cache_dir, backend_name)
    _ensure_dir(root)
    return root


def _get_cache_path(config: MieConfig, backend_name: str, cache_dir: str = "cache") -> str:
    """
    cache/<backend>/...bin
    """
    root = _get_backend_cache_root(backend_name, cache_dir=cache_dir)
    return os.path.join(root, f"{config.output_filename}.bin")


def _get_polar_path(config: MieConfig, backend_name: str, cache_dir: str = "cache") -> str:
    root = _get_backend_cache_root(backend_name, cache_dir=cache_dir)
    polar_dir = os.path.join(root, "polars")
    _ensure_dir(polar_dir)
    return os.path.join(polar_dir, f"{config.output_filename}.png")


def _get_heatmap_path(config: MieConfig, backend_name: str, cache_dir: str = "cache") -> str:
    """
    cache/<backend>/heatmaps/...png
    """
    root = _get_backend_cache_root(backend_name, cache_dir=cache_dir)
    heatmap_dir = os.path.join(root, "heatmaps")
    _ensure_dir(heatmap_dir)
    return os.path.join(heatmap_dir, f"{config.output_filename}.png")


def create_atmospheric_phase(
    radius_mean_um=2.0,
    radius_std_um=0.5,
    num_angles=4096,
    num_wavelengths=64,
    note="mist",
    force_regen=False,
    generate_heatmap=True,
    generate_polar=True,
    backend="mie",
    cache_dir="cache",
    x_mie_only=700.0, x_go_only=1600.0, hybrid_x0=800.0, hybrid_x1=1400.0
):
    config = MieConfig(
        radius_mean_um=radius_mean_um,
        radius_std_um=radius_std_um,
        num_angles=num_angles,
        num_wavelengths=num_wavelengths,
        note=note,
    )

    be = _resolve_backend(backend)
    if be is None:
        lambda_mid = 0.5 * (config.min_wavelength + config.max_wavelength)
        x_mean = _size_parameter(config.radius_mean_um, lambda_mid)

        if x_mean <= x_mie_only:
            be = MiePythonBackend()
        elif x_mean >= x_go_only:
            be = GOAiryBackend()
        else:
            be = HybridBackend(MiePythonBackend(), GOAiryBackend(), x0=hybrid_x0, x1=hybrid_x1)

        print(f"[Wrapper] Auto backend: x_mean={x_mean:.1f} -> {be.name}")

    be = _resolve_backend(backend)
    file_path = _get_cache_path(config, backend_name=be.name, cache_dir=cache_dir)
    heatmap_path = _get_heatmap_path(config, backend_name=be.name, cache_dir=cache_dir)
    polar_path = _get_polar_path(config, backend_name=be.name, cache_dir=cache_dir)

    if force_regen or not os.path.exists(file_path):
        print(f"[Wrapper] Cache miss! Generating LUT for {config.output_filename}...")
        print(f"          Backend: {be.name}")

        table = table = generate_phase_table(config, backend=be)
        save_binary_file(file_path, table, config)

        if generate_heatmap:
            try:
                visualize_binary_file(file_path, heatmap_path)
            except Exception as e:
                print(f"[Wrapper] Heatmap generation failed: {e}")

        if generate_polar:
            try:
                visualize_polar_plot(file_path, polar_path)
            except Exception as e:
                print(f"[Wrapper] Polar plot generation failed: {e}")
    else:
        print(f"[Wrapper] Found cached LUT: {file_path}")
        if generate_polar and not os.path.exists(polar_path):
            try:
                visualize_polar_plot(file_path, polar_path)
            except Exception as e:
                print(f"[Wrapper] Polar plot generation failed (cache-hit): {e}")

        if generate_heatmap and not os.path.exists(heatmap_path):
            try:
                visualize_binary_file(file_path, heatmap_path)
            except Exception as e:
                print(f"[Wrapper] Heatmap generation failed (cache-hit): {e}")

    return {"type": "atmosphericphase", "filename": file_path}
import os

from python.atmospheric.generator.generate import generate_mie_table, save_binary_file
from python.atmospheric.generator.visualize import visualize_binary_file
from .config import MieConfig


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _get_cache_path(config: MieConfig, cache_dir="cache"):
    """
    Determines the unique filename based on the config.
    """
    _ensure_dir(cache_dir)
    return os.path.join(cache_dir, f"{config.output_filename}.bin")


def _get_heatmap_path(config: MieConfig, cache_dir="cache"):
    """
    Heatmaps are stored alongside the LUT cache in cache/heatmaps/.
    """
    heatmap_dir = os.path.join(cache_dir, "heatmaps")
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
):
    """
    1. Checks if binary cache exists for these parameters.
    2. If not (or forced), runs the heavy LUT generation.
    3. Optionally generates a heatmap visualization into cache/heatmaps/.
    4. Returns the dictionary for Mitsuba XML/load_dict.
    """
    # 1. Setup Config
    config = MieConfig(
        radius_mean_um=radius_mean_um,
        radius_std_um=radius_std_um,
        num_angles=num_angles,
        num_wavelengths=num_wavelengths,
        note=note,
    )

    file_path = _get_cache_path(config)
    heatmap_path = _get_heatmap_path(config)

    # 2. Check Cache
    if force_regen or not os.path.exists(file_path):
        print(f"[Wrapper] Cache miss! Generating LUT for {config.output_filename}...")
        print(f"          (This might take a moment on the CPU)")

        # Run the generator
        table = generate_mie_table(config)

        # Save it (Transposed/Interleaved)
        save_binary_file(file_path, table, config)
        print(f"[Wrapper] Saved to {file_path}")

        # 3. Generate heatmap only on generation
        if generate_heatmap:
            try:
                visualize_binary_file(file_path, heatmap_path)
            except Exception as e:
                # Don’t fail rendering because of viz issues (matplotlib backend etc.)
                print(f"[Wrapper] Heatmap generation failed: {e}")
    else:
        print(f"[Wrapper] Found cached LUT: {file_path}")

        # Optional: if heatmap is missing, generate it even on cache-hit
        if generate_heatmap and not os.path.exists(heatmap_path):
            try:
                visualize_binary_file(file_path, heatmap_path)
            except Exception as e:
                print(f"[Wrapper] Heatmap generation failed (cache-hit): {e}")

    # 4. Return the Mitsuba Scene Dict Entry
    return {
        "type": "atmosphericphase",
        "filename": file_path,
    }
import os
from python.atmospheric.generator.generate import generate_mie_table, save_binary_file
from .config import MieConfig

def _get_cache_path(config: MieConfig, cache_dir="cache"):
    """
    Determines the unique filename based on the config.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return os.path.join(cache_dir, f"{config.output_filename}.bin")

def create_atmospheric_phase(radius_mean_um=2.0, radius_std_um=0.5,
                             num_angles=4096, num_wavelengths=64,
                             note="mist", force_regen=False):
    """
    1. Checks if binary cache exists for these parameters.
    2. If not (or forced), runs the heavy Mie generation.
    3. Returns the dictionary for Mitsuba XML/load_dict.
    """

    # 1. Setup Config
    config = MieConfig(
        radius_mean_um=radius_mean_um,
        radius_std_um=radius_std_um,
        num_angles=num_angles,
        num_wavelengths=num_wavelengths,
        note=note
    )

    file_path = _get_cache_path(config)

    # 2. Check Cache
    if force_regen or not os.path.exists(file_path):
        print(f"[Wrapper] Cache miss! Generating Mie LUT for {config.output_filename}...")
        print(f"          (This might take a moment on the CPU)")

        # Run the generator
        table = generate_mie_table(config)

        # Save it (Transposed/Interleaved)
        save_binary_file(file_path, table, config)
        print(f"[Wrapper] Saved to {file_path}")
    else:
        print(f"[Wrapper] Found cached LUT: {file_path}")

    # 3. Return the Mitsuba Scene Dict Entry
    # The C++ plugin expects a string param named 'filename'
    return {
        'type': 'atmosphericphase',
        'filename': file_path,
    }
"""
Side-by-side backend generation test.

Generates the same LUT configuration twice:
  - backend="mie"
  - backend="hybrid"

Relies on wrapper to:
  - cache into cache/<backend_name>/
  - generate heatmaps into cache/<backend_name>/heatmaps/
  - generate polars into cache/<backend_name>/polars/

This test is intended for visual inspection (not strict assertions).
"""
from __future__ import annotations

import os
from dataclasses import asdict

from python.atmospheric.wrapper import create_atmospheric_phase
from python.atmospheric.config import MieConfig

def _dump_expected_paths(config: MieConfig, backend_dir_name: str, cache_dir: str = "cache") -> None:
    base = os.path.join(cache_dir, backend_dir_name)
    bin_path = os.path.join(base, f"{config.output_filename}.bin")
    heat_path = os.path.join(base, "heatmaps", f"{config.output_filename}.png")
    polar_path = os.path.join(base, "polars", f"{config.output_filename}.png")

    print(f"[Expected:{backend_dir_name}] bin     : {bin_path}")
    print(f"[Expected:{backend_dir_name}] heatmap : {heat_path}")
    print(f"[Expected:{backend_dir_name}] polar   : {polar_path}")


def run_one_config(
    config: MieConfig,
    cache_dir: str = "cache",
    force_regen: bool = True,
) -> None:
    print("\n============================================================")
    print("[Test] Running config:")
    for k, v in asdict(config).items():
        print(f"  - {k}: {v}")
    print(f"  - output_filename: {config.output_filename}")
    print("============================================================\n")

    # # MIE
    # print("\n[Test] Generating MIE backend...")
    # create_atmospheric_phase(
    #     radius_mean_um=config.radius_mean_um,
    #     radius_std_um=config.radius_std_um,
    #     num_angles=config.num_angles,
    #     num_wavelengths=config.num_wavelengths,
    #     note=config.note,
    #     backend="mie",
    #     cache_dir=cache_dir,
    #     force_regen=force_regen,
    #     generate_heatmap=True,
    #     generate_polar=True,
    # )
    # _dump_expected_paths(config, backend_dir_name="miepython", cache_dir=cache_dir)

    # HYBRID
    print("\n[Test] Generating HYBRID backend...")
    create_atmospheric_phase(
        radius_mean_um=config.radius_mean_um,
        radius_std_um=config.radius_std_um,
        num_angles=config.num_angles,
        num_wavelengths=config.num_wavelengths,
        note=config.note,
        backend="hybrid",
        cache_dir=cache_dir,
        force_regen=force_regen,
        generate_heatmap=True,
        generate_polar=True,
    )
    _dump_expected_paths(config, backend_dir_name="hybrid", cache_dir=cache_dir)

    print("\n[Test] Done. Open the heatmaps/polars side-by-side for comparison.")


def main():
    # Pick configs that straddle the hybrid thresholds so switching actually happens.
    # You can add more configs here.

    configs = [
    #     MieConfig(
    #         radius_mean_um=10.0,
    #         radius_std_um=2.0,
    #         num_angles=2048,
    #         num_wavelengths=32,
    #         note="side_by_side_fog",
    #     ),
    #     # Drizzle-ish: likely straddles x0/x1 at visible wavelengths
        MieConfig(
            radius_mean_um=150.0,
            radius_std_um=20.0,
            num_angles=2048,
            num_wavelengths=32,
            note="side_by_side_drizzle",
        ),
        # Heavier rain-ish: should lean into GOAiry more
        # MieConfig(
        #     radius_mean_um=500.0,
        #     radius_std_um=150.0,
        #     num_angles=2048,
        #     num_wavelengths=32,
        #     note="side_by_side_rain",
        # ),
    ]

    for cfg in configs:
        run_one_config(cfg, cache_dir="cache", force_regen=True)


if __name__ == "__main__":
    main()
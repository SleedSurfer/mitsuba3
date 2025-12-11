"""
config.py
Single source of truth for Atmospheric Phase Function parameters.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MieConfig:
    # --- PHYSICS PARAMETERS ---
    radius_mean_um: float = 150.0
    radius_std_um: float = 20.0

    # --- SIMULATION RESOLUTION (The "Quality") ---
    num_angles: int = 1024
    num_wavelengths: int = 32
    min_wavelength: float = 360.0 # Near-UV
    max_wavelength: float = 830.0 # Near-IR

    num_samples: int = 50
    note: str = "mist" # Tag for the scenario

    @property
    def output_filename(self) -> str:
        """Standardized naming convention: physics + resolution."""
        return f"{int(self.radius_mean_um)}um_mean_{int(self.radius_std_um)}um_std_{self.num_wavelengths}bins_{self.num_angles}ang_{self.note}"

    def get_expected_file_size(self) -> tuple[int, int]:
        """
        Returns (min_size, max_size) for validation.

        min_size: Base LUT data only
        max_size: Base + reasonable MIS metadata overhead (~1KB)
        """
        header_size = 28
        data_size = self.num_angles * self.num_wavelengths * 4
        base_size = header_size + data_size

        # MIS metadata is typically 200-800 bytes depending on components
        max_mis_overhead = 1024  # 1KB buffer

        return base_size, base_size + max_mis_overhead

    def get_base_file_size(self):
        """Returns size of LUT data without MIS metadata."""
        header_size = 28
        data_size = self.num_angles * self.num_wavelengths * 4
        return header_size + data_size
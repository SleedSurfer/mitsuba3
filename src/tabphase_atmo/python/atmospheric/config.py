"""
config.py
Single source of truth for Atmospheric Phase Function parameters.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MieConfig:
    # --- PHYSICS PARAMETERS ---
    radius_mean_um: float = 2.0
    radius_std_um: float = 0.5

    # --- SIMULATION RESOLUTION (The "Quality") ---
    num_angles: int = 4096
    num_wavelengths: int = 64
    min_wavelength: float = 360.0 # Near-UV
    max_wavelength: float = 830.0 # Near-IR

    num_samples: int = 50
    note: str = "mist" # Tag for the scenario

    @property
    def output_filename(self) -> str:
        """Standardized naming convention: physics + resolution."""
        return f"{int(self.radius_mean_um)}um_mean_{int(self.radius_std_um)}um_std_{self.num_wavelengths}bins_{self.num_angles}ang_{self.note}"

    def get_expected_file_size(self) -> int:
        """Calculates exact binary size in bytes for validation."""
        # Header: Magic(8) + Version(4) + Angles(4) + Wvls(4) + Min(4) + Max(4) = 28 bytes
        header_size = 28
        # Data: Angles * Wavelengths * Float32(4)
        data_size = self.num_angles * self.num_wavelengths * 4
        return header_size + data_size
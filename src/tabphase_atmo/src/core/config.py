import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

class ParticleShape(Enum):
    SPHERE = auto()
    BEARD_CHUANG = auto()

@dataclass
class MieConfig:
    # --- PHYSICS PARAMETERS ---
    radius_mean_um: float = 700.0
    variance: float = 0.2  # e.g., 0.2 means a standard deviation of 20% of the mean
    material: str = "water"
    shape: ParticleShape = ParticleShape.SPHERE

    # --- SIMULATION RESOLUTION ---
    num_angles: int = 1024
    num_wavelengths: int = 0
    min_wavelength: float = 360.0
    max_wavelength: float = 830.0

    # --- ARCHITECTURE FLAGS ---
    brute_force_integration: bool = False # Set to True to actually simulate N particles
    num_samples: int = 0
    note: str = "auto"

    def __post_init__(self):
        # Autogen wavelengths
        if self.num_wavelengths <= 0:
            radius_mm = self.radius_mean_um / 1000.0
            base_samples = 300.0 * radius_mm
            self.num_wavelengths = int(np.clip(base_samples, 32, 512))

        # Autogen sample count (only matters if they are masochistic enough to brute force)
        if self.num_samples <= 0:
            if self.variance <= 0.0 or not self.brute_force_integration:
                self.num_samples = 1 # Force 1 sample for the fast-path
            else:
                # Dynamic scaling: higher variance = more samples needed to not alias the integral
                self.num_samples = int(np.clip(300.0 * self.variance, 16, 256))

    @property
    def output_filename(self) -> str:
        shape_str = self.shape.name.lower()
        var_pct = int(self.variance * 100)
        mode = "brute" if self.brute_force_integration else "fast"
        return f"{int(self.radius_mean_um)}um_{shape_str}_{var_pct}pctVar_{self.num_wavelengths}b_{mode}_{self.note}"
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto


class ParticleShape(Enum):
    SPHERE = auto()
    BEARD_CHUANG = auto()


class BackendType(Enum):
    AUTO = auto()
    MIEPYTHON = auto()
    REFERENCE = auto()
    DRJIT = auto()
    HYBRID = auto()


@dataclass
class MieConfig:
    # --- PHYSICS PARAMETERS ---
    radius_mean_um: float = 700.0
    variance: float = 0.2
    material: str = "water"
    shape: ParticleShape = ParticleShape.SPHERE

    # --- SIMULATION RESOLUTION ---
    num_angles: int = 1024
    num_wavelengths: int = 0
    min_wavelength: float = 360.0
    max_wavelength: float = 830.0

    # --- ARCHITECTURE FLAGS ---
    backend: BackendType = BackendType.AUTO  # <--- Moved here
    brute_force_integration: bool = False
    num_samples: int = 0
    note: str = ""  # No longer needed for backend hacks

    def __post_init__(self):
        # Autogen wavelengths
        if self.num_wavelengths <= 0:
            radius_mm = self.radius_mean_um / 1000.0
            base_samples = 300.0 * radius_mm
            self.num_wavelengths = int(np.clip(base_samples, 32, 512))

        # Autogen sample count
        if self.num_samples <= 0:
            if self.variance <= 0.0 or not self.brute_force_integration:
                self.num_samples = 1
            else:
                # Gauss-Hermite is hyper-efficient. 24 is the absolute ceiling for visual accuracy.
                self.num_samples = 16

    @property
    def output_filename(self) -> str:
        shape_str = self.shape.name.lower()
        var_pct = int(self.variance * 100)
        mode = "brute" if self.brute_force_integration else "fast"
        be_str = self.backend.name.lower()

        base = f"{int(self.radius_mean_um)}um_{shape_str}_{var_pct}pctVar_{self.num_wavelengths}ang_{self.num_angles}b_{mode}_{be_str}"
        return f"{base}_{self.note}" if self.note else base
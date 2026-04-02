import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

class ParticleShape(Enum):
    SPHERE = auto()
    OBLATE = auto()
    HEXAGONAL = auto()

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
    num_angles: int = 1024  # This is Theta (elevation)
    num_phi_bins: int = 1  # NEW: Azimuthal resolution. Default 1 for spheres.
    num_wavelengths: int = 0
    min_wavelength: float = 360.0
    max_wavelength: float = 830.0

    # --- ARCHITECTURE FLAGS ---
    backend: BackendType = BackendType.AUTO
    num_samples: int = 0
    note: str = ""

    def __post_init__(self):
        # Autogen wavelengths
        if self.num_wavelengths <= 0:
            radius_mm = self.radius_mean_um / 1000.0
            base_samples = 300.0 * radius_mm
            self.num_wavelengths = int(np.clip(base_samples, 32, 512))

        # Autogen sample count (Always assume full integration now)
        if self.num_samples <= 0:
            if self.variance <= 0.0:
                self.num_samples = 1
            else:
                self.num_samples = 16 # Gauss-Hermite ceiling

    @property
    def output_filename(self) -> str:
        shape_str = self.shape.name.lower()
        var_pct = int(self.variance * 100)
        be_str = self.backend.name.lower()

        base = f"{int(self.radius_mean_um)}um_{shape_str}_{var_pct}pctVar_{self.num_wavelengths}ang_{self.num_angles}b_{be_str}"
        return f"{base}_{self.note}" if self.note else base
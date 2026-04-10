from __future__ import annotations
from typing import Protocol, Union
from dataclasses import dataclass
import numpy as np

# --- STRICT SIZE STRUCTS ---

@dataclass(frozen=True)
class SphereSize:
    r_um: float

@dataclass(frozen=True)
class HexSize:
    c_axis_um: float
    a_axis_um: float

# Type alias for cleaner signatures
ParticleSize = Union[SphereSize, HexSize]

# --- THE PROTOCOL ---

class ScatteringBackend(Protocol):
    name: str

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, mu: np.ndarray, size: ParticleSize) -> np.ndarray:
        ...
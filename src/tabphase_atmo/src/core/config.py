import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, NamedTuple
from enum import Enum


# ==========================================
# 1. SHARED ENUMS (Unchanged)
# ==========================================
class BackendType(Enum):
    AUTO = "auto"
    MIEPYTHON = "miepython"
    REFERENCE = "reference"
    DRJIT = "drjit"
    HYBRID = "hybrid"


class HexagonalHabit(Enum):
    TUMBLING = "tumbling"
    PLATE = "plate"
    COLUMN_HORIZONTAL = "column_horizontal"
    PARRY = "parry"


class DropletShape(Enum):
    SPHERE = "sphere"
    OBLATE = "oblate"


class ParticleMaterial(Enum):
    ICE = "ice"
    WATER = "water"
    CUSTOM = "custom"

class DropletComposition(NamedTuple):
    """
    Defines a specific droplet population within the cloud volume.
    """
    shape: DropletShape
    weight: float           # Fractional weight (e.g., 0.4 for 40%)
    radius_mean_um: float   # Mean radius in micrometers (µm)
    variance: float         # Log-normal variance

class HexagonalComposition(NamedTuple):
    """
    Defines a specific ice crystal population within the cloud volume.
    """
    habit: HexagonalHabit
    weight: float           # Fractional weight (e.g., 0.6 for 60%)
    length_axis_um: float        # Length of the crystal in micrometers (µm)
    width_axis_um: float        # Width of the crystal in micrometers (µm)
    variance: float  # Log-normal variance

# ==========================================
# 2. INTERNAL DTOs
# ==========================================
@dataclass
class BaseParticleConfig:
    backend: BackendType = BackendType.AUTO
    num_phi_bins: int = 360
    num_angles: int = 1800
    material: ParticleMaterial = ParticleMaterial.ICE
    num_wavelengths: int = 16
    min_wavelength_nm: float = 380.0
    max_wavelength_nm: float = 700
    custom_ior_list: Optional[List[float]] = None

    composition: Optional[List[Tuple]] = field(default=None, init=False)
    wavelengths_nm: List[float] = field(init=False)
    computed_iors: List[float] = field(init=False)

    def __post_init__(self):
        # Swap out the dumb linspace for our smart picker
        self.wavelengths_nm = self._calculate_smart_wavelengths()

        if self.custom_ior_list is not None:
            self.material = ParticleMaterial.CUSTOM
            if len(self.custom_ior_list) != self.num_wavelengths:
                raise ValueError(f"[Physics Error] custom_ior_list must have {self.num_wavelengths} elements.")
            self.computed_iors = self.custom_ior_list
        else:
            self.computed_iors = self._calculate_ior_array()

    def _calculate_smart_wavelengths(self) -> List[float]:
        """
        Biases wavelength selection for low-res bakes to ensure core RGB colors
        are always sampled, preventing 'invisible' UV/IR bounds from stealing samples.
        """
        # If we have a healthy amount of samples, uniform spacing is mathematically best for the CDF
        if self.num_wavelengths >= 12:
            return np.linspace(self.min_wavelength_nm, self.max_wavelength_nm, self.num_wavelengths).tolist()

        # For "dirty" bakes, we prioritize the core optical anchors.
        # Order:            Green(Luma), Blue, Red, Deep Violet, Deep Red, Cyan, Yellow, Edge IR, Edge UV
        priority_anchors = [540.0,      450.0,650.0,   400.0,      700.0, 490.0, 590.0,   770.0,   380.0, 420.0, 620.0]

        # Take the top N requested
        selected = priority_anchors[:self.num_wavelengths]

        # If they ask for exactly 11 but somehow trigger this branch, fallback gracefully
        if self.num_wavelengths > len(priority_anchors):
            return np.linspace(self.min_wavelength_nm, self.max_wavelength_nm, self.num_wavelengths).tolist()

        # Clamp them to the user's defined boundaries just in case they modified min/max
        clamped = [max(self.min_wavelength_nm, min(w, self.max_wavelength_nm)) for w in selected]

        # Sort them linearly so the renderer processes them sequentially (helps with debugging output)
        return sorted(clamped)

    def _calculate_ior_array(self) -> List[float]:
        iors = []
        for wl_nm in self.wavelengths_nm:
            wl_um = wl_nm / 1000.0
            if self.material == ParticleMaterial.WATER:
                b1, b2, b3 = 5.689093832e-1, 1.719708856e-1, 2.062501582e-2
                c1, c2, c3 = 5.110301794e-3, 1.825180155e-2, 2.624158904e-2
                wl_sq = wl_um ** 2
                n_sq = 1.0 + (b1 * wl_sq) / (wl_sq - c1) + (b2 * wl_sq) / (wl_sq - c2) + (b3 * wl_sq) / (wl_sq - c3)
                iors.append(math.sqrt(n_sq))
            elif self.material == ParticleMaterial.ICE:
                wb_wavs = np.array([0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 1.000, 1.200])
                wb_iors = np.array([1.334, 1.319, 1.313, 1.309, 1.306, 1.304, 1.301, 1.298])
                iors.append(float(np.interp(wl_um, wb_wavs, wb_iors)))
            else:
                iors.append(1.0)
        return iors


@dataclass
class DropletConfig(BaseParticleConfig):
    # Composition: Tuple[Shape, Weight, Radius_Mean_um, Variance]
    composition: List[DropletComposition] = field(default_factory=list)

    def __post_init__(self):
        if not self.composition:
            self.composition = [DropletComposition(DropletShape.SPHERE, 1.0, 10.0, 0.1)]

        total_weight = sum(w for _, w, _, _ in self.composition)
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            raise ValueError(f"[Config Error] Weights sum to {total_weight}, not 1.0.")

        if self.custom_ior_list is None:
            self.material = ParticleMaterial.WATER
        super().__post_init__()


@dataclass
class HexagonalConfig(BaseParticleConfig):
    # Composition: Tuple[Habit, Weight, C_axis_length_um, A_axis_width_um]
    composition: List[HexagonalComposition] = field(default_factory=list)
    air_turbulence_factor: float = 1.0
    sun_elevation_deg: float = 0.0
    def __post_init__(self):
        if not self.composition:
            raise ValueError("[Config Error] Empty composition. Give me some crystal habits.")

        total_weight = sum(w for _, w, _, _, _ in self.composition)
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            raise ValueError(f"[Config Error] Weights sum to {total_weight}, not 1.0.")

        if self.custom_ior_list is None:
            self.material = ParticleMaterial.ICE
        super().__post_init__()


# ==========================================
# 3. THE PUBLIC API FACADE
# ==========================================
class Particle:
    @staticmethod
    def droplet(
            composition: List[DropletComposition],
            backend: BackendType = BackendType.AUTO,
            material: ParticleMaterial = ParticleMaterial.WATER,
            num_wavelengths: int = 16,
            min_wavelength_nm: float = 380.0,
            max_wavelength_nm: float = 700,
            custom_ior_list: Optional[List[float]] = None,
            num_phi_bins: int = 360,
            num_angles: int = 1800
    ) -> DropletConfig:
        return DropletConfig(
            composition=composition,
            backend=backend,
            material=material,
            num_wavelengths=num_wavelengths,
            min_wavelength_nm=min_wavelength_nm,
            max_wavelength_nm=max_wavelength_nm,
            custom_ior_list=custom_ior_list,
            num_phi_bins=num_phi_bins,
            num_angles=num_angles
        )

    @staticmethod
    def hexagonal(
            composition: List[HexagonalComposition],
            sun_elevation_deg: float = 0.0,
            air_turbulence_factor: float = 1.0,
            backend: BackendType = BackendType.DRJIT,
            material: ParticleMaterial = ParticleMaterial.ICE,
            num_wavelengths: int = 16,
            min_wavelength_nm: float = 380.0,
            max_wavelength_nm: float = 700,
            custom_ior_list: Optional[List[float]] = None,
            num_phi_bins: int = 360,
            num_angles: int = 1800
    ) -> HexagonalConfig:
        return HexagonalConfig(
            composition=composition,
            sun_elevation_deg=sun_elevation_deg,
            air_turbulence_factor=air_turbulence_factor,
            backend=backend,
            material=material,
            num_wavelengths=num_wavelengths,
            min_wavelength_nm=min_wavelength_nm,
            max_wavelength_nm=max_wavelength_nm,
            custom_ior_list=custom_ior_list,
            num_phi_bins=num_phi_bins,
            num_angles=num_angles
        )
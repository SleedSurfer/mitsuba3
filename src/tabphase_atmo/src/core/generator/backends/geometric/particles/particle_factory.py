from .spherical.sphere import SphereParticle
from .spherical.oblate_sphere import OblateSpheroidParticle
from .hexagonal.hexagonal_tumbling import TumblingHexagon
from .hexagonal.hexagonal_plate import PlateHexagon
from .hexagonal.hexagonal_horizontal_column import HorizontalColumn
from .hexagonal.hexagonal_parry import ParryColumn

from .base_particle import SphericalParticle, PrismParticle
from ...base import ParticleSize, SphereSize, HexSize


def build_particle(shape_str: str, size: ParticleSize, sun_elevation_deg: float = 0.0):
    # Smooth Geometries (Unchanged)
    if isinstance(size, SphereSize):
        radius_mm = size.r_um / 1000.0
        if shape_str == "sphere":
            return SphereParticle(radius_mm=radius_mm)
        elif shape_str == "oblate":
            return OblateSpheroidParticle(radius_mm=radius_mm)
        else:
            raise ValueError(f"Shape '{shape_str}' is not compatible with SphereSize.")

    # Prism Geometries
    elif isinstance(size, HexSize):
        a_mm = size.a_axis_um / 1000.0
        c_mm = size.c_axis_um / 1000.0

        if shape_str == "tumbling":
            # Tumbling doesn't care about sun angle (statistically invariant)
            return TumblingHexagon(radius_mm=a_mm, height_mm=c_mm)

        elif shape_str == "plate":
            return PlateHexagon(
                radius_mm=a_mm,
                height_mm=c_mm,
                sun_elevation_deg=sun_elevation_deg
            )

        elif shape_str == "column_horizontal":
            # Tangent Arcs: Horizontal columns with free roll
            return HorizontalColumn(
                radius_mm=a_mm,
                height_mm=c_mm,
                sun_elevation_deg=sun_elevation_deg
            )

        elif shape_str == "parry":
            # Parry Arcs: Horizontal columns with locked roll
            return ParryColumn(
                radius_mm=a_mm,
                height_mm=c_mm,
                sun_elevation_deg=sun_elevation_deg
            )

        else:
            raise ValueError(f"Shape '{shape_str}' is not recognized in HexagonalHabit.")

    else:
        raise TypeError(f"Unrecognized size struct: {type(size)}")
from .spherical.sphere import SphereParticle
from .spherical.oblate_sphere import OblateSpheroidParticle
from .hexagonal.hexagonal_tumbling import TumblingHexagon
from .hexagonal.hexagonal_plate import PlateHexagon
from .hexagonal.hexagonal_horizontal_column import HorizontalColumn
from .hexagonal.hexagonal_parry import ParryColumn
from .cuboctahedral.cuboctahedral_tumbling import TumblingCuboctahedron
from .cuboctahedral.cuboctahedral_plate import PlateCuboctahedron

from .base_particle import SphericalParticle, PrismParticle
from ...base import ParticleSize, SphereSize, HexSize, CuboctaSize


def build_particle(shape_str: str, size: ParticleSize, sun_elevation_deg: float = 0.0, air_turbulence_factor: float = 1.0):
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
            return TumblingHexagon(radius_mm=a_mm, height_mm=c_mm)

        elif shape_str == "plate":
            return PlateHexagon(
                radius_mm=a_mm,
                height_mm=c_mm,
                sun_elevation_deg=sun_elevation_deg,
                air_turbulence_factor=air_turbulence_factor
            )

        elif shape_str == "column_horizontal":
            # Tangent Arcs: Horizontal columns with free roll
            return HorizontalColumn(
                radius_mm=a_mm,
                height_mm=c_mm,
                sun_elevation_deg=sun_elevation_deg,
                air_turbulence_factor=air_turbulence_factor
            )

        elif shape_str == "parry":
            # Parry Arcs: Horizontal columns with locked roll
            return ParryColumn(
                radius_mm=a_mm,
                height_mm=c_mm,
                sun_elevation_deg=sun_elevation_deg,
                air_turbulence_factor=air_turbulence_factor
            )

        else:
            raise ValueError(f"Shape '{shape_str}' is not recognized in HexagonalHabit.")

    elif isinstance(size, CuboctaSize):
        # We pass the raw um value down, let the class handle the mm conversion
        if shape_str == "tumbling":
            # Assuming you put the class in a cuboctahedral_tumbling file
            return TumblingCuboctahedron(a_axis_um=size.a_axis_um)
        elif shape_str == "plate":
            return PlateCuboctahedron(a_axis_um=size.a_axis_um,sun_elevation_deg=sun_elevation_deg,air_turbulence_factor=air_turbulence_factor)
        else:
            raise ValueError(f"Shape '{shape_str}' is not recognized in CuboctahedralHabit.")

    else:
        raise TypeError(f"Unrecognized size struct: {type(size)}")
import numpy as np
import drjit as dr
from drjit.auto import Float, Array3f

from .hexagonal_base import HexagonalParticle


class PlateHexagon(HexagonalParticle):
    """
    Forms Sun Dogs (Parhelia) and Circumzenithal Arcs.
    """

    def __init__(self, radius_mm: float, height_mm: float, flutter_deg: float = 1.5, sun_elevation_deg: float = 0.0,
                 air_turbulence_factor: float = 1.0):
        if radius_mm <= height_mm:
            raise ValueError(
                f"[Physics Error] Plate crystals must be wider than they are tall to fall flat. "
                f"Got radius: {radius_mm}mm, height: {height_mm}mm."
            )
        super().__init__(radius_mm, height_mm)
        self.flutter_rad = (flutter_deg * air_turbulence_factor) * np.pi / 180.0
        self.sun_elevation_deg = sun_elevation_deg

    def randomize_orientations(self, num_rays: int, rng):
        u1 = dr.maximum(rng.next_float32(), 1e-6)
        tilt = self.flutter_rad * dr.sqrt(-2.0 * dr.log(u1))
        tilt_azimuth = rng.next_float32() * 2.0 * np.pi

        n0_x = dr.sin(tilt) * dr.cos(tilt_azimuth)
        n0_y = dr.cos(tilt)
        n0_z = dr.sin(tilt) * dr.sin(tilt_azimuth)

        sun_elevation_rad = Float(self.sun_elevation_deg * np.pi / 180.0)
        cos_sun = dr.cos(sun_elevation_rad)
        sin_sun = dr.sin(sun_elevation_rad)

        n0_y_tilted = n0_y * cos_sun - n0_z * sin_sun
        n0_z_tilted = n0_y * sin_sun + n0_z * cos_sun

        self.n0 = dr.normalize(Array3f(n0_x, n0_y_tilted, n0_z_tilted))

        temp_x = Array3f(1.0, 0.0, 0.0)
        b1 = dr.normalize(dr.cross(self.n0, temp_x))
        b2 = dr.cross(self.n0, b1)

        yaw = rng.next_float32() * 2.0 * np.pi
        cos_y = dr.cos(yaw)
        sin_y = dr.sin(yaw)

        self.n1 = b1 * cos_y + b2 * sin_y

        s32 = Float(np.sqrt(3.0) / 2.0)
        cross_01 = dr.cross(self.n0, self.n1)

        self.n2 = self.n1 * 0.5 + cross_01 * s32
        self.n3 = self.n1 * (-0.5) + cross_01 * s32
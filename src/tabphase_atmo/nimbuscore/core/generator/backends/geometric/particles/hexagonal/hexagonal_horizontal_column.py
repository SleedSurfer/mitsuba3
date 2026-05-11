import numpy as np
import drjit as dr
from drjit.auto import Float, Array3f

from .hexagonal_base import HexagonalParticle


class HorizontalColumn(HexagonalParticle):
    def __init__(self, radius_mm: float, height_mm: float, flutter_deg: float = 1.0, sun_elevation_deg: float = 0.0,
                 air_turbulence_factor: float = 1.0):
        super().__init__(radius_mm, height_mm)
        self.flutter_rad = (flutter_deg * air_turbulence_factor) * np.pi / 180.0
        self.sun_elevation_deg = sun_elevation_deg

    def randomize_orientations(self, num_rays: int, rng):
        azimuth = rng.next_float32() * 2.0 * np.pi
        n0_x = dr.cos(azimuth)
        n0_z = dr.sin(azimuth)

        u1 = dr.maximum(rng.next_float32(), 1e-6)
        tilt = self.flutter_rad * dr.sqrt(-2.0 * dr.log(u1))
        n0_y = dr.sin(tilt)
        cos_tilt = dr.cos(tilt)

        n0_vec = dr.normalize(Array3f(n0_x * cos_tilt, n0_y, n0_z * cos_tilt))

        sun_elev_rad = Float(self.sun_elevation_deg * np.pi / 180.0)
        cos_s = dr.cos(sun_elev_rad)
        sin_s = dr.sin(sun_elev_rad)

        final_n0_y = n0_vec.y * cos_s - n0_vec.z * sin_s
        final_n0_z = n0_vec.y * sin_s + n0_vec.z * cos_s
        self.n0 = dr.normalize(Array3f(n0_vec.x, final_n0_y, final_n0_z))

        tilted_up = Array3f(0.0, cos_s, sin_s)
        b1 = dr.normalize(dr.cross(self.n0, tilted_up))
        b2 = dr.cross(self.n0, b1)

        # Random Roll
        roll = rng.next_float32() * 2.0 * np.pi
        self.n1 = b1 * dr.cos(roll) + b2 * dr.sin(roll)

        # Generate remaining faces
        s32 = Float(np.sqrt(3.0) / 2.0)
        cross_01 = dr.cross(self.n0, self.n1)
        self.n2 = self.n1 * 0.5 + cross_01 * s32
        self.n3 = self.n1 * (-0.5) + cross_01 * s32
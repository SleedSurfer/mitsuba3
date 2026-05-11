import numpy as np
import drjit as dr
from drjit.auto import Float, Array3f, Matrix3f
from .cuboctahedral_base import CuboctahedralParticle


class PlateCuboctahedron(CuboctahedralParticle):
    """
    Martian Plate Habit.
    Falls with a square face aligned to the horizon, creating Martian Sundogs.
    """

    def __init__(self, a_axis_um: float, flutter_deg: float = 1.5,
                 sun_elevation_deg: float = 0.0, air_turbulence_factor: float = 1.0):
        super().__init__(a_axis_um)
        self.flutter_rad = (flutter_deg * air_turbulence_factor) * np.pi / 180.0
        self.sun_elevation_deg = sun_elevation_deg

    def randomize_orientations(self, num_rays: int, rng):
        # 1. Generate the fluttering "Up" vector (Local Y mapping to World)
        u1 = dr.maximum(rng.next_float32(), 1e-6)
        tilt = self.flutter_rad * dr.sqrt(-2.0 * dr.log(u1))
        tilt_azimuth = rng.next_float32() * 2.0 * np.pi

        n0_x = dr.sin(tilt) * dr.cos(tilt_azimuth)
        n0_y = dr.cos(tilt)
        n0_z = dr.sin(tilt) * dr.sin(tilt_azimuth)

        # 2. Apply Sun Elevation (Pitch) to the Up vector
        sun_elev_rad = Float(self.sun_elevation_deg * np.pi / 180.0)
        cos_sun = dr.cos(sun_elev_rad)
        sin_sun = dr.sin(sun_elev_rad)

        n0_y_tilted = n0_y * cos_sun - n0_z * sin_sun
        n0_z_tilted = n0_y * sin_sun + n0_z * cos_sun

        # This is our Local Y-axis expressed in World Space
        n0 = dr.normalize(Array3f(n0_x, n0_y_tilted, n0_z_tilted))

        # 3. Generate a random Yaw to define the Local X-axis
        temp_x = Array3f(1.0, 0.0, 0.0)
        b1 = dr.normalize(dr.cross(n0, temp_x))
        b2 = dr.cross(n0, b1)

        yaw = rng.next_float32() * 2.0 * np.pi
        cos_y = dr.cos(yaw)
        sin_y = dr.sin(yaw)

        # This is our Local X-axis expressed in World Space
        n1 = b1 * cos_y + b2 * sin_y

        # 4. Local Z-axis is just X cross Y
        n2 = dr.cross(n1, n0)

        # 5. Assemble the Master Rotation Matrix
        # Because n1, n0, n2 are the local X, Y, Z axes in world coordinates,
        # they perfectly slot in as the columns of the transformation matrix.
        rot = Matrix3f(
            [n1.x, n0.x, n2.x],
            [n1.y, n0.y, n2.y],
            [n1.z, n0.z, n2.z]
        )

        self.to_world = rot
        self.to_local = rot.T
        return rot
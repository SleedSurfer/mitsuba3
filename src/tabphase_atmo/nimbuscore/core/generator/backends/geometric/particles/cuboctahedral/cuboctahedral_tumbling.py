import drjit as dr
from drjit.auto import Matrix3f
from .cuboctahedral_base import CuboctahedralParticle

class TumblingCuboctahedron(CuboctahedralParticle):
    """
    A Martian CO2 crystal that tumbles isotropically.
    """
    def __init__(self, a_axis_um: float, sun_elevation_deg: float = 0.0):
        super().__init__(a_axis_um)
        self.sun_elevation_deg = sun_elevation_deg

    def randomize_orientations(self, num_rays: int, rng):
        """
        Generates a batch of random 3D rotation matrices.
        """
        u1 = rng.next_float32()
        u2 = rng.next_float32()
        u3 = rng.next_float32()

        tau = 2.0 * dr.pi
        s1, c1 = dr.sincos(tau * u2)
        s2, c2 = dr.sincos(tau * u3)
        r1 = dr.sqrt(1.0 - u1)
        r2 = dr.sqrt(u1)

        qw = r1 * s1
        qx = r1 * c1
        qy = r2 * s2
        qz = r2 * c2

        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        xw, yw, zw = qx * qw, qy * qw, qz * qw

        m00 = 1.0 - 2.0 * (yy + zz)
        m01 = 2.0 * (xy - zw)
        m02 = 2.0 * (xz + yw)

        m10 = 2.0 * (xy + zw)
        m11 = 1.0 - 2.0 * (xx + zz)
        m12 = 2.0 * (yz - xw)

        m20 = 2.0 * (xz - yw)
        m21 = 2.0 * (yz + xw)
        m22 = 1.0 - 2.0 * (xx + yy)

        rot = Matrix3f(
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]
        )

        # Update the base class transformation matrices
        self.to_world = rot
        self.to_local = dr.transpose(rot)
        return rot
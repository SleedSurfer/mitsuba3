import numpy as np
import drjit as dr
from drjit.auto import Float, Bool, Array3f, Complex2f
from .hexagonal_base import HexagonalParticle

class TumblingHexagon(HexagonalParticle):
    """
    Forms the 22-degree halo.
    These are columns (pencils) that tumble chaotically in 3D space.
    """

    def __init__(self, radius_mm: float, height_mm: float):
        # Strict geometric validation
        if height_mm < radius_mm:
            raise ValueError(
                f"[Physics Error] Tumbling columns must be longer than they are wide. "
                f"Got height: {height_mm}mm, radius: {radius_mm}mm. Check your config."
            )
        super().__init__(radius_mm, height_mm)

    def randomize_orientations(self, num_rays: int, rng):
        # Full 3D quaternion rotation (Your original black magic)
        u1 = rng.next_float32()
        u2 = rng.next_float32()
        u3 = rng.next_float32()

        w = dr.sqrt(1.0 - u1) * dr.sin(2.0 * np.pi * u2)
        x = dr.sqrt(1.0 - u1) * dr.cos(2.0 * np.pi * u2)
        y = dr.sqrt(u1) * dr.sin(2.0 * np.pi * u3)
        z = dr.sqrt(u1) * dr.cos(2.0 * np.pi * u3)

        R00 = 1.0 - 2.0 * dr.sqr(y) - 2.0 * dr.sqr(z)
        R01 = 2.0 * x * y - 2.0 * w * z
        R02 = 2.0 * x * z + 2.0 * w * y

        R10 = 2.0 * x * y + 2.0 * w * z
        R11 = 1.0 - 2.0 * dr.sqr(x) - 2.0 * dr.sqr(z)
        R12 = 2.0 * y * z - 2.0 * w * x

        R20 = 2.0 * x * z - 2.0 * w * y
        R21 = 2.0 * y * z + 2.0 * w * x
        R22 = 1.0 - 2.0 * dr.sqr(x) - 2.0 * dr.sqr(y)

        self.n0 = Array3f(R01, R11, R21)
        self.n1 = Array3f(R00, R10, R20)

        s32 = Float(np.sqrt(3.0) / 2.0)
        self.n2 = Array3f(R00 * 0.5 + R02 * s32, R10 * 0.5 + R12 * s32, R20 * 0.5 + R22 * s32)
        self.n3 = Array3f(R00 * -0.5 + R02 * s32, R10 * -0.5 + R12 * s32, R20 * -0.5 + R22 * s32)


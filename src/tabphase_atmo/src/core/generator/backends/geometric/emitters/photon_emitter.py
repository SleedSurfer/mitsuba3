import drjit as dr
from drjit.auto import Float, Array3f, Complex2f, PCG32, UInt32
import numpy as np

from src.core.generator.backends.geometric.models.phasor_ray import PhasorRay


class PhotonEmitter:
    @staticmethod
    def emit(num_rays: int, target_radius_mm: float, rng, pol: str = 'X') -> PhasorRay:
        u1 = rng.next_float32()
        u2 = rng.next_float32()

        r = Float(target_radius_mm) * dr.sqrt(u1)
        theta = u2 * 2.0 * np.pi

        ox = r * dr.cos(theta)
        oy = r * dr.sin(theta)
        oz = dr.full(Float, -10.0, num_rays)

        direction = Array3f(0.0, 0.0, 1.0)

        if pol == 'X':
            ex, ey = Complex2f(1.0, 0.0), Complex2f(0.0, 0.0)
        else:
            ex, ey = Complex2f(0.0, 0.0), Complex2f(1.0, 0.0)

        rays = PhasorRay(
            origin=Array3f(ox, oy, oz),
            direction=direction,
            Ex=ex, Ey=ey,
            basis_x=Array3f(1.0, 0.0, 0.0),
            basis_y=Array3f(0.0, 1.0, 0.0),
            opt_path_length=Float(0.0),
            focal_lines_crossed=dr.zeros(UInt32, num_rays)
        )
        return rays
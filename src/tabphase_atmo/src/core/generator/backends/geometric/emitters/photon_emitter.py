import drjit as dr
from drjit.auto import Float, Array3f, Complex2f, PCG32, UInt32
import numpy as np

from ..models.photon_ray import PhotonRay

class PhotonEmitter:
    @staticmethod
    def emit(num_rays: int, target_radius_mm: float, rng) -> PhotonRay:
        u1 = rng.next_float32()
        u2 = rng.next_float32()

        r = Float(target_radius_mm) * dr.sqrt(u1)
        theta = u2 * 2.0 * np.pi

        ox = r * dr.cos(theta)
        oy = r * dr.sin(theta)
        oz = dr.full(Float, -10.0, num_rays)

        direction = Array3f(0.0, 0.0, 1.0)

        rays = PhotonRay(
            origin=Array3f(ox, oy, oz),
            direction=direction,
            weight=dr.ones(Float, num_rays)
        )
        return rays
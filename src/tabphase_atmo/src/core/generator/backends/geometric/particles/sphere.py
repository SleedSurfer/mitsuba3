import drjit as dr
from drjit.auto import Float, Bool, Array3f, Complex2f
from typing import Tuple

from .base import Particle
from ..models.phasor_ray import PhasorRay
from ..optics import compute_fresnel_and_scatter


class SphericalParticle(Particle):
    def __init__(self, radius_mm: float):
        self.radius = Float(radius_mm)

    def intersect(self, rays: PhasorRay) -> Tuple[Bool, Float]:
        b = 2.0 * dr.dot(rays.origin, rays.direction)
        c = dr.dot(rays.origin, rays.origin) - (self.radius * self.radius)
        delta = (b * b) - (4.0 * c)

        hit_mask = delta >= 0.0
        sqrt_delta = dr.sqrt(dr.maximum(delta, Float(0.0)))

        t1 = (-b - sqrt_delta) / 2.0
        t2 = (-b + sqrt_delta) / 2.0

        eps = Float(1e-5)

        t1_valid = t1 > eps
        t2_valid = t2 > eps

        t = dr.select(t1_valid, t1, t2)

        valid_hit_mask = hit_mask & (t1_valid | t2_valid)

        return valid_hit_mask, t

    def scatter(self, rays: PhasorRay, hit_mask: Bool, t: Float, ior_water: float = 1.333) -> Tuple[
        Array3f, Array3f, Complex2f, Complex2f, Complex2f, Complex2f, Array3f]:
        # 1. Exact 3D point of impact
        p = rays.origin + rays.direction * t

        # 2. Surface normal (pointing OUT of the sphere)
        n = p / self.radius

        # 3. Offload physics to optics.py
        out = compute_fresnel_and_scatter(rays.direction, n, 1.0, ior_water)
        d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, is_tir = out

        return d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, n
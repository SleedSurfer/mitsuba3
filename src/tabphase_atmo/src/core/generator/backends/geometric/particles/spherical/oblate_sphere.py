import drjit as dr
from drjit.auto import Float, Bool, Array3f, Complex2f
from typing import Tuple

from ...models.phasor_ray import PhasorRay
from ...optics import compute_fresnel_and_scatter
from ..base_particle import SphericalParticle


class OblateSpheroidParticle(SphericalParticle):
    def __init__(self, radius_mm: float, aspect_ratio: float = 0.98):
        self.radius = Float(radius_mm)
        self.ar = Float(aspect_ratio)
        # Precompute the squared aspect ratio for normal scaling
        self.ar_sq = self.ar * self.ar

    def intersect(self, rays: PhasorRay) -> Tuple[Bool, Float]:
        # 1. Warp the rays to turn the oblate into a perfect sphere
        # We only scale the Y component by 1.0 / aspect_ratio
        inv_a = 1.0 / self.ar
        o_warp = Array3f(rays.origin.x, rays.origin.y * inv_a, rays.origin.z)
        d_warp = Array3f(rays.direction.x, rays.direction.y * inv_a, rays.direction.z)

        # 2. Standard stable quadratic solver
        a = dr.dot(d_warp, d_warp)
        b = 2.0 * dr.dot(o_warp, d_warp)
        c = dr.dot(o_warp, o_warp) - dr.sqr(self.radius)

        delta = (b * b) - (4.0 * a * c)
        hit_mask = delta >= 0.0

        sqrt_delta = dr.sqrt(dr.maximum(delta, 0.0))
        t1 = (-b - sqrt_delta) / (2.0 * a)
        t2 = (-b + sqrt_delta) / (2.0 * a)

        # 3. Inside/Outside logic (same as the perfect sphere)
        eps = Float(1e-4)
        t1_valid = t1 > eps
        t2_valid = t2 > eps

        # If t1 is positive, we are outside. If only t2 is positive, we are inside.
        t = dr.select(t1_valid, t1, dr.select(t2_valid, t2, Float(0.0)))
        active = hit_mask & (t1_valid | t2_valid)

        return active, t

    def scatter(self, rays: PhasorRay, hit_mask: Bool, t: Float, ior_water: float = 1.333) -> Tuple[
        Array3f, Array3f, Complex2f, Complex2f, Complex2f, Complex2f, Array3f, Bool]:
        # 1. Real exact world-space hit point
        p_world = rays.origin + rays.direction * t

        # 2. Perfect analytical normal for an oblate spheroid
        # The gradient of (x^2 + (y/a)^2 + z^2) is [2x, 2y/a^2, 2z]
        # We drop the 2s and normalize. ZERO cross products = ZERO seams.
        n_unnorm = Array3f(p_world.x, p_world.y / self.ar_sq, p_world.z)
        n_world = dr.normalize(n_unnorm)

        # 3. Scatter
        out = compute_fresnel_and_scatter(rays.direction, n_world, 1.0, ior_water)
        d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, is_tir = out

        return d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, n_world, is_tir
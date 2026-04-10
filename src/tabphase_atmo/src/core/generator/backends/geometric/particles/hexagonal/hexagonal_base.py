import numpy as np
import drjit as dr
from drjit.auto import Float, Bool, Array3f, Complex2f
from typing import Tuple
from ..base_particle import PrismParticle
from ...models.phasor_ray import PhasorRay
from ...optics import compute_fresnel_and_scatter



class HexagonalParticle(PrismParticle):
    """
    The pure mathematical skeleton of a hexagonal slab.
    It knows how to intersect and scatter, but it has no idea how it's oriented in the sky.
    """

    def __init__(self, radius_mm: float, height_mm: float):
        self.R = Float(radius_mm)
        self.H = Float(height_mm / 2.0)

        # Distances to the planes
        self.d0 = self.H
        self.d1 = self.R
        self.d2 = self.R
        self.d3 = self.R

        # Default unrotated normals (Aligned to Y-axis)
        self.n0 = Array3f(0.0, 1.0, 0.0)
        self.n1 = Array3f(1.0, 0.0, 0.0)
        self.n2 = Array3f(0.5, 0.0, float(np.sqrt(3.0) / 2.0))
        self.n3 = Array3f(-0.5, 0.0, float(np.sqrt(3.0) / 2.0))

    def _intersect_slab(self, rays: PhasorRay, N_i: Array3f, D_i: Float):
        denom = dr.dot(rays.direction, N_i)
        p_dot_n = dr.dot(rays.origin, N_i)

        inv_denom = Float(1.0) / denom

        t_pos_raw = (D_i - p_dot_n) * inv_denom
        t_neg_raw = (-D_i - p_dot_n) * inv_denom

        tn = dr.minimum(t_pos_raw, t_neg_raw)
        tf = dr.maximum(t_pos_raw, t_neg_raw)

        is_parallel = dr.abs(denom) < 1e-6
        is_inside = dr.abs(p_dot_n) <= (D_i + 1e-5)

        tn_safe = dr.select(is_parallel, dr.select(is_inside, Float(-1e5), Float(1e5)), tn)
        tf_safe = dr.select(is_parallel, dr.select(is_inside, Float(1e5), Float(-1e5)), tf)

        return tn_safe, tf_safe

    def intersect(self, rays: PhasorRay) -> Tuple[Bool, Float]:
        tn0, tf0 = self._intersect_slab(rays, self.n0, self.d0)
        tn1, tf1 = self._intersect_slab(rays, self.n1, self.d1)
        tn2, tf2 = self._intersect_slab(rays, self.n2, self.d2)
        tn3, tf3 = self._intersect_slab(rays, self.n3, self.d3)

        t_entry = dr.maximum(dr.maximum(tn0, tn1), dr.maximum(tn2, tn3))
        t_exit = dr.minimum(dr.minimum(tf0, tf1), dr.minimum(tf2, tf3))

        hit_mask = (t_entry <= t_exit) & (t_exit > 1e-4) & (t_entry < 1e5)

        is_internal = t_entry < 1e-4
        t = dr.select(is_internal, dr.maximum(t_exit, Float(0.0)), t_entry)

        return hit_mask, dr.select(hit_mask, t, Float(0.0))

    def scatter(self, rays: PhasorRay, hit_mask: Bool, t: Float, ior_water: float = 1.31) -> Tuple[
        Array3f, Array3f, Complex2f, Complex2f, Complex2f, Complex2f, Array3f, Bool]:
        p_world = rays.origin + rays.direction * t

        dot0 = dr.dot(p_world, self.n0) / self.d0
        dot1 = dr.dot(p_world, self.n1) / self.d1
        dot2 = dr.dot(p_world, self.n2) / self.d2
        dot3 = dr.dot(p_world, self.n3) / self.d3

        abs0, abs1 = dr.abs(dot0), dr.abs(dot1)
        abs2, abs3 = dr.abs(dot2), dr.abs(dot3)

        max_dot = dr.maximum(dr.maximum(abs0, abs1), dr.maximum(abs2, abs3))

        n_world = dr.select(
            abs0 >= max_dot * 0.9999, self.n0 * dr.sign(dot0),
            dr.select(
                abs1 >= max_dot * 0.9999, self.n1 * dr.sign(dot1),
                dr.select(
                    abs2 >= max_dot * 0.9999, self.n2 * dr.sign(dot2),
                    self.n3 * dr.sign(dot3)
                )
            )
        )

        out = compute_fresnel_and_scatter(rays.direction, n_world, 1.0, ior_water)
        d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, is_tir = out

        return d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, n_world, is_tir
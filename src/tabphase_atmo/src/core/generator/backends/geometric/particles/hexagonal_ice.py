import drjit as dr
from drjit.auto import Float, Bool, Array3f, Complex2f
from typing import Tuple

from .base import Particle
from ..models.phasor_ray import PhasorRay
from ..optics import compute_fresnel_and_scatter


class HexagonalCrystalParticle(Particle):
    # Added **kwargs and optional height_mm so your driver doesn't choke
    def __init__(self, radius_mm: float, height_mm: float = None, **kwargs):
        self.R = Float(radius_mm)
        # Default to a standard column (Height = 2x Radius) if driver doesn't provide it
        self.H = Float(height_mm / 2.0 if height_mm else radius_mm)

        # The 4 normal vectors defining the slabs
        self.n0 = Array3f(0.0, 1.0, 0.0)  # Top/Bottom Caps
        self.n1 = Array3f(1.0, 0.0, 0.0)  # Side Pair 1
        self.n2 = Array3f(0.5, 0.0, dr.sqrt(3.0) / 2.0)  # Side Pair 2
        self.n3 = Array3f(-0.5, 0.0, dr.sqrt(3.0) / 2.0)  # Side Pair 3

        self.d0 = self.H
        self.d1 = self.R
        self.d2 = self.R
        self.d3 = self.R

    def _intersect_slab(self, rays: PhasorRay, N_i: Array3f, D_i: Float):
        denom = dr.dot(rays.direction, N_i)
        p_dot_n = dr.dot(rays.origin, N_i)

        # Let DrJit do its thing. 1.0 / 0.0 yields inf, which is exactly what we want.
        inv_denom = Float(1.0) / denom

        t_pos_raw = (D_i - p_dot_n) * inv_denom
        t_neg_raw = (-D_i - p_dot_n) * inv_denom

        tn = dr.minimum(t_pos_raw, t_neg_raw)
        tf = dr.maximum(t_pos_raw, t_neg_raw)

        # Hard-override the NaN hazard for perfectly parallel rays
        is_parallel = dr.abs(denom) < 1e-6
        is_inside = dr.abs(p_dot_n) <= (D_i + 1e-5)

        # If parallel and inside -> interval is infinitely open.
        # If parallel and outside -> interval is aggressively empty (tn > tf).
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

        # Ripped off the 1e-5 band-aid. True hits only.
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

        # Fixed the FMA register precision bug. Tolerance is key.
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
import numpy as np
import drjit as dr
from drjit.auto import Float, Bool, Array3f, Complex2f, Matrix3f
from typing import Tuple
from ..base_particle import PrismParticle
from ...models.photon_ray import PhotonRay  # <-- RIP PhasorRay
from ...optics import compute_fresnel_and_scatter

class CuboctahedralParticle(PrismParticle):
    """
    The pure mathematical skeleton of a CO2 cuboctahedron.
    Now running on lightweight PhotonRays because my dev is a troll.
    """

    def __init__(self, a_axis_um: float):
        self.H = Float(a_axis_um / 1000.0)
        self.R = self.H * Float(np.sqrt(2.0)) # Circumscribed radius
        self.D_tri = self.H * Float(2.0 / np.sqrt(3.0))

        self.d0, self.d1, self.d2 = self.H, self.H, self.H
        self.d3, self.d4, self.d5, self.d6 = self.D_tri, self.D_tri, self.D_tri, self.D_tri

        self.n0 = Array3f(1.0, 0.0, 0.0)
        self.n1 = Array3f(0.0, 1.0, 0.0)
        self.n2 = Array3f(0.0, 0.0, 1.0)

        inv_sqrt3 = float(1.0 / np.sqrt(3.0))
        self.n3 = Array3f(1.0, 1.0, 1.0) * inv_sqrt3
        self.n4 = Array3f(-1.0, 1.0, 1.0) * inv_sqrt3
        self.n5 = Array3f(1.0, -1.0, 1.0) * inv_sqrt3
        self.n6 = Array3f(1.0, 1.0, -1.0) * inv_sqrt3

        # Base identity matrices
        self.to_world = Matrix3f(1.0)
        self.to_local = Matrix3f(1.0)

    def _intersect_slab(self, rays: PhotonRay, N_i: Array3f, D_i: Float):
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

    def intersect(self, rays: PhotonRay) -> Tuple[Bool, Float]:
        # 1. Shift incoming lightweight ray to local unrotated space
        local_rays = PhotonRay(
            origin=self.to_local @ rays.origin,
            direction=self.to_local @ rays.direction,
            weight=rays.weight
        )

        # 2. Intersect 7 slabs
        tn0, tf0 = self._intersect_slab(local_rays, self.n0, self.d0)
        tn1, tf1 = self._intersect_slab(local_rays, self.n1, self.d1)
        tn2, tf2 = self._intersect_slab(local_rays, self.n2, self.d2)
        tn3, tf3 = self._intersect_slab(local_rays, self.n3, self.d3)
        tn4, tf4 = self._intersect_slab(local_rays, self.n4, self.d4)
        tn5, tf5 = self._intersect_slab(local_rays, self.n5, self.d5)
        tn6, tf6 = self._intersect_slab(local_rays, self.n6, self.d6)

        t_entry = dr.maximum(dr.maximum(dr.maximum(tn0, tn1), dr.maximum(tn2, tn3)), dr.maximum(dr.maximum(tn4, tn5), tn6))
        t_exit = dr.minimum(dr.minimum(dr.minimum(tf0, tf1), dr.minimum(tf2, tf3)), dr.minimum(dr.minimum(tf4, tf5), tf6))

        hit_mask = (t_entry <= t_exit) & (t_exit > 1e-4) & (t_entry < 1e5)
        is_internal = t_entry < 1e-4
        t = dr.select(is_internal, dr.maximum(t_exit, Float(0.0)), t_entry)

        return hit_mask, dr.select(hit_mask, t, Float(0.0))

    def scatter(self, rays: PhotonRay, hit_mask: Bool, t: Float, ior_water: float = 1.43):
        # 1. Shift incoming ray to local space
        local_rays = PhotonRay(
            origin=self.to_local @ rays.origin,
            direction=self.to_local @ rays.direction,
            weight=rays.weight
        )

        p_local = local_rays.origin + local_rays.direction * t

        # 2. Distance to planes
        dot0, dot1, dot2 = dr.dot(p_local, self.n0) / self.d0, dr.dot(p_local, self.n1) / self.d1, dr.dot(p_local, self.n2) / self.d2
        dot3, dot4 = dr.dot(p_local, self.n3) / self.d3, dr.dot(p_local, self.n4) / self.d4
        dot5, dot6 = dr.dot(p_local, self.n5) / self.d5, dr.dot(p_local, self.n6) / self.d6

        abs0, abs1, abs2 = dr.abs(dot0), dr.abs(dot1), dr.abs(dot2)
        abs3, abs4, abs5, abs6 = dr.abs(dot3), dr.abs(dot4), dr.abs(dot5), dr.abs(dot6)

        max_dot = dr.maximum(dr.maximum(dr.maximum(abs0, abs1), dr.maximum(abs2, abs3)), dr.maximum(dr.maximum(abs4, abs5), abs6))

        n_local = dr.select(
            abs0 >= max_dot * 0.9999, self.n0 * dr.sign(dot0),
            dr.select(abs1 >= max_dot * 0.9999, self.n1 * dr.sign(dot1),
            dr.select(abs2 >= max_dot * 0.9999, self.n2 * dr.sign(dot2),
            dr.select(abs3 >= max_dot * 0.9999, self.n3 * dr.sign(dot3),
            dr.select(abs4 >= max_dot * 0.9999, self.n4 * dr.sign(dot4),
            dr.select(abs5 >= max_dot * 0.9999, self.n5 * dr.sign(dot5),
            self.n6 * dr.sign(dot6)))))))

        # 3. Compute optics
        out = compute_fresnel_and_scatter(local_rays.direction, n_local, 1.0, ior_water)
        d_refl_local, d_refr_local, r_perp, r_para, t_perp, t_para, is_tir = out

        # 4. Shift outgoing geometry back to world space
        d_refl_world = self.to_world @ d_refl_local
        d_refr_world = self.to_world @ d_refr_local
        n_world = self.to_world @ n_local

        return d_refl_world, d_refr_world, r_perp, r_para, t_perp, t_para, n_world, is_tir
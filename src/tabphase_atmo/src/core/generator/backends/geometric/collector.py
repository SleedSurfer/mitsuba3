import drjit as dr
from drjit.auto import Float, Complex2f, Array3f, UInt32
import numpy as np
from typing import Tuple

from .models.phasor_ray import PhasorRay
from .models.ray_patch import RayPatch


class CollectionSphere:
    def __init__(self, mu_bins: np.ndarray, num_phi_bins: int, wavelength_nm: float):
        self.mu_bins_np = mu_bins
        self.num_mu_bins = len(mu_bins)
        self.num_phi_bins = num_phi_bins
        self.total_bins = self.num_mu_bins * self.num_phi_bins

        wavelength_mm = wavelength_nm / 1e6
        self.k = (2.0 * np.pi) / wavelength_mm

        theta = np.arccos(mu_bins)
        dtheta = np.pi / (self.num_mu_bins - 1)
        dphi = (2.0 * np.pi) / self.num_phi_bins

        theta_lo = np.maximum(0, theta - dtheta / 2.0)
        theta_hi = np.minimum(np.pi, theta + dtheta / 2.0)

        self._solid_angles_1d = dphi * (np.cos(theta_lo) - np.cos(theta_hi))
        self._solid_angles_1d = np.maximum(self._solid_angles_1d, 1e-12)

        self._bins_ex_real = None
        self._bins_ex_imag = None
        self._bins_ey_real = None
        self._bins_ey_imag = None

    def reset(self):
        self._bins_ex_real = dr.zeros(Float, self.total_bins)
        self._bins_ex_imag = dr.zeros(Float, self.total_bins)
        self._bins_ey_real = dr.zeros(Float, self.total_bins)
        self._bins_ey_imag = dr.zeros(Float, self.total_bins)

    def _gather_ray_data_full(self, rays: PhasorRay, indices: UInt32):
        d = Array3f(
            dr.gather(Float, rays.direction.x, indices),
            dr.gather(Float, rays.direction.y, indices),
            dr.gather(Float, rays.direction.z, indices)
        )
        l = dr.gather(Float, rays.opt_path_length, indices)
        f = Float(dr.gather(type(rays.focal_lines_crossed), rays.focal_lines_crossed, indices))
        ex = Complex2f(dr.gather(Float, rays.Ex.real, indices), dr.gather(Float, rays.Ex.imag, indices))
        ey = Complex2f(dr.gather(Float, rays.Ey.real, indices), dr.gather(Float, rays.Ey.imag, indices))
        return d, l, f, ex, ey

    def _apply_phasor_rotation(self, ex: Complex2f, ey: Complex2f, l: Float, f: Float):
        phi_unwrapped = Float(self.k) * l - f * Float(np.pi / 2.0)
        two_pi = Float(2.0 * np.pi)
        phi_wrapped = phi_unwrapped - dr.floor(phi_unwrapped / two_pi + Float(0.5)) * two_pi
        rotation = Complex2f(dr.cos(phi_wrapped), dr.sin(phi_wrapped))
        return ex * rotation, ey * rotation

    def _dir_to_continuous_coords(self, d: Array3f) -> Tuple[Float, Float]:
        theta = dr.acos(dr.clip(d.z, Float(-1.0), Float(1.0)))
        theta_coord = (theta / Float(np.pi)) * Float(self.num_mu_bins - 1)

        phi = dr.atan2(d.y, d.x)
        normalized_phi = (phi + Float(np.pi)) / Float(2.0 * np.pi)
        phi_coord = normalized_phi * Float(self.num_phi_bins)

        return phi_coord, theta_coord

    def _barycentric(self, px, py, ax, ay, bx, by, cx, cy):
        """ Computes Barycentric coordinates for a 2D triangle """
        v0x, v0y = cx - ax, cy - ay
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = px - ax, py - ay

        d00 = v0x * v0x + v0y * v0y
        d01 = v0x * v1x + v0y * v1y
        d11 = v1x * v1x + v1y * v1y
        d20 = v2x * v0x + v2y * v0y
        d21 = v2x * v1x + v2y * v1y

        denom = d00 * d11 - d01 * d01
        denom_safe = dr.select(dr.abs(denom) > 1e-8, denom, Float(1e-8))

        v = (d11 * d20 - d01 * d21) / denom_safe
        w = (d00 * d21 - d01 * d20) / denom_safe
        u = Float(1.0) - v - w

        inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0) & (w >= 0.0) & (w <= 1.0) & (dr.abs(denom) > 1e-8)
        return inside, u, v, w

    def _scatter_cplx(self, arr_r, arr_i, val_r, val_i, idx, mask):
        dr.scatter_add(arr_r, val_r, idx, mask)
        dr.scatter_add(arr_i, val_i, idx, mask)

    def accumulate(self, rays: PhasorRay, patches: RayPatch, patch_area: float):
        if self._bins_ex_real is None:
            self.reset()

        d0, l0, f0, ex0, ey0 = self._gather_ray_data_full(rays, patches.v0)
        d1, l1, f1, ex1, ey1 = self._gather_ray_data_full(rays, patches.v1)
        d2, l2, f2, ex2, ey2 = self._gather_ray_data_full(rays, patches.v2)
        d3, l3, f3, ex3, ey3 = self._gather_ray_data_full(rays, patches.v3)

        # 1. STRICT SILHOUETTE CULLING
        # If any corner has exactly zero energy, the patch bridges the physical void. Kill it.
        patch_valid = (dr.abs(ex0.real) > 1e-6) & (dr.abs(ex1.real) > 1e-6) & (dr.abs(ex2.real) > 1e-6) & (
                    dr.abs(ex3.real) > 1e-6)

        cross1 = dr.cross(d1 - d0, d2 - d0)
        cross2 = dr.cross(d3 - d1, d2 - d1)

        # Norm of the cross product is twice the triangle area
        s_i = 0.5 * (dr.norm(cross1) + dr.norm(cross2))

        # Protect against division by zero at the exact caustic singularity
        s_i = dr.maximum(s_i, Float(1e-12))

        # Apply Sadeghi's amplitude scaling factor: sqrt(a_i / s_i)
        amp_scale = dr.sqrt(Float(patch_area) / s_i)

        ex0 *= amp_scale
        ey0 *= amp_scale
        ex1 *= amp_scale
        ey1 *= amp_scale
        ex2 *= amp_scale
        ey2 *= amp_scale
        ex3 *= amp_scale
        ey3 *= amp_scale

        # 2. CONTINUOUS COORDINATES
        phi0, th0 = self._dir_to_continuous_coords(d0)
        phi1, th1 = self._dir_to_continuous_coords(d1)
        phi2, th2 = self._dir_to_continuous_coords(d2)
        phi3, th3 = self._dir_to_continuous_coords(d3)

        # 3. BOUNDING BOX
        min_phi = dr.floor(dr.minimum(dr.minimum(phi0, phi1), dr.minimum(phi2, phi3)))
        max_phi = dr.ceil(dr.maximum(dr.maximum(phi0, phi1), dr.maximum(phi2, phi3)))
        min_th = dr.floor(dr.minimum(dr.minimum(th0, th1), dr.minimum(th2, th3)))
        max_th = dr.ceil(dr.maximum(dr.maximum(th0, th1), dr.maximum(th2, th3)))

        # 4. SOFTWARE RASTERIZATION LOOP (5x5 Maximum Footprint)
        for dx in range(5):
            for dy in range(5):
                target_phi = min_phi + Float(dx)
                target_th = min_th + Float(dy)

                active = (target_phi <= max_phi) & (target_th <= max_th) & patch_valid

                # Split quad into two triangles.
                # T1: v0(top-left), v1(top-right), v2(bottom-left)
                in_t1, u1, v1, w1 = self._barycentric(target_phi, target_th, phi0, th0, phi1, th1, phi2, th2)

                # T2: v3(bottom-right), v2(bottom-left), v1(top-right)
                in_t2, u2, v2, w2 = self._barycentric(target_phi, target_th, phi3, th3, phi2, th2, phi1, th1)

                hit_active = active & (in_t1 | in_t2)

                # 5. EXACT INTERPOLATION
                l_interp = dr.select(in_t1, l0 * u1 + l1 * v1 + l2 * w1, l3 * u2 + l2 * v2 + l1 * w2)
                f_interp = Float(dr.round(dr.select(in_t1, f0 * u1 + f1 * v1 + f2 * w1, f3 * u2 + f2 * v2 + f1 * w2)))

                ex_interp = dr.select(in_t1, ex0 * u1 + ex1 * v1 + ex2 * w1, ex3 * u2 + ex2 * v2 + ex1 * w2)
                ey_interp = dr.select(in_t1, ey0 * u1 + ey1 * v1 + ey2 * w1, ey3 * u2 + ey2 * v2 + ey1 * w2)

                # 6. PHASOR ROTATION AT THE BIN
                px, py = self._apply_phasor_rotation(ex_interp, ey_interp, l_interp, f_interp)

                # 7. ATOMIC SPLAT
                phi_idx = UInt32(target_phi) % self.num_phi_bins
                th_idx = UInt32(dr.clip(target_th, 0.0, self.num_mu_bins - 1))
                flat_idx = phi_idx * self.num_mu_bins + th_idx

                self._scatter_cplx(self._bins_ex_real, self._bins_ex_imag, px.real, px.imag, flat_idx, hit_active)
                self._scatter_cplx(self._bins_ey_real, self._bins_ey_imag, py.real, py.imag, flat_idx, hit_active)

    def finalize(self) -> np.ndarray:
        omega_1d = np.tile(self._solid_angles_1d, self.num_phi_bins)
        omega_dr = Float(omega_1d)

        intensity = (self._bins_ex_real ** 2 + self._bins_ex_imag ** 2 +
                     self._bins_ey_real ** 2 + self._bins_ey_imag ** 2) / omega_dr

        intensity_2d = np.array(intensity).reshape((self.num_phi_bins, self.num_mu_bins))
        intensity_1d = np.mean(intensity_2d, axis=0)

        self.reset()
        return intensity_1d
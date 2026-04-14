import drjit as dr
from drjit.auto import Float, Complex2f, Array3f, UInt32
import numpy as np
from typing import Tuple

from ..models.phasor_ray import PhasorRay
from ..models.ray_patch import RayPatch


class CollectionSphere:
    def __init__(self, theta_bins: np.ndarray, num_phi_bins: int, wavelength_nm: float):
        self.theta_bins = theta_bins
        self.num_theta_bins = len(theta_bins)
        self.num_phi_bins = num_phi_bins
        self.total_bins = self.num_theta_bins * self.num_phi_bins

        wavelength_mm = wavelength_nm / 1e6
        self.k = (2.0 * np.pi) / wavelength_mm
        self.wavelength_nm = wavelength_nm

        # Look how much cleaner this is. No arccos required.
        dtheta = np.pi / (self.num_theta_bins - 1)
        dphi = (2.0 * np.pi) / self.num_phi_bins

        theta_lo = np.maximum(0, self.theta_bins - dtheta / 2.0)
        theta_hi = np.minimum(np.pi, self.theta_bins + dtheta / 2.0)

        # Solid angle calculation remains the same, but now it's honest
        self._solid_angles_1d = dphi * (np.cos(theta_lo) - np.cos(theta_hi))
        self._solid_angles_1d = np.maximum(self._solid_angles_1d, 1e-12)

        self._bins_ex_real = None
        self._bins_ex_imag = None
        self._bins_ey_real = None
        self._bins_ey_imag = None
        self._bins_ez_real = None
        self._bins_ez_imag = None

    def reset(self):
        self._bins_ex_real = dr.zeros(Float, self.total_bins)
        self._bins_ex_imag = dr.zeros(Float, self.total_bins)
        self._bins_ey_real = dr.zeros(Float, self.total_bins)
        self._bins_ey_imag = dr.zeros(Float, self.total_bins)
        self._bins_ez_real = dr.zeros(Float, self.total_bins)
        self._bins_ez_imag = dr.zeros(Float, self.total_bins)

    def _gather_ray_data_full(self, rays: PhasorRay, indices: UInt32):
        d = Array3f(
            dr.gather(Float, rays.direction.x, indices),
            dr.gather(Float, rays.direction.y, indices),
            dr.gather(Float, rays.direction.z, indices)
        )
        bx = Array3f(
            dr.gather(Float, rays.basis_x.x, indices),
            dr.gather(Float, rays.basis_x.y, indices),
            dr.gather(Float, rays.basis_x.z, indices)
        )
        by = Array3f(
            dr.gather(Float, rays.basis_y.x, indices),
            dr.gather(Float, rays.basis_y.y, indices),
            dr.gather(Float, rays.basis_y.z, indices)
        )
        l = dr.gather(Float, rays.opt_path_length, indices)
        f = Float(dr.gather(type(rays.focal_lines_crossed), rays.focal_lines_crossed, indices))
        ex = Complex2f(dr.gather(Float, rays.Ex.real, indices), dr.gather(Float, rays.Ex.imag, indices))
        ey = Complex2f(dr.gather(Float, rays.Ey.real, indices), dr.gather(Float, rays.Ey.imag, indices))
        return d, bx, by, l, f, ex, ey

    def _to_3d_complex(self, ex: Complex2f, ey: Complex2f, bx: Array3f, by: Array3f):
        """ Projects the local 2D wave amplitudes back into the global 3D vector space """
        c_x = Complex2f(ex.real * bx.x + ey.real * by.x, ex.imag * bx.x + ey.imag * by.x)
        c_y = Complex2f(ex.real * bx.y + ey.real * by.y, ex.imag * bx.y + ey.imag * by.y)
        c_z = Complex2f(ex.real * bx.z + ey.real * by.z, ex.imag * bx.z + ey.imag * by.z)
        return c_x, c_y, c_z

    def _apply_3d_phasor_rotation(self, Ex: Complex2f, Ey: Complex2f, Ez: Complex2f, l: Float, f: Float):
        # [cite_start]Phase advance per focal line matched to Sadeghi et al. [cite: 1]
        phi_unwrapped = Float(self.k) * l + f * Float(np.pi / 2.0)
        two_pi = Float(2.0 * np.pi)
        phi_wrapped = phi_unwrapped - dr.floor(phi_unwrapped / two_pi + Float(0.5)) * two_pi
        rotation = Complex2f(dr.cos(phi_wrapped), dr.sin(phi_wrapped))
        return Ex * rotation, Ey * rotation, Ez * rotation

    def _dir_to_continuous_coords(self, d: Array3f) -> Tuple[Float, Float]:
        theta = dr.acos(dr.clip(d.z, Float(-1.0), Float(1.0)))
        theta_coord = (theta / Float(np.pi)) * Float(self.num_theta_bins - 1)

        phi = dr.atan2(Float(d.y), Float(d.x))
        normalized_phi = (phi + Float(np.pi)) / Float(2.0 * np.pi)
        phi_coord = normalized_phi * Float(self.num_phi_bins)
        return phi_coord, theta_coord

    def _barycentric(self, px, py, ax, ay, bx, by, cx, cy):
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

    @dr.syntax
    def accumulate(self, rays: PhasorRay, patches: RayPatch, patch_area: float):
        if self._bins_ex_real is None:
            self.reset()

        d0, bx0, by0, l0, f0, ex0, ey0 = self._gather_ray_data_full(rays, patches.v0)
        d1, bx1, by1, l1, f1, ex1, ey1 = self._gather_ray_data_full(rays, patches.v1)
        d2, bx2, by2, l2, f2, ex2, ey2 = self._gather_ray_data_full(rays, patches.v2)
        d3, bx3, by3, l3, f3, ex3, ey3 = self._gather_ray_data_full(rays, patches.v3)

        patch_valid = (dr.abs(ex0.real) > 1e-6) & (dr.abs(ex1.real) > 1e-6) & \
                      (dr.abs(ex2.real) > 1e-6) & (dr.abs(ex3.real) > 1e-6)

        # Standard geometric focusing, no weird wavelength clamps
        cross1 = dr.cross(d1 - d0, d2 - d0)
        cross2 = dr.cross(d3 - d1, d2 - d1)
        s_i = dr.maximum(0.5 * (dr.norm(cross1) + dr.norm(cross2)), Float(1e-12))
        amp_scale = dr.sqrt(Float(patch_area) / s_i)

        E0_x, E0_y, E0_z = self._to_3d_complex(ex0 * amp_scale, ey0 * amp_scale, bx0, by0)
        E1_x, E1_y, E1_z = self._to_3d_complex(ex1 * amp_scale, ey1 * amp_scale, bx1, by1)
        E2_x, E2_y, E2_z = self._to_3d_complex(ex2 * amp_scale, ey2 * amp_scale, bx2, by2)
        E3_x, E3_y, E3_z = self._to_3d_complex(ex3 * amp_scale, ey3 * amp_scale, bx3, by3)

        phi0, th0 = self._dir_to_continuous_coords(d0)
        phi1, th1 = self._dir_to_continuous_coords(d1)
        phi2, th2 = self._dir_to_continuous_coords(d2)
        phi3, th3 = self._dir_to_continuous_coords(d3)

        # Azimuthal Wrapping Safety
        half_phi = Float(self.num_phi_bins / 2.0)
        max_phi_val = Float(self.num_phi_bins)
        phi1 = dr.select(phi1 - phi0 > half_phi, phi1 - max_phi_val,
                         dr.select(phi1 - phi0 < -half_phi, phi1 + max_phi_val, phi1))
        phi2 = dr.select(phi2 - phi0 > half_phi, phi2 - max_phi_val,
                         dr.select(phi2 - phi0 < -half_phi, phi2 + max_phi_val, phi2))
        phi3 = dr.select(phi3 - phi0 > half_phi, phi3 - max_phi_val,
                         dr.select(phi3 - phi0 < -half_phi, phi3 + max_phi_val, phi3))

        min_phi = dr.floor(dr.minimum(dr.minimum(phi0, phi1), dr.minimum(phi2, phi3)))
        max_phi = dr.ceil(dr.maximum(dr.maximum(phi0, phi1), dr.maximum(phi2, phi3)))
        min_th = dr.floor(dr.minimum(dr.minimum(th0, th1), dr.minimum(th2, th3)))
        max_th = dr.ceil(dr.maximum(dr.maximum(th0, th1), dr.maximum(th2, th3)))

        w = dr.maximum(max_phi - min_phi + Float(1.0), Float(1.0))
        h = dr.maximum(max_th - min_th + Float(1.0), Float(1.0))
        max_idx = UInt32(dr.minimum(w * h, Float(256)))

        # Precompute Determinants
        v0x_t1, v0y_t1 = phi1 - phi0, th1 - th0
        v1x_t1, v1y_t1 = phi2 - phi0, th2 - th0
        d00_t1, d01_t1, d11_t1 = v0x_t1 ** 2 + v0y_t1 ** 2, v0x_t1 * v1x_t1 + v0y_t1 * v1y_t1, v1x_t1 ** 2 + v1y_t1 ** 2
        denom_t1 = d00_t1 * d11_t1 - d01_t1 * d01_t1
        inv_denom_t1 = Float(1.0) / dr.select(dr.abs(denom_t1) > 1e-8, denom_t1, Float(1e-8))

        v0x_t2, v0y_t2 = phi2 - phi3, th2 - th3
        v1x_t2, v1y_t2 = phi1 - phi3, th1 - th3
        d00_t2, d01_t2, d11_t2 = v0x_t2 ** 2 + v0y_t2 ** 2, v0x_t2 * v1x_t2 + v0y_t2 * v1y_t2, v1x_t2 ** 2 + v1y_t2 ** 2
        denom_t2 = d00_t2 * d11_t2 - d01_t2 * d01_t2
        inv_denom_t2 = Float(1.0) / dr.select(dr.abs(denom_t2) > 1e-8, denom_t2, Float(1e-8))

        idx = dr.zeros(UInt32, dr.width(min_phi))
        dx = dr.zeros(Float, dr.width(min_phi))
        dy = dr.zeros(Float, dr.width(min_phi))

        while idx < max_idx:
            target_phi = min_phi + dx
            target_th = min_th + dy
            active = (target_phi <= max_phi) & (target_th <= max_th) & patch_valid

            # Ruthless Barycentric T1
            v2x_t1, v2y_t1 = target_phi - phi0, target_th - th0
            d20_t1, d21_t1 = v2x_t1 * v0x_t1 + v2y_t1 * v0y_t1, v2x_t1 * v1x_t1 + v2y_t1 * v1y_t1
            v1 = (d11_t1 * d20_t1 - d01_t1 * d21_t1) * inv_denom_t1
            w1_b = (d00_t1 * d21_t1 - d01_t1 * d20_t1) * inv_denom_t1
            u1 = Float(1.0) - v1 - w1_b
            in_t1 = (u1 >= 0.0) & (u1 <= 1.0) & (v1 >= 0.0) & (v1 <= 1.0) & (w1_b >= 0.0) & (w1_b <= 1.0) & (
                        dr.abs(denom_t1) > 1e-8)

            v2x_t2, v2y_t2 = target_phi - phi3, target_th - th3
            d20_t2, d21_t2 = v2x_t2 * v0x_t2 + v2y_t2 * v0y_t2, v2x_t2 * v1x_t2 + v2y_t2 * v1y_t2
            v2 = (d11_t2 * d20_t2 - d01_t2 * d21_t2) * inv_denom_t2
            w2_b = (d00_t2 * d21_t2 - d01_t2 * d20_t2) * inv_denom_t2
            u2 = Float(1.0) - v2 - w2_b
            in_t2 = (u2 >= 0.0) & (u2 <= 1.0) & (v2 >= 0.0) & (v2 <= 1.0) & (w2_b >= 0.0) & (w2_b <= 1.0) & (
                        dr.abs(denom_t2) > 1e-8)

            phi_idx = UInt32(target_phi + max_phi_val) % self.num_phi_bins
            th_idx = UInt32(dr.clip(target_th, 0.0, self.num_theta_bins - 1))
            flat_idx = phi_idx * self.num_theta_bins + th_idx

            l1_val = l0 * u1 + l1 * v1 + l2 * w1_b
            ex1_x, ex1_y, ex1_z = E0_x * u1 + E1_x * v1 + E2_x * w1_b, E0_y * u1 + E1_y * v1 + E2_y * w1_b, E0_z * u1 + E1_z * v1 + E2_z * w1_b
            px1, py1, pz1 = self._apply_3d_phasor_rotation(ex1_x, ex1_y, ex1_z, l1_val, f0)

            l2_val = l3 * u2 + l2 * v2 + l1 * w2_b
            ex2_x, ex2_y, ex2_z = E3_x * u2 + E2_x * v2 + E1_x * w2_b, E3_y * u2 + E2_y * v2 + E1_y * w2_b, E3_z * u2 + E2_z * v2 + E1_z * w2_b
            px2, py2, pz2 = self._apply_3d_phasor_rotation(ex2_x, ex2_y, ex2_z, l2_val, f0)

            valid_t1 = active & in_t1
            valid_t2 = active & in_t2

            px_final = dr.select(valid_t1, px1, Complex2f(0.0)) + dr.select(valid_t2, px2, Complex2f(0.0))
            py_final = dr.select(valid_t1, py1, Complex2f(0.0)) + dr.select(valid_t2, py2, Complex2f(0.0))
            pz_final = dr.select(valid_t1, pz1, Complex2f(0.0)) + dr.select(valid_t2, pz2, Complex2f(0.0))

            hit_any = valid_t1 | valid_t2

            self._scatter_cplx(self._bins_ex_real, self._bins_ex_imag, px_final.real, px_final.imag, flat_idx, hit_any)
            self._scatter_cplx(self._bins_ey_real, self._bins_ey_imag, py_final.real, py_final.imag, flat_idx, hit_any)
            self._scatter_cplx(self._bins_ez_real, self._bins_ez_imag, pz_final.real, pz_final.imag, flat_idx, hit_any)

            dx += Float(1.0)
            wrap_mask = dx >= w
            dx = dr.select(wrap_mask, Float(0.0), dx)
            dy = dr.select(wrap_mask, dy + Float(1.0), dy)
            idx += 1

    def finalize(self) -> np.ndarray:
        intensity = (self._bins_ex_real ** 2 + self._bins_ex_imag ** 2 +
                     self._bins_ey_real ** 2 + self._bins_ey_imag ** 2 +
                     self._bins_ez_real ** 2 + self._bins_ez_imag ** 2)

        density_2d = np.array(intensity).reshape((self.num_phi_bins, self.num_theta_bins))

        self.reset()

        if self.num_phi_bins == 1:
            return np.mean(density_2d, axis=0)

        return density_2d
import drjit as dr
from drjit.llvm.ad import Float, Complex2f, Array3f, UInt32
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

        dtheta = np.pi / (self.num_theta_bins - 1)
        dphi = (2.0 * np.pi) / self.num_phi_bins

        theta_lo = np.maximum(0, self.theta_bins - dtheta / 2.0)
        theta_hi = np.minimum(np.pi, self.theta_bins + dtheta / 2.0)

        self._solid_angles_1d = dphi * (np.cos(theta_lo) - np.cos(theta_hi))
        self._solid_angles_1d = np.maximum(self._solid_angles_1d, 1e-12)

        # The dual bin setup
        self._bins_ex_real_X = None
        self._bins_ex_imag_X = None
        self._bins_ey_real_X = None
        self._bins_ey_imag_X = None
        self._bins_ez_real_X = None
        self._bins_ez_imag_X = None

        self._bins_ex_real_Y = None
        self._bins_ex_imag_Y = None
        self._bins_ey_real_Y = None
        self._bins_ey_imag_Y = None
        self._bins_ez_real_Y = None
        self._bins_ez_imag_Y = None

    def reset(self):
        self._bins_ex_real_X = dr.zeros(Float, self.total_bins)
        self._bins_ex_imag_X = dr.zeros(Float, self.total_bins)
        self._bins_ey_real_X = dr.zeros(Float, self.total_bins)
        self._bins_ey_imag_X = dr.zeros(Float, self.total_bins)
        self._bins_ez_real_X = dr.zeros(Float, self.total_bins)
        self._bins_ez_imag_X = dr.zeros(Float, self.total_bins)

        self._bins_ex_real_Y = dr.zeros(Float, self.total_bins)
        self._bins_ex_imag_Y = dr.zeros(Float, self.total_bins)
        self._bins_ey_real_Y = dr.zeros(Float, self.total_bins)
        self._bins_ey_imag_Y = dr.zeros(Float, self.total_bins)
        self._bins_ez_real_Y = dr.zeros(Float, self.total_bins)
        self._bins_ez_imag_Y = dr.zeros(Float, self.total_bins)

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

        ex_X = Complex2f(dr.gather(Float, rays.Ex_X.real, indices), dr.gather(Float, rays.Ex_X.imag, indices))
        ey_X = Complex2f(dr.gather(Float, rays.Ey_X.real, indices), dr.gather(Float, rays.Ey_X.imag, indices))
        ex_Y = Complex2f(dr.gather(Float, rays.Ex_Y.real, indices), dr.gather(Float, rays.Ex_Y.imag, indices))
        ey_Y = Complex2f(dr.gather(Float, rays.Ey_Y.real, indices), dr.gather(Float, rays.Ey_Y.imag, indices))

        return d, bx, by, l, f, ex_X, ey_X, ex_Y, ey_Y

    def _to_3d_complex(self, ex: Complex2f, ey: Complex2f, bx: Array3f, by: Array3f):
        c_x = Complex2f(ex.real * bx.x + ey.real * by.x, ex.imag * bx.x + ey.imag * by.x)
        c_y = Complex2f(ex.real * bx.y + ey.real * by.y, ex.imag * bx.y + ey.imag * by.y)
        c_z = Complex2f(ex.real * bx.z + ey.real * by.z, ex.imag * bx.z + ey.imag * by.z)
        return c_x, c_y, c_z

    def _apply_3d_phasor_rotation(self, Ex: Complex2f, Ey: Complex2f, Ez: Complex2f, l: Float, f: Float):
        phi_unwrapped = Float(self.k) * l - f * Float(np.pi / 2.0)
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

    def _scatter_cplx(self, arr_r, arr_i, val_r, val_i, idx, mask):
        dr.scatter_add(arr_r, val_r, idx, mask)
        dr.scatter_add(arr_i, val_i, idx, mask)

    @dr.syntax
    def accumulate(self, rays: PhasorRay, patches: RayPatch, patch_area: float):
        if self._bins_ex_real_X is None:
            self.reset()

        d0, bx0, by0, l0, f0, ex0_X, ey0_X, ex0_Y, ey0_Y = self._gather_ray_data_full(rays, patches.v0)
        d1, bx1, by1, l1, f1, ex1_X, ey1_X, ex1_Y, ey1_Y = self._gather_ray_data_full(rays, patches.v1)
        d2, bx2, by2, l2, f2, ex2_X, ey2_X, ex2_Y, ey2_Y = self._gather_ray_data_full(rays, patches.v2)
        d3, bx3, by3, l3, f3, ex3_X, ey3_X, ex3_Y, ey3_Y = self._gather_ray_data_full(rays, patches.v3)

        v0_valid = (dr.abs(ex0_X.real) > 1e-6) | (dr.abs(ex0_Y.real) > 1e-6)
        v1_valid = (dr.abs(ex1_X.real) > 1e-6) | (dr.abs(ex1_Y.real) > 1e-6)
        v2_valid = (dr.abs(ex2_X.real) > 1e-6) | (dr.abs(ex2_Y.real) > 1e-6)
        v3_valid = (dr.abs(ex3_X.real) > 1e-6) | (dr.abs(ex3_Y.real) > 1e-6)

        patch_valid = v0_valid & v1_valid & v2_valid & v3_valid

        cross1 = dr.cross(d1 - d0, d2 - d0)
        cross2 = dr.cross(d3 - d1, d2 - d1)
        s_i = dr.maximum(0.5 * (dr.norm(cross1) + dr.norm(cross2)), Float(1e-12))
        amp_scale = dr.sqrt(Float(patch_area) / s_i)

        # To 3D for X Source
        E0_x_X, E0_y_X, E0_z_X = self._to_3d_complex(ex0_X * amp_scale, ey0_X * amp_scale, bx0, by0)
        E1_x_X, E1_y_X, E1_z_X = self._to_3d_complex(ex1_X * amp_scale, ey1_X * amp_scale, bx1, by1)
        E2_x_X, E2_y_X, E2_z_X = self._to_3d_complex(ex2_X * amp_scale, ey2_X * amp_scale, bx2, by2)
        E3_x_X, E3_y_X, E3_z_X = self._to_3d_complex(ex3_X * amp_scale, ey3_X * amp_scale, bx3, by3)

        # To 3D for Y Source
        E0_x_Y, E0_y_Y, E0_z_Y = self._to_3d_complex(ex0_Y * amp_scale, ey0_Y * amp_scale, bx0, by0)
        E1_x_Y, E1_y_Y, E1_z_Y = self._to_3d_complex(ex1_Y * amp_scale, ey1_Y * amp_scale, bx1, by1)
        E2_x_Y, E2_y_Y, E2_z_Y = self._to_3d_complex(ex2_Y * amp_scale, ey2_Y * amp_scale, bx2, by2)
        E3_x_Y, E3_y_Y, E3_z_Y = self._to_3d_complex(ex3_Y * amp_scale, ey3_Y * amp_scale, bx3, by3)

        phi0, th0 = self._dir_to_continuous_coords(d0)
        phi1, th1 = self._dir_to_continuous_coords(d1)
        phi2, th2 = self._dir_to_continuous_coords(d2)
        phi3, th3 = self._dir_to_continuous_coords(d3)

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

            valid_t1 = active & in_t1
            valid_t2 = active & in_t2
            hit_any = valid_t1 | valid_t2

            l1_val = l0 * u1 + l1 * v1 + l2 * w1_b
            l2_val = l3 * u2 + l2 * v2 + l1 * w2_b

            # X Polarization Math
            ex1_x_X, ex1_y_X, ex1_z_X = E0_x_X * u1 + E1_x_X * v1 + E2_x_X * w1_b, E0_y_X * u1 + E1_y_X * v1 + E2_y_X * w1_b, E0_z_X * u1 + E1_z_X * v1 + E2_z_X * w1_b
            px1_X, py1_X, pz1_X = self._apply_3d_phasor_rotation(ex1_x_X, ex1_y_X, ex1_z_X, l1_val, f0)
            ex2_x_X, ex2_y_X, ex2_z_X = E3_x_X * u2 + E2_x_X * v2 + E1_x_X * w2_b, E3_y_X * u2 + E2_y_X * v2 + E1_y_X * w2_b, E3_z_X * u2 + E2_z_X * v2 + E1_z_X * w2_b
            px2_X, py2_X, pz2_X = self._apply_3d_phasor_rotation(ex2_x_X, ex2_y_X, ex2_z_X, l2_val, f0)

            px_final_X = dr.select(valid_t1, px1_X, Complex2f(0.0)) + dr.select(valid_t2, px2_X, Complex2f(0.0))
            py_final_X = dr.select(valid_t1, py1_X, Complex2f(0.0)) + dr.select(valid_t2, py2_X, Complex2f(0.0))
            pz_final_X = dr.select(valid_t1, pz1_X, Complex2f(0.0)) + dr.select(valid_t2, pz2_X, Complex2f(0.0))

            self._scatter_cplx(self._bins_ex_real_X, self._bins_ex_imag_X, px_final_X.real, px_final_X.imag, flat_idx,
                               hit_any)
            self._scatter_cplx(self._bins_ey_real_X, self._bins_ey_imag_X, py_final_X.real, py_final_X.imag, flat_idx,
                               hit_any)
            self._scatter_cplx(self._bins_ez_real_X, self._bins_ez_imag_X, pz_final_X.real, pz_final_X.imag, flat_idx,
                               hit_any)

            # Y Polarization Math
            ex1_x_Y, ex1_y_Y, ex1_z_Y = E0_x_Y * u1 + E1_x_Y * v1 + E2_x_Y * w1_b, E0_y_Y * u1 + E1_y_Y * v1 + E2_y_Y * w1_b, E0_z_Y * u1 + E1_z_Y * v1 + E2_z_Y * w1_b
            px1_Y, py1_Y, pz1_Y = self._apply_3d_phasor_rotation(ex1_x_Y, ex1_y_Y, ex1_z_Y, l1_val, f0)
            ex2_x_Y, ex2_y_Y, ex2_z_Y = E3_x_Y * u2 + E2_x_Y * v2 + E1_x_Y * w2_b, E3_y_Y * u2 + E2_y_Y * v2 + E1_y_Y * w2_b, E3_z_Y * u2 + E2_z_Y * v2 + E1_z_Y * w2_b
            px2_Y, py2_Y, pz2_Y = self._apply_3d_phasor_rotation(ex2_x_Y, ex2_y_Y, ex2_z_Y, l2_val, f0)

            px_final_Y = dr.select(valid_t1, px1_Y, Complex2f(0.0)) + dr.select(valid_t2, px2_Y, Complex2f(0.0))
            py_final_Y = dr.select(valid_t1, py1_Y, Complex2f(0.0)) + dr.select(valid_t2, py2_Y, Complex2f(0.0))
            pz_final_Y = dr.select(valid_t1, pz1_Y, Complex2f(0.0)) + dr.select(valid_t2, pz2_Y, Complex2f(0.0))

            self._scatter_cplx(self._bins_ex_real_Y, self._bins_ex_imag_Y, px_final_Y.real, px_final_Y.imag, flat_idx,
                               hit_any)
            self._scatter_cplx(self._bins_ey_real_Y, self._bins_ey_imag_Y, py_final_Y.real, py_final_Y.imag, flat_idx,
                               hit_any)
            self._scatter_cplx(self._bins_ez_real_Y, self._bins_ez_imag_Y, pz_final_Y.real, pz_final_Y.imag, flat_idx,
                               hit_any)

            dx += Float(1.0)
            wrap_mask = dx >= w
            dx = dr.select(wrap_mask, Float(0.0), dx)
            dy = dr.select(wrap_mask, dy + Float(1.0), dy)
            idx += 1

    def finalize(self) -> np.ndarray:
        intensity_X = (self._bins_ex_real_X ** 2 + self._bins_ex_imag_X ** 2 +
                       self._bins_ey_real_X ** 2 + self._bins_ey_imag_X ** 2 +
                       self._bins_ez_real_X ** 2 + self._bins_ez_imag_X ** 2)

        intensity_Y = (self._bins_ex_real_Y ** 2 + self._bins_ex_imag_Y ** 2 +
                       self._bins_ey_real_Y ** 2 + self._bins_ey_imag_Y ** 2 +
                       self._bins_ez_real_Y ** 2 + self._bins_ez_imag_Y ** 2)

        total_intensity = (intensity_X + intensity_Y) / 2.0

        density_2d = np.array(total_intensity).reshape((self.num_phi_bins, self.num_theta_bins))

        self.reset()

        if self.num_phi_bins == 1:
            return np.mean(density_2d, axis=0)

        return density_2d
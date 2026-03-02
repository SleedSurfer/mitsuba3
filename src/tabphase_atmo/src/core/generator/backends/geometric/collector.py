import drjit as dr
from drjit.auto import Float, Bool, Complex2f, Array3f, UInt32
import numpy as np
from typing import Tuple

from .models.phasor_ray import PhasorRay
from .models.ray_patch import RayPatch


class CollectionSphere:
    """
    Collects exiting wavefront patches into a 2D grid (theta, phi).
    Uses the paper's exact amplitude density scaling sqrt(a_i / s_i)
    to account for wavefront divergence and caustics.
    """

    def __init__(self, mu_bins: np.ndarray, num_phi_bins: int, wavelength_nm: float):
        self.mu_bins_np = mu_bins
        self.num_mu_bins = len(mu_bins)
        self.num_phi_bins = num_phi_bins
        self.total_bins = self.num_mu_bins * self.num_phi_bins
        self._bins_power = None
        self._total_patches = 0

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
        self._total_patches = 0

    def reset(self):
        self._bins_power = dr.zeros(Float, self.total_bins)
        self._total_patches = 0

        self._bins_ex_real = dr.zeros(Float, self.total_bins)
        self._bins_ex_imag = dr.zeros(Float, self.total_bins)
        self._bins_ey_real = dr.zeros(Float, self.total_bins)
        self._bins_ey_imag = dr.zeros(Float, self.total_bins)
        self._total_patches = 0

    def _gather_ray_data_full(self, rays: PhasorRay, indices: UInt32):
        d = Array3f(
            dr.gather(Float, rays.d.x, indices),
            dr.gather(Float, rays.d.y, indices),
            dr.gather(Float, rays.d.z, indices)
        )
        l = dr.gather(Float, rays.l, indices)
        f = Float(dr.gather(type(rays.f), rays.f, indices))

        ex = Complex2f(
            dr.gather(Float, rays.Ex.real, indices),
            dr.gather(Float, rays.Ex.imag, indices)
        )
        ey = Complex2f(
            dr.gather(Float, rays.Ey.real, indices),
            dr.gather(Float, rays.Ey.imag, indices)
        )
        return d, l, f, ex, ey

    def _apply_phasor_rotation(self, ex: Complex2f, ey: Complex2f, l: Float, f: Float):
        phi_unwrapped = Float(self.k) * l - f * Float(np.pi / 2.0)
        two_pi = Float(2.0 * np.pi)
        phi_wrapped = phi_unwrapped - dr.floor(phi_unwrapped / two_pi + Float(0.5)) * two_pi
        rotation = Complex2f(dr.cos(phi_wrapped), dr.sin(phi_wrapped))
        return ex * rotation, ey * rotation

    def _dir_to_continuous_coords(self, d: Array3f) -> Tuple[Float, Float]:
        # 1. Continuous Theta (Polar) - Strictly Linear!
        theta = dr.acos(dr.clip(d.z, Float(-1.0), Float(1.0)))
        theta_coord = (theta / Float(np.pi)) * Float(self.num_mu_bins - 1)

        # 2. Continuous Phi (Azimuthal)
        phi = dr.atan2(d.y, d.x)
        normalized_phi = (phi + Float(np.pi)) / Float(2.0 * np.pi)
        phi_coord = normalized_phi * Float(self.num_phi_bins)

        return phi_coord, theta_coord

    def _dir_to_bin_index(self, d: Array3f) -> UInt32:
        mu = dr.clip(d.z, Float(-1.0), Float(1.0))
        normalized_mu = (Float(1.0) - mu) / Float(2.0)
        mu_idx = UInt32(dr.clip(normalized_mu * Float(self.num_mu_bins - 1), Float(0.0), Float(self.num_mu_bins - 1)))

        phi = dr.atan2(d.y, d.x)
        normalized_phi = (phi + Float(np.pi)) / Float(2.0 * np.pi)
        phi_idx = UInt32(
            dr.clip(normalized_phi * Float(self.num_phi_bins - 1), Float(0.0), Float(self.num_phi_bins - 1)))

        return phi_idx * self.num_mu_bins + mu_idx

    def accumulate(self, rays: PhasorRay, patches: RayPatch, patch_area: float):
        if self._bins_ex_real is None:
            self.reset()

        self._total_patches += len(patches.v0)

        d0, l0, f0, ex0, ey0 = self._gather_ray_data_full(rays, patches.v0)
        d1, l1, f1, ex1, ey1 = self._gather_ray_data_full(rays, patches.v1)
        d2, l2, f2, ex2, ey2 = self._gather_ray_data_full(rays, patches.v2)
        d3, l3, f3, ex3, ey3 = self._gather_ray_data_full(rays, patches.v3)

        d_avg = dr.normalize(d0 + d1 + d2 + d3)

        # 1. Caustic Tracking (We still need the cross product for the sign flip!)
        diag1 = d3 - d0
        diag2 = d2 - d1
        cross_area = dr.cross(diag1, diag2)
        signed_area = dr.dot(cross_area, d_avg)

        # If the wavefront folded, add a focal crossing (Gouy phase shift)
        caustic_penalty = dr.select(signed_area < 0.0, Float(1.0), Float(0.0))

        # 2. Phase Rotations
        px0, py0 = self._apply_phasor_rotation(ex0, ey0, l0, f0 + caustic_penalty)
        px1, py1 = self._apply_phasor_rotation(ex1, ey1, l1, f1 + caustic_penalty)
        px2, py2 = self._apply_phasor_rotation(ex2, ey2, l2, f2 + caustic_penalty)
        px3, py3 = self._apply_phasor_rotation(ex3, ey3, l3, f3 + caustic_penalty)

        # 3. THE FIX: Pure Area Weighting (No s_i division!)
        # We just weight the complex amplitude by the physical emitted area
        patch_weight = dr.sqrt(Float(patch_area * 0.25))

        ex_avg = (px0 + px1 + px2 + px3) * patch_weight
        ey_avg = (py0 + py1 + py2 + py3) * patch_weight

        # 4. Bilinear Splatting Coordinates
        phi_c, theta_c = self._dir_to_continuous_coords(d_avg)
        phi_0 = dr.floor(phi_c)
        theta_0 = dr.floor(theta_c)

        w_phi = phi_c - phi_0
        w_theta = theta_c - theta_0

        p0_idx = UInt32(phi_0) % self.num_phi_bins
        p1_idx = (p0_idx + 1) % self.num_phi_bins
        t0_idx = UInt32(dr.clip(theta_0, Float(0.0), Float(self.num_mu_bins - 1)))
        t1_idx = UInt32(dr.clip(theta_0 + Float(1.0), Float(0.0), Float(self.num_mu_bins - 1)))

        idx_00 = p0_idx * self.num_mu_bins + t0_idx
        idx_10 = p1_idx * self.num_mu_bins + t0_idx
        idx_01 = p0_idx * self.num_mu_bins + t1_idx
        idx_11 = p1_idx * self.num_mu_bins + t1_idx

        w00 = (Float(1.0) - w_phi) * (Float(1.0) - w_theta)
        w10 = w_phi * (Float(1.0) - w_theta)
        w01 = (Float(1.0) - w_phi) * w_theta
        w11 = w_phi * w_theta

        valid = dr.isfinite(d_avg.z)

        # 5. Splat the complex amplitudes bilinearly
        def scatter_cplx(arr_r, arr_i, val_r, val_i, w, idx, mask):
            dr.scatter_add(arr_r, val_r * w, idx, mask)
            dr.scatter_add(arr_i, val_i * w, idx, mask)

        scatter_cplx(self._bins_ex_real, self._bins_ex_imag, ex_avg.real, ex_avg.imag, w00, idx_00, valid)
        scatter_cplx(self._bins_ex_real, self._bins_ex_imag, ex_avg.real, ex_avg.imag, w10, idx_10, valid)
        scatter_cplx(self._bins_ex_real, self._bins_ex_imag, ex_avg.real, ex_avg.imag, w01, idx_01, valid)
        scatter_cplx(self._bins_ex_real, self._bins_ex_imag, ex_avg.real, ex_avg.imag, w11, idx_11, valid)

        # Do the same for Ey
        scatter_cplx(self._bins_ey_real, self._bins_ey_imag, ey_avg.real, ey_avg.imag, w00, idx_00, valid)
        scatter_cplx(self._bins_ey_real, self._bins_ey_imag, ey_avg.real, ey_avg.imag, w10, idx_10, valid)
        scatter_cplx(self._bins_ey_real, self._bins_ey_imag, ey_avg.real, ey_avg.imag, w01, idx_01, valid)
        scatter_cplx(self._bins_ey_real, self._bins_ey_imag, ey_avg.real, ey_avg.imag, w11, idx_11, valid)

    def finalize(self) -> np.ndarray:
        omega_1d = np.tile(self._solid_angles_1d, self.num_phi_bins)
        omega_dr = Float(omega_1d)

        # THE FIX: Square first, THEN divide by solid angle
        intensity = (self._bins_ex_real ** 2 + self._bins_ex_imag ** 2 +
                     self._bins_ey_real ** 2 + self._bins_ey_imag ** 2) / omega_dr

        intensity_2d = np.array(intensity).reshape((self.num_phi_bins, self.num_mu_bins))
        intensity_1d = np.mean(intensity_2d, axis=0)

        self.reset()
        return intensity_1d
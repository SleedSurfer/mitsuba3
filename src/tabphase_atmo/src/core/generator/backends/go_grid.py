import numpy as np
import drjit as dr
from drjit.auto import Float, Bool, Complex2f, Array3f
import copy

# Make the cache spam fuck off
dr.set_log_level(dr.LogLevel.Error)

from src.core.config import MieConfig as config
from .base import ScatteringBackend
from .geometric.particles.sphere import SphericalParticle
# IMPORT THE CHUANGUS
from .geometric.particles.oblate_sphere import OblateSpheroidParticle
from .geometric.particles.hexagonal_ice import HexagonalCrystalParticle
from .geometric.optics import apply_basis_rotation
from .geometric.ray_emitter import GridEmitter
from .geometric.collector import CollectionSphere
from .geometric.postprocess import apply_diffraction_smoothing

class DrJitRaytracerBackend(ScatteringBackend):
    name: str = "drjit_raytracer"

    def __init__(self, grid_res: int = 950, num_batches: int = 10, num_phi_bins: int = 1, particle_shape: str = "sphere"):
        self.grid_res = grid_res
        self.num_batches = num_batches
        self.num_phi_bins = num_phi_bins
        self.particle_shape = particle_shape

    def _trace_batch(self, active_rays, patches, logical_x, logical_y, particle, ior_real_dr, ior_inv_dr):
        # Capture initial coords for the 2D gradient
        x_init = active_rays.origin.x
        y_init = active_rays.origin.y

        # SCRUBBED THE USELESS POLAR CONVERSION MATH FROM HERE

        exiting_wavefronts = []
        thetas_to_eval = []

        # ==========================================
        # BOUNCE 0
        # ==========================================
        hit_mask, t = particle.intersect(active_rays)
        safe_t = dr.select(hit_mask, t, Float(0.0))

        active_rays.Ex *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
        active_rays.Ey *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
        active_rays.opt_path_length += safe_t * 1.0

        out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=ior_real_dr)
        d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = out

        surface_o = active_rays.origin + active_rays.direction * safe_t

        # Branch A: Exit (p=0)
        E_x_refl, E_y_refl, b_x_refl, b_y_refl = apply_basis_rotation(
            active_rays.Ex, active_rays.Ey, active_rays.basis_x, active_rays.basis_y,
            active_rays.direction, normals, d_refl, r_perp, r_para
        )

        p0_rays = copy.copy(active_rays)
        p0_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        p0_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
        p0_rays.Ex = E_x_refl
        p0_rays.Ey = E_y_refl
        p0_rays.basis_x = b_x_refl
        p0_rays.basis_y = b_y_refl

        far_field_l = active_rays.opt_path_length - dr.dot(p0_rays.origin, p0_rays.direction)
        p0_rays.opt_path_length = far_field_l

        theta_out_0 = dr.acos(dr.clip(p0_rays.direction.z, Float(-1.0), Float(1.0)))
        thetas_to_eval.append(theta_out_0)
        exiting_wavefronts.append((p0_rays, patches, 0, theta_out_0))

        # Branch B: Refract IN
        E_x_refr, E_y_refr, b_x_refr, b_y_refr = apply_basis_rotation(
            active_rays.Ex, active_rays.Ey, active_rays.basis_x, active_rays.basis_y,
            active_rays.direction, normals, d_refr, t_perp, t_para
        )
        active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        active_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)
        active_rays.Ex = E_x_refr
        active_rays.Ey = E_y_refr
        active_rays.basis_x = b_x_refr
        active_rays.basis_y = b_y_refr

        # ==========================================
        # BOUNCES 1 to 3
        # ==========================================
        for p in range(1, 4):
            hit_mask, t = particle.intersect(active_rays)
            safe_t = dr.select(hit_mask, t, Float(0.0))

            active_rays.opt_path_length += safe_t * ior_real_dr

            out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=ior_inv_dr)
            d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = out

            surface_o = active_rays.origin + active_rays.direction * safe_t

            # Branch A: Refract OUT
            E_x_exit, E_y_exit, b_x_exit, b_y_exit = apply_basis_rotation(
                active_rays.Ex, active_rays.Ey, active_rays.basis_x, active_rays.basis_y,
                active_rays.direction, normals, d_refr, t_perp, t_para
            )

            p_exit_rays = copy.copy(active_rays)
            p_exit_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
            p_exit_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)

            valid_transmission = hit_mask & ~is_tir
            p_exit_rays.Ex = dr.select(valid_transmission, E_x_exit, Complex2f(0.0, 0.0))
            p_exit_rays.Ey = dr.select(valid_transmission, E_y_exit, Complex2f(0.0, 0.0))
            p_exit_rays.basis_x = b_x_exit
            p_exit_rays.basis_y = b_y_exit

            far_field_l = active_rays.opt_path_length - dr.dot(p_exit_rays.origin, p_exit_rays.direction)
            p_exit_rays.opt_path_length = far_field_l

            theta_out_p = dr.acos(dr.clip(p_exit_rays.direction.z, Float(-1.0), Float(1.0)))
            thetas_to_eval.append(theta_out_p)
            exiting_wavefronts.append((p_exit_rays, patches, p, theta_out_p))

            # Branch B: Reflect IN
            E_x_int, E_y_int, b_x_int, b_y_int = apply_basis_rotation(
                active_rays.Ex, active_rays.Ey, active_rays.basis_x, active_rays.basis_y,
                active_rays.direction, normals, d_refl, r_perp, r_para
            )

            active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
            active_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
            active_rays.Ex = E_x_int
            active_rays.Ey = E_y_int
            active_rays.basis_x = b_x_int
            active_rays.basis_y = b_y_int

        final_wavefronts = []
        for rays, patch, p, theta_out in exiting_wavefronts:
            # Read from the logical grid, NOT the rotated world origins
            u_0 = dr.gather(Float, logical_x, patch.v0)
            v_0 = dr.gather(Float, logical_y, patch.v0)

            t_0 = dr.gather(Float, theta_out, patch.v0)
            t_1 = dr.gather(Float, theta_out, patch.v1)
            t_2 = dr.gather(Float, theta_out, patch.v2)

            dt_du = t_1 - t_0
            dt_dv = t_2 - t_0

            derivative_sign = (dt_du * u_0) + (dt_dv * v_0)

            caustic_addition = dr.select(derivative_sign > 0.0, 1, 0)
            base_focal = max(0, p - 1)
            rays.focal_lines_crossed = type(rays.focal_lines_crossed)(base_focal + caustic_addition)

            final_wavefronts.append((rays, patch))

        return final_wavefronts

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, radius_um: float, mu: np.ndarray) -> np.ndarray:
        radius_mm = radius_um / 1000.0
        grid_width_mm = radius_mm * 2.2
        patch_area = (grid_width_mm / self.grid_res) ** 2
        step_size = grid_width_mm / (self.grid_res - 1) if self.grid_res > 1 else 0.0

        ior_real_dr = dr.opaque(Float, float(m.real))
        ior_inv_dr = dr.opaque(Float, 1.0 / float(m.real))

        if self.particle_shape == "sphere":
            particle = SphericalParticle(radius_mm=radius_mm)
        elif self.particle_shape == "oblate":
            # HIJACKED: Now actually uses the correct math class
            particle = OblateSpheroidParticle(radius_mm=radius_mm)
        elif self.particle_shape == "hexagonal":
            particle = HexagonalCrystalParticle(radius_mm=radius_mm, height_mm=radius_mm * 2.0)
        else:
            raise NotImplementedError(f"Particle shape {self.particle_shape} not built yet.")

        # Setup collector...
        internal_bins = 8192
        theta_internal = np.linspace(0.0, np.pi, internal_bins)
        collector = CollectionSphere(mu_bins=np.cos(theta_internal), wavelength_nm=wavelength_nm,
                                     num_phi_bins=self.num_phi_bins)

        # Pre-roll the synchronized randomness for this wavelength
        batch_params = []
        for b in range(self.num_batches):
            ox = (np.random.rand() - 0.5) * step_size
            oy = (np.random.rand() - 0.5) * step_size
            rot = np.random.rand() * np.pi * 2.0
            batch_params.append((ox, oy, rot))

        print(
            f"      -> Running {self.num_batches} batches of {self.grid_res}x{self.grid_res} rays (Pass 1: X-Pol)...",
            flush=True)
        for ox, oy, rot in batch_params:
            rays, patches, log_x, log_y = GridEmitter.emit(self.grid_res, grid_width_mm, ox, oy, rot, pol='X')
            final_wavefronts = self._trace_batch(rays, patches, log_x, log_y, particle, ior_real_dr, ior_inv_dr)
            for batch_rays, batch_patch in final_wavefronts:
                collector.accumulate(batch_rays, batch_patch, patch_area)
            dr.eval(collector._bins_ex_real, collector._bins_ey_real, collector._bins_ez_real)

        intensity_x = collector.finalize()

        print(
            f"      -> Running {self.num_batches} batches of {self.grid_res}x{self.grid_res} rays (Pass 2: Y-Pol)...",
            flush=True)
        for ox, oy, rot in batch_params:
            rays, patches, log_x, log_y = GridEmitter.emit(self.grid_res, grid_width_mm, ox, oy, rot, pol='Y')
            final_wavefronts = self._trace_batch(rays, patches, log_x, log_y, particle, ior_real_dr, ior_inv_dr)
            for batch_rays, batch_patch in final_wavefronts:
                collector.accumulate(batch_rays, batch_patch, patch_area)
            dr.eval(collector._bins_ex_real, collector._bins_ey_real, collector._bins_ez_real)

        intensity_y = collector.finalize()

        total_raw_intensity = (intensity_x + intensity_y) / 2.0
        theta_requested = np.arccos(np.clip(mu, -1.0, 1.0))

        if self.num_phi_bins == 1:
            total_intensity = apply_diffraction_smoothing(
                theta_rad=theta_internal,
                intensity=total_raw_intensity,
                radius_mm=radius_mm,
            )
            return np.interp(theta_requested, theta_internal, total_intensity).astype(np.float32)
        else:
            # 2D Anisotropic interpolation
            result = np.zeros((self.num_phi_bins, len(mu)), dtype=np.float32)
            for p in range(self.num_phi_bins):
                result[p, :] = np.interp(theta_requested, theta_internal, total_raw_intensity[p, :])
            return result
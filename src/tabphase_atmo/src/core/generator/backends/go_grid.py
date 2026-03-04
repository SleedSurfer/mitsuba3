import numpy as np
import drjit as dr
from drjit.auto import Float, Bool, Complex2f,Array3f

from .base import ScatteringBackend
from .geometric.particles.sphere import SphericalParticle
from .geometric.ray_emitter import GridEmitter
from .geometric.collector import CollectionSphere
from .geometric.postprocess import apply_diffraction_smoothing, compute_fraunhofer_diffraction
import copy


class DrJitRaytracerBackend(ScatteringBackend):
    name: str = "drjit_raytracer"

    def __init__(self, grid_res: int = 512, particle_shape: str = "sphere"):
        #self._last_wavefronts = None  # DEBUG ONLY
        self.grid_res = grid_res
        self.particle_shape = particle_shape

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, radius_um: float, mu: np.ndarray) -> np.ndarray:
        radius_mm = radius_um / 1000.0

        if self.particle_shape == "sphere":
            particle = SphericalParticle(radius_mm=radius_mm)
        else:
            raise NotImplementedError(f"Particle shape {self.particle_shape} not built yet.")

        grid_width_mm = radius_mm * 2.2
        active_rays, patches = GridEmitter.emit(self.grid_res, grid_width_mm)
        active_rays.opt_path_length = dr.zeros(Float, dr.shape(active_rays.origin)[1])

        patch_area = (grid_width_mm / self.grid_res) ** 2

        x_init = active_rays.origin.x
        y_init = active_rays.origin.y

        phi = dr.atan2(y_init, x_init) #polar angle extraction
        b_impact = x_init ** 2 + y_init ** 2
        dr.enable_grad(b_impact)

        r_new = dr.sqrt(dr.maximum(b_impact, Float(0.0)))
        active_rays.origin = Array3f(
            r_new * dr.cos(phi),
            r_new * dr.sin(phi),
            active_rays.origin.z
        )

        exiting_wavefronts = []
        thetas_to_eval = []

        # BOUNCE 0
        hit_mask, t = particle.intersect(active_rays)
        safe_t = dr.select(hit_mask, t, Float(0.0))

        active_rays.Ex *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
        active_rays.Ey *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
        active_rays.opt_path_length += safe_t * 1.0

        out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=m.real)
        d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals = out

        surface_o = active_rays.origin + active_rays.direction * safe_t

        # Branch A: Exit (p=0)
        p0_rays = copy.copy(active_rays)
        p0_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        p0_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
        p0_rays.Ex = active_rays.Ex * r_perp
        p0_rays.Ey = active_rays.Ey * r_para

        far_field_l = active_rays.opt_path_length - dr.dot(p0_rays.origin, p0_rays.direction)
        p0_rays.opt_path_length = far_field_l

        theta_out_0 = dr.acos(dr.clip(p0_rays.direction.z, Float(-1.0), Float(1.0)))
        thetas_to_eval.append(theta_out_0)
        exiting_wavefronts.append((p0_rays, patches, 0, theta_out_0))

        # Branch B: Refract IN
        active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        active_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)
        active_rays.Ex *= t_perp
        active_rays.Ey *= t_para

        # BOUNCES 1 to 3
        for p in range(1, 4):
            hit_mask, t = particle.intersect(active_rays)
            safe_t = dr.select(hit_mask, t, Float(0.0))
            active_rays.opt_path_length += safe_t * float(m.real)

            out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=1.0 / m.real)
            d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals = out

            surface_o = active_rays.origin + active_rays.direction * safe_t

            # Branch A: Refract OUT
            p_exit_rays = copy.copy(active_rays)
            p_exit_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
            p_exit_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)
            p_exit_rays.Ex = active_rays.Ex * t_perp
            p_exit_rays.Ey = active_rays.Ey * t_para

            far_field_l = active_rays.opt_path_length - dr.dot(p_exit_rays.origin, p_exit_rays.direction)
            p_exit_rays.opt_path_length = far_field_l

            theta_out_p = dr.acos(dr.clip(p_exit_rays.direction.z, Float(-1.0), Float(1.0)))
            thetas_to_eval.append(theta_out_p)
            exiting_wavefronts.append((p_exit_rays, patches, p, theta_out_p))

            # Branch B: Reflect IN
            active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
            active_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
            active_rays.Ex *= r_perp
            active_rays.Ey *= r_para

        # gradient evaluation block
        dr.set_grad(b_impact, 1.0)
        dr.forward_to(*thetas_to_eval,flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)

        # Apply Sadeghi's analytical logic
        final_wavefronts = []
        for rays, patch, p, theta_out in exiting_wavefronts:
            dtheta_db = dr.grad(theta_out)
            base_focal = max(0, p - 1)

            # Local caustic folding (Sadeghi's derivative check)
            caustic_addition = dr.select(dtheta_db > 0.0, 1, 0)

            rays.focal_lines_crossed = type(rays.focal_lines_crossed)(base_focal + caustic_addition)
            final_wavefronts.append((rays, patch, f"p={p}"))

        # Collectior Phase
        internal_bins = 8192

        theta_internal = np.linspace(0.0, np.pi, internal_bins)
        mu_internal = np.cos(theta_internal)
        collector = CollectionSphere(mu_bins=mu_internal, wavelength_nm=wavelength_nm, num_phi_bins=360)

        for rays, patch, name in final_wavefronts:
            collector.accumulate(rays, patch, patch_area)

        # (Computes |E_p0 + E_p1 + E_p2 + E_p3|^2) for coherent wavefronts
        total_raw_intensity = collector.finalize()
        total_intensity_internal = apply_diffraction_smoothing(
            theta_rad=theta_internal,
            intensity=total_raw_intensity,
            radius_mm=radius_mm,
        )
        total_intensity = total_intensity_internal

        integral = float(np.trapezoid(total_intensity * np.sin(theta_internal), theta_internal) * 2.0 * np.pi)
        normalized_total = (total_intensity / (integral + 1e-12)).astype(np.float32)

        theta_requested = np.arccos(np.clip(mu, -1.0, 1.0))
        return np.interp(theta_requested, theta_internal, normalized_total)
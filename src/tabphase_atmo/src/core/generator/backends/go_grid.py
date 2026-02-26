import numpy as np
import drjit as dr
from drjit.auto import Float, Bool, Complex2f

from .base import ScatteringBackend
from .geometric.particles.sphere import SphericalParticle
from .geometric.ray_emitter import GridEmitter
from .geometric.collector import CollectionSphere
import copy


class DrJitRaytracerBackend(ScatteringBackend):
    name: str = "drjit_raytracer"

    def __init__(self, grid_res: int = 1000, particle_shape: str = "sphere"):
        self._last_wavefronts = None  # DEBUG ONLY
        self.grid_res = grid_res
        self.particle_shape = particle_shape

    def intensity_unpolarized(self, m: complex, x: float, mu: np.ndarray) -> np.ndarray:
        # Derive physical radius from size parameter
        # x = 2πr/λ, using dummy λ = 0.5μm
        dummy_lambda_um = 0.5
        radius_um = (x * dummy_lambda_um) / (2.0 * np.pi)
        radius_mm = radius_um / 1000.0
        wavelength_nm = dummy_lambda_um * 1000.0  # 500nm

        if self.particle_shape == "sphere":
            particle = SphericalParticle(radius_mm=radius_mm)
        else:
            raise NotImplementedError(f"Particle shape {self.particle_shape} not built yet.")

        grid_width_mm = radius_mm * 2.2
        active_rays, patches = GridEmitter.emit(self.grid_res, grid_width_mm)
        active_rays.l = dr.zeros(Float, dr.shape(active_rays.o)[1])

        patch_area = (grid_width_mm / self.grid_res) ** 2

        exiting_wavefronts = []

        # --- BOUNCE 0: Outside hitting the front of the drop ---
        hit_mask, t = particle.intersect(active_rays)
        safe_t = dr.select(hit_mask, t, Float(0.0))

        active_rays.Ex *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
        active_rays.Ey *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
        active_rays.l += safe_t * 1.0

        out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=m.real)
        d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals = out

        surface_o = active_rays.o + active_rays.d * safe_t

        # Branch A: Exit (p=0)
        p0_rays = copy.copy(active_rays)
        p0_rays.o = dr.select(hit_mask, surface_o, active_rays.o)
        p0_rays.d = dr.select(hit_mask, d_refl, active_rays.d)
        p0_rays.Ex = active_rays.Ex * r_perp
        p0_rays.Ey = active_rays.Ey * r_para
        p0_rays.f = active_rays.f + particle.get_focal_crossings(0)
        p0_rays.l = active_rays.l + 0.0
        exiting_wavefronts.append((p0_rays, patches, "p=0"))

        # Branch B: Refract IN
        active_rays.o = dr.select(hit_mask, surface_o, active_rays.o)
        active_rays.d = dr.select(hit_mask, d_refr, active_rays.d)
        active_rays.Ex *= t_perp
        active_rays.Ey *= t_para

        # --- BOUNCES 1 to 3: Inside the Drop ---
        for p in range(1, 4):
            hit_mask, t = particle.intersect(active_rays)
            safe_t = dr.select(hit_mask, t, Float(0.0))
            active_rays.l += safe_t * float(m.real)

            out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=1.0 / m.real)
            d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals = out

            surface_o = active_rays.o + active_rays.d * safe_t

            # Branch A: Refract OUT (Exit)
            p_exit_rays = copy.copy(active_rays)
            p_exit_rays.o = dr.select(hit_mask, surface_o, active_rays.o)
            p_exit_rays.d = dr.select(hit_mask, d_refr, active_rays.d)
            p_exit_rays.Ex = active_rays.Ex * t_perp
            p_exit_rays.Ey = active_rays.Ey * t_para
            p_exit_rays.f = active_rays.f + particle.get_focal_crossings(p)
            p_exit_rays.l = active_rays.l + 0.0
            exiting_wavefronts.append((p_exit_rays, patches, f"p={p}"))

            # Branch B: Reflect IN (Stay inside)
            active_rays.o = dr.select(hit_mask, surface_o, active_rays.o)
            active_rays.d = dr.select(hit_mask, d_refl, active_rays.d)
            active_rays.Ex *= r_perp
            active_rays.Ey *= r_para

        self._last_wavefronts = exiting_wavefronts

        # --- COLLECTION ---
        # Use a UNIFORM internal grid for collection (linear in μ from 1 to -1)
        internal_bins = max(1024, len(mu) * 2)
        mu_internal = np.linspace(1.0, -1.0, internal_bins)

        collector = CollectionSphere(mu_bins=mu_internal, wavelength_nm=wavelength_nm)

        for rays, patch, name in exiting_wavefronts:
            collector.accumulate(rays, patch, patch_area)

        intensity_internal = collector.finalize()

        # Interpolate to the requested mu grid
        # mu_internal goes from 1 to -1, mu (requested) may be anything
        theta_internal = np.arccos(np.clip(mu_internal, -1.0, 1.0))
        theta_requested = np.arccos(np.clip(mu, -1.0, 1.0))

        return np.interp(theta_requested, theta_internal, intensity_internal)
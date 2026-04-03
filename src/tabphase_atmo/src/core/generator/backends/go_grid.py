import numpy as np
import drjit as dr
from drjit.auto import Float, Bool, Complex2f, Array3f, PCG32, UInt32
import copy

# Make the cache spam fuck off
dr.set_log_level(dr.LogLevel.Error)

from .base import ScatteringBackend
from .geometric.particles.sphere import SphericalParticle
from .geometric.particles.oblate_sphere import OblateSpheroidParticle
from .geometric.particles.hexagonal_ice import HexagonalCrystalParticle
from .geometric.optics import apply_basis_rotation

from .geometric.emitters.grid_emitter import GridEmitter
from .geometric.emitters.photon_emitter import PhotonEmitter  # <-- NEW EMITTER

from Collectors.grid_collector import CollectionSphere
from Collectors.photon_collector import PhotonCollectionSphere  # <-- NEW COLLECTOR

from .geometric.postprocess import apply_diffraction_smoothing, circularize_anisotropic_data


class DrJitRaytracerBackend(ScatteringBackend):
    name: str = "drjit_raytracer"

    def __init__(self, grid_res: int = 950, num_batches: int = 10, num_phi_bins: int = 1,
                 particle_shape: str = "sphere"):
        self.grid_res = grid_res
        self.num_batches = num_batches
        self.num_phi_bins = num_phi_bins
        self.particle_shape = particle_shape

    def _trace_photons(self, active_rays, particle, ior_real_dr, ior_inv_dr, collector):
        # ==========================================
        # BOUNCE 0
        # ==========================================
        hit_mask, t = particle.intersect(active_rays)
        safe_t = dr.select(hit_mask, t, Float(0.0))

        out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=ior_real_dr)
        d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = out

        surface_o = active_rays.origin + active_rays.direction * safe_t

        R_eff = dr.sqrt((dr.squared_norm(r_perp) + dr.squared_norm(r_para)) / 2.0)
        T_eff = dr.sqrt((dr.squared_norm(t_perp) + dr.squared_norm(t_para)) / 2.0)

        # Branch A: Exit (p=0)
        p0_rays = copy.copy(active_rays)
        p0_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
        p0_rays.Ex = active_rays.Ex * R_eff
        p0_rays.Ey = active_rays.Ey * R_eff
        collector.accumulate(p0_rays, hit_mask)

        # Branch B: Refract IN
        active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        active_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)
        active_rays.Ex *= T_eff
        active_rays.Ey *= T_eff

        # ==========================================
        # BOUNCES 1 to 3
        # ==========================================
        for p in range(1, 4):
            hit_mask, t = particle.intersect(active_rays)
            safe_t = dr.select(hit_mask, t, Float(0.0))

            out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=ior_inv_dr)
            d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = out

            surface_o = active_rays.origin + active_rays.direction * safe_t

            R_eff = dr.sqrt((dr.squared_norm(r_perp) + dr.squared_norm(r_para)) / 2.0)
            T_eff = dr.sqrt((dr.squared_norm(t_perp) + dr.squared_norm(t_para)) / 2.0)

            # Branch A: Refract OUT
            p_exit_rays = copy.copy(active_rays)
            p_exit_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)

            valid_transmission = hit_mask & ~is_tir
            p_exit_rays.Ex = dr.select(valid_transmission, active_rays.Ex * T_eff, Complex2f(0.0, 0.0))
            p_exit_rays.Ey = dr.select(valid_transmission, active_rays.Ey * T_eff, Complex2f(0.0, 0.0))
            collector.accumulate(p_exit_rays, valid_transmission)

            # Branch B: Reflect IN
            active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
            active_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
            active_rays.Ex *= R_eff
            active_rays.Ey *= R_eff

    def _trace_batch(self, active_rays, patches, logical_x, logical_y, particle, ior_real_dr, ior_inv_dr):
        x_init = active_rays.origin.x
        y_init = active_rays.origin.y

        exiting_wavefronts = []
        thetas_to_eval = []

        hit_mask, t = particle.intersect(active_rays)
        safe_t = dr.select(hit_mask, t, Float(0.0))

        active_rays.Ex *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
        active_rays.Ey *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
        active_rays.opt_path_length += safe_t * 1.0

        out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=ior_real_dr)
        d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = out

        surface_o = active_rays.origin + active_rays.direction * safe_t

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

        for p in range(1, 4):
            hit_mask, t = particle.intersect(active_rays)
            safe_t = dr.select(hit_mask, t, Float(0.0))

            active_rays.opt_path_length += safe_t * ior_real_dr

            out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=ior_inv_dr)
            d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = out

            surface_o = active_rays.origin + active_rays.direction * safe_t

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
        ior_real_dr = dr.opaque(Float, float(m.real))
        ior_inv_dr = dr.opaque(Float, 1.0 / float(m.real))

        internal_bins = 8192
        theta_internal = np.linspace(0.0, np.pi, internal_bins)
        theta_requested = np.arccos(np.clip(mu, -1.0, 1.0))

        # ====================================================================
        # PATH B: THE MONTE CARLO PIPELINE (ICE CRYSTALS)
        # ====================================================================
        if self.particle_shape == "hexagonal":
            chunk_size = 6_000_000
            num_chunks = 20

            particle = HexagonalCrystalParticle(radius_mm=radius_mm, height_mm=radius_mm * 2.0)
            collector = PhotonCollectionSphere(mu_bins=np.cos(theta_internal), num_phi_bins=self.num_phi_bins)

            print(f"      -> Bypassing Wavefronts. Monte Carlo Brute Force Engaged.", flush=True)
            print(f"      -> Blasting {chunk_size * num_chunks:,} Photons (Pass 1: X-Pol)...", flush=True)

            # X-POL CHUNKING
            for c in range(num_chunks):
                idx = dr.arange(UInt32, chunk_size) + (c * chunk_size)
                # Knuth's multiplicative hash completely scrambles the initial state
                hashed_seed = idx * 2654435769

                rng_x = PCG32(size=chunk_size, initstate=hashed_seed, initseq=1)

                rays_x = PhotonEmitter.emit(chunk_size, target_radius_mm=radius_mm * 2.2, rng=rng_x, pol='X')
                particle.randomize_orientations(chunk_size, rng=rng_x)

                self._trace_photons(rays_x, particle, ior_real_dr, ior_inv_dr, collector)
                dr.eval(collector.bins_intensity)

            intensity_x = collector.finalize()

            print(f"      -> Blasting {chunk_size * num_chunks:,} Photons (Pass 2: Y-Pol)...", flush=True)

            # Y-POL CHUNKING
            for c in range(num_chunks):
                idx = dr.arange(UInt32, chunk_size) + (c * chunk_size)

                hashed_seed = idx * 3266489917

                rng_y = PCG32(size=chunk_size, initstate=hashed_seed, initseq=2)

                rays_y = PhotonEmitter.emit(chunk_size, target_radius_mm=radius_mm * 2.2, rng=rng_y, pol='Y')
                particle.randomize_orientations(chunk_size, rng=rng_y)

                self._trace_photons(rays_y, particle, ior_real_dr, ior_inv_dr, collector)
                dr.eval(collector.bins_intensity)

            intensity_y = collector.finalize()

            total_raw_intensity = (intensity_x + intensity_y) / 2.0

            if self.num_phi_bins == 1:
                return np.interp(theta_requested, theta_internal, total_raw_intensity).astype(np.float32)
            else:
                result = np.zeros((self.num_phi_bins, len(mu)), dtype=np.float32)
                for p in range(self.num_phi_bins):
                    result[p, :] = np.interp(theta_requested, theta_internal, total_raw_intensity[p, :])
                return result


        # ====================================================================
        # PATH A: THE WAVEFRONT PIPELINE (SPHERES & OBLATES)
        # ====================================================================
        else:
            grid_width_mm = radius_mm * 2.2
            patch_area = (grid_width_mm / self.grid_res) ** 2
            step_size = grid_width_mm / (self.grid_res - 1) if self.grid_res > 1 else 0.0

            if self.particle_shape == "sphere":
                print(f"[DEBUG] Branch: Sphere (Path A)")
                print(f"[DEBUG] Particle: SphericalParticle")
                particle = SphericalParticle(radius_mm=radius_mm)
            elif self.particle_shape == "oblate":
                print(f"[DEBUG] Branch: Oblate (Path A)")
                print(f"[DEBUG] Particle: OblateSpheroidParticle")
                particle = OblateSpheroidParticle(radius_mm=radius_mm)
            else:
                raise NotImplementedError(f"Particle shape {self.particle_shape} not built yet.")

            print(f"[DEBUG] Collector: CollectionSphere")
            print(f"[DEBUG] Emitter: GridEmitter")

            collector = CollectionSphere(mu_bins=np.cos(theta_internal), wavelength_nm=wavelength_nm,
                                         num_phi_bins=self.num_phi_bins)

            batch_params = []
            for b in range(self.num_batches):
                ox = (np.random.rand() - 0.5) * step_size
                oy = (np.random.rand() - 0.5) * step_size
                rot = np.random.rand() * np.pi * 2.0
                batch_params.append((ox, oy, rot))

            print(f"      -> Wavefront Tracer Engaged.", flush=True)
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

            if self.num_phi_bins == 1:
                total_intensity = apply_diffraction_smoothing(
                    theta_rad=theta_internal,
                    intensity=total_raw_intensity,
                    radius_mm=radius_mm,
                )
                return np.interp(theta_requested, theta_internal, total_intensity).astype(np.float32)
            else:
                result = np.zeros((self.num_phi_bins, len(mu)), dtype=np.float32)
                for p in range(self.num_phi_bins):
                    result[p, :] = np.interp(theta_requested, theta_internal, total_raw_intensity[p, :])
                return result


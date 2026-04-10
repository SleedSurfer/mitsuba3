import numpy as np
import drjit as dr
from drjit.auto import Float, Complex2f
import copy
from ..tracer_core import trace_bounce
from ..optics import apply_basis_rotation
from ..emitters.grid_emitter import GridEmitter
from ..postprocess import apply_diffraction_smoothing


def calculate_focal_crossings(exiting_wavefronts, logical_x, logical_y):
    """
    The Caustic Engine. Takes the raw exiting wavefronts and calculates the
    Jacobian determinants across the ray quads to detect focal line crossings
    (Gouy phase shifts).
    """
    final_wavefronts = []

    for rays, patch, bounce_num, theta_out in exiting_wavefronts:
        # Fetch logical starting coordinates for the patch triangle (v0, v1, v2)
        u_0 = dr.gather(Float, logical_x, patch.v0)
        v_0 = dr.gather(Float, logical_y, patch.v0)

        # Fetch exiting angles for the patch
        t_0 = dr.gather(Float, theta_out, patch.v0)
        t_1 = dr.gather(Float, theta_out, patch.v1)
        t_2 = dr.gather(Float, theta_out, patch.v2)

        # Numerical derivatives across the wavefront quad
        dt_du = t_1 - t_0
        dt_dv = t_2 - t_0

        # The sign of the derivative tells us if the wavefront has folded in on itself
        derivative_sign = (dt_du * u_0) + (dt_dv * v_0)
        caustic_addition = dr.select(derivative_sign > 0.0, 1, 0)

        # Base focal crossings depend on the internal bounce number
        base_focal = max(0, bounce_num - 1)
        rays.focal_lines_crossed = type(rays.focal_lines_crossed)(base_focal + caustic_addition)

        final_wavefronts.append((rays, patch))

    return final_wavefronts


def _trace_phasor_batch(active_rays, patches, logical_x, logical_y, particle, ior_real_dr, ior_inv_dr):
    """
    Executes a single batch of phasor tracking. Clean, modular, no boilerplate.
    """
    exiting_wavefronts = []

    # ==========================================
    # BOUNCE 0 (Entering the Particle)
    # ==========================================
    hit_mask, safe_t, surface_o, d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = trace_bounce(
        active_rays, particle, ior_real_dr
    )

    # Initialize path length and zero out missed rays
    active_rays.Ex *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
    active_rays.Ey *= dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
    active_rays.opt_path_length += safe_t * 1.0

    # 1. Handle external reflection (Branch A)
    E_x_refl, E_y_refl, b_x_refl, b_y_refl = apply_basis_rotation(
        active_rays.Ex, active_rays.Ey, active_rays.basis_x, active_rays.basis_y,
        active_rays.direction, normals, d_refl, r_perp, r_para
    )

    p0_rays = copy.copy(active_rays)
    p0_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
    p0_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
    p0_rays.Ex, p0_rays.Ey = E_x_refl, E_y_refl
    p0_rays.basis_x, p0_rays.basis_y = b_x_refl, b_y_refl

    p0_rays.opt_path_length = active_rays.opt_path_length - dr.dot(p0_rays.origin, p0_rays.direction)
    theta_out_0 = dr.acos(dr.clip(p0_rays.direction.z, Float(-1.0), Float(1.0)))
    exiting_wavefronts.append((p0_rays, patches, 0, theta_out_0))

    # 2. Handle refraction inward (Branch B - Updates active_rays for next loops)
    E_x_refr, E_y_refr, b_x_refr, b_y_refr = apply_basis_rotation(
        active_rays.Ex, active_rays.Ey, active_rays.basis_x, active_rays.basis_y,
        active_rays.direction, normals, d_refr, t_perp, t_para
    )
    active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
    active_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)
    active_rays.Ex, active_rays.Ey = E_x_refr, E_y_refr
    active_rays.basis_x, active_rays.basis_y = b_x_refr, b_y_refr

    # ==========================================
    # BOUNCES 1 to 3 (Internal Reflections)
    # ==========================================
    for p in range(1, 4):
        hit_mask, safe_t, surface_o, d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = trace_bounce(
            active_rays, particle, ior_inv_dr
        )

        active_rays.opt_path_length += safe_t * ior_real_dr

        # 1. Handle refraction outward (Exit)
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
        p_exit_rays.basis_x, p_exit_rays.basis_y = b_x_exit, b_y_exit

        p_exit_rays.opt_path_length = active_rays.opt_path_length - dr.dot(p_exit_rays.origin, p_exit_rays.direction)
        theta_out_p = dr.acos(dr.clip(p_exit_rays.direction.z, Float(-1.0), Float(1.0)))
        exiting_wavefronts.append((p_exit_rays, patches, p, theta_out_p))

        # 2. Handle internal reflection (Updates active_rays)
        E_x_int, E_y_int, b_x_int, b_y_int = apply_basis_rotation(
            active_rays.Ex, active_rays.Ey, active_rays.basis_x, active_rays.basis_y,
            active_rays.direction, normals, d_refl, r_perp, r_para
        )
        active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        active_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
        active_rays.Ex, active_rays.Ey = E_x_int, E_y_int
        active_rays.basis_x, active_rays.basis_y = b_x_int, b_y_int

    # Process all exiting rays through the caustic engine
    return calculate_focal_crossings(exiting_wavefronts, logical_x, logical_y)


def run_phasor_pipeline(config, particle, collector, theta_internal, theta_requested, ior_real_dr, ior_inv_dr):
    """
    The orchestrator for Path A (Wavefronts).
    """
    radius_mm = particle.radius
    grid_width_mm = radius_mm * 2.2
    patch_area = (grid_width_mm / config.grid_res) ** 2
    step_size = grid_width_mm / (config.grid_res - 1) if config.grid_res > 1 else 0.0

    # Pre-generate jittered batch offsets to avoid aliasing
    batch_params = [
        ((np.random.rand() - 0.5) * step_size, (np.random.rand() - 0.5) * step_size, np.random.rand() * np.pi * 2.0)
        for _ in range(config.num_batches)
    ]

    def _run_pass(pol_type):
        for ox, oy, rot in batch_params:
            rays, patches, log_x, log_y = GridEmitter.emit(config.grid_res, grid_width_mm, ox, oy, rot,
                                                           pol=pol_type)
            final_wavefronts = _trace_phasor_batch(rays, patches, log_x, log_y, particle, ior_real_dr, ior_inv_dr)

            for batch_rays, batch_patch in final_wavefronts:
                collector.accumulate(batch_rays, batch_patch, patch_area)

            dr.eval(collector._bins_ex_real, collector._bins_ey_real, collector._bins_ez_real)
        return collector.finalize()

    # Pass 1: X-Polarized
    intensity_x = _run_pass('X')

    # Pass 2: Y-Polarized
    intensity_y = _run_pass('Y')

    total_raw_intensity = (intensity_x + intensity_y) / 2.0

    # Post-process: Apply diffraction if it's a 1D trace, otherwise interpolate the 2D field
    if collector.num_phi_bins == 1:
        total_intensity = apply_diffraction_smoothing(
            theta_rad=theta_internal,
            intensity=total_raw_intensity,
            radius_mm=float(radius_mm),
        )
        return np.interp(theta_requested, theta_internal, total_intensity).astype(np.float32)
    else:
        result = np.zeros((collector.num_phi_bins, len(theta_requested)), dtype=np.float32)
        for p in range(collector.num_phi_bins):
            result[p, :] = np.interp(theta_requested, theta_internal, total_raw_intensity[p, :])
        return result
import numpy as np
import drjit as dr
from drjit.llvm.ad import Float, Complex2f, Array3f
import copy
from ..tracer_core import trace_bounce
from ..optics import apply_basis_rotation
from ..emitters.grid_emitter import GridEmitter

import numpy as np
import scipy.special as sp


def apply_diffraction_block(theta_internal, total_raw_intensity, radius_mm):
    """
    Dumps the total diffracted energy into a flat block within the first 5 degrees.
    Shape is irrelevant, we just need the exact energy for the HG lobe weight.
    """
    # 5 degrees in radians
    cutoff_rad = np.radians(5.0)

    # Calculate the solid angle of a 5-degree cone: 2 * pi * (1 - cos(theta))
    solid_angle = 2.0 * np.pi * (1.0 - np.cos(cutoff_rad))

    # Total diffracted energy = geometric cross-section (pi * r^2)
    diffracted_energy = np.pi * (radius_mm ** 2)

    # The constant intensity needed to pack all that energy into this specific cone
    intensity_block = diffracted_energy / solid_angle

    # Slap that block of energy ONLY into the first 5 degrees. Badabing badabong.
    diffraction_array = np.where(theta_internal <= cutoff_rad, intensity_block, 0.0)

    return total_raw_intensity + diffraction_array


def _compute_jacobian_determinants(x_init, y_init, thetas_to_eval, phis_to_eval):
    """
    Performs a double forward-mode AD sweep to compute the 2D Jacobian determinant
    for the mapping from entry coordinates (x, y) to exit angles (theta, phi).
    """
    # Sweep 1: Partial derivatives with respect to X
    dr.set_grad(x_init, 1.0)
    dr.set_grad(y_init, 0.0)
    dr.forward_to(*thetas_to_eval, *phis_to_eval, flags=dr.ADFlag.Default)

    dtheta_dx = [dr.grad(t) for t in thetas_to_eval]
    dphi_dx = [dr.grad(p) for p in phis_to_eval]

    # Sweep 2: Partial derivatives with respect to Y
    dr.set_grad(x_init, 0.0)
    dr.set_grad(y_init, 1.0)
    dr.forward_to(*thetas_to_eval, *phis_to_eval, flags=dr.ADFlag.Default)

    dtheta_dy = [dr.grad(t) for t in thetas_to_eval]
    dphi_dy = [dr.grad(p) for p in phis_to_eval]

    # Calculate the determinants
    determinants = []
    for i in range(len(thetas_to_eval)):
        det_J = (dtheta_dx[i] * dphi_dy[i]) - (dtheta_dy[i] * dphi_dx[i])
        determinants.append(det_J)

    return determinants


def _trace_phasor_batch(active_rays, patches, particle, ior_real_dr, ior_inv_dr):
    x_init = active_rays.origin.x
    y_init = active_rays.origin.y

    # Enable gradients for the independent input variables
    dr.enable_grad(x_init, y_init)

    b_impact = x_init ** 2 + y_init ** 2
    r_new = dr.sqrt(dr.maximum(b_impact, Float(0.0)))
    phi = dr.atan2(y_init, x_init)
    active_rays.origin = Array3f(r_new * dr.cos(phi), r_new * dr.sin(phi), active_rays.origin.z)

    exiting_wavefronts = []
    thetas_to_eval = []
    phis_to_eval = []

    # BOUNCE 0
    hit_mask, safe_t, surface_o, d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = trace_bounce(
        active_rays, particle, ior_real_dr
    )

    mask_cplx = dr.select(hit_mask, Complex2f(1.0, 0.0), Complex2f(0.0, 0.0))
    active_rays.Ex_X *= mask_cplx
    active_rays.Ey_X *= mask_cplx
    active_rays.Ex_Y *= mask_cplx
    active_rays.Ey_Y *= mask_cplx
    active_rays.opt_path_length += safe_t * 1.0

    # Branch A: External Reflection
    Ex_refl_X, Ey_refl_X, Ex_refl_Y, Ey_refl_Y, b_x_refl, b_y_refl = apply_basis_rotation(
        active_rays.Ex_X, active_rays.Ey_X, active_rays.Ex_Y, active_rays.Ey_Y,
        active_rays.basis_x, active_rays.basis_y, active_rays.direction, normals, d_refl, r_perp, r_para
    )

    p0_rays = copy.copy(active_rays)
    p0_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
    p0_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
    p0_rays.Ex_X, p0_rays.Ey_X = Ex_refl_X, Ey_refl_X
    p0_rays.Ex_Y, p0_rays.Ey_Y = Ex_refl_Y, Ey_refl_Y
    p0_rays.basis_x, p0_rays.basis_y = b_x_refl, b_y_refl
    p0_rays.opt_path_length = active_rays.opt_path_length - dr.dot(p0_rays.origin, p0_rays.direction)

    theta_out_0 = dr.acos(dr.clip(p0_rays.direction.z, Float(-1.0), Float(1.0)))
    phi_out_0 = dr.atan2(p0_rays.direction.y, p0_rays.direction.x)

    thetas_to_eval.append(theta_out_0)
    phis_to_eval.append(phi_out_0)
    exiting_wavefronts.append((p0_rays, patches, 0))

    # Branch B: Refraction Inward
    Ex_refr_X, Ey_refr_X, Ex_refr_Y, Ey_refr_Y, b_x_refr, b_y_refr = apply_basis_rotation(
        active_rays.Ex_X, active_rays.Ey_X, active_rays.Ex_Y, active_rays.Ey_Y,
        active_rays.basis_x, active_rays.basis_y, active_rays.direction, normals, d_refr, t_perp, t_para
    )
    active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
    active_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)
    active_rays.Ex_X, active_rays.Ey_X = Ex_refr_X, Ey_refr_X
    active_rays.Ex_Y, active_rays.Ey_Y = Ex_refr_Y, Ey_refr_Y
    active_rays.basis_x, active_rays.basis_y = b_x_refr, b_y_refr

    # BOUNCES 1 to 3
    for p in range(1, 4):
        hit_mask, safe_t, surface_o, d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = trace_bounce(
            active_rays, particle, ior_inv_dr
        )
        active_rays.opt_path_length += safe_t * ior_real_dr

        # Branch A: Refract Outward
        Ex_exit_X, Ey_exit_X, Ex_exit_Y, Ey_exit_Y, b_x_exit, b_y_exit = apply_basis_rotation(
            active_rays.Ex_X, active_rays.Ey_X, active_rays.Ex_Y, active_rays.Ey_Y,
            active_rays.basis_x, active_rays.basis_y, active_rays.direction, normals, d_refr, t_perp, t_para
        )

        p_exit_rays = copy.copy(active_rays)
        p_exit_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        p_exit_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)

        valid_trans = hit_mask & ~is_tir
        p_exit_rays.Ex_X = dr.select(valid_trans, Ex_exit_X, Complex2f(0.0))
        p_exit_rays.Ey_X = dr.select(valid_trans, Ey_exit_X, Complex2f(0.0))
        p_exit_rays.Ex_Y = dr.select(valid_trans, Ex_exit_Y, Complex2f(0.0))
        p_exit_rays.Ey_Y = dr.select(valid_trans, Ey_exit_Y, Complex2f(0.0))
        p_exit_rays.basis_x, p_exit_rays.basis_y = b_x_exit, b_y_exit
        p_exit_rays.opt_path_length = active_rays.opt_path_length - dr.dot(p_exit_rays.origin, p_exit_rays.direction)

        theta_out_p = dr.acos(dr.clip(p_exit_rays.direction.z, Float(-1.0), Float(1.0)))
        phi_out_p = dr.atan2(p_exit_rays.direction.y, p_exit_rays.direction.x)

        thetas_to_eval.append(theta_out_p)
        phis_to_eval.append(phi_out_p)
        exiting_wavefronts.append((p_exit_rays, patches, p))

        # Branch B: Reflect Inward
        Ex_int_X, Ey_int_X, Ex_int_Y, Ey_int_Y, b_x_int, b_y_int = apply_basis_rotation(
            active_rays.Ex_X, active_rays.Ey_X, active_rays.Ex_Y, active_rays.Ey_Y,
            active_rays.basis_x, active_rays.basis_y, active_rays.direction, normals, d_refl, r_perp, r_para
        )
        active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        active_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
        active_rays.Ex_X, active_rays.Ey_X = Ex_int_X, Ey_int_X
        active_rays.Ex_Y, active_rays.Ey_Y = Ex_int_Y, Ey_int_Y
        active_rays.basis_x, active_rays.basis_y = b_x_int, b_y_int

    # CAUSTIC ENGINE
    determinants = _compute_jacobian_determinants(x_init, y_init, thetas_to_eval, phis_to_eval)

    final_wavefronts = []
    for i, (rays, patch, p) in enumerate(exiting_wavefronts):
        det_J = determinants[i]
        base_focal = max(0, p - 1)

        if p == 3:
            caustic_addition = dr.select(det_J < 0.0, 1, 0)
        else:
            caustic_addition = dr.select(det_J > 0.0, 1, 0)

        rays.focal_lines_crossed = type(rays.focal_lines_crossed)(base_focal + caustic_addition)
        final_wavefronts.append((rays, patch))

    dr.disable_grad(x_init, y_init)
    return final_wavefronts


def run_phasor_pipeline(config, particle, collector, theta_internal, theta_requested, ior_real_dr, ior_inv_dr):
    radius_mm = particle.radius
    grid_width_mm = radius_mm * 2.2
    patch_area = (grid_width_mm / config.grid_res) ** 2
    step_size = grid_width_mm / (config.grid_res - 1) if config.grid_res > 1 else 0.0
    #config.num_batches = 2
    batch_params = [
        ((np.random.rand() - 0.5) * step_size, (np.random.rand() - 0.5) * step_size, np.random.rand() * np.pi * 2.0)
        for _ in range(config.num_batches)
    ]

    total_intensity = 0
    for ox, oy, rot in batch_params:
        rays, patches, _, _ = GridEmitter.emit(config.grid_res, grid_width_mm, ox, oy, rot)
        final_wavefronts = _trace_phasor_batch(rays, patches, particle, ior_real_dr, ior_inv_dr)

        for batch_rays, batch_patch in final_wavefronts:
            collector.accumulate(batch_rays, batch_patch, patch_area)

        # Eval both sets of bins before finalizing
        dr.eval(
            collector._bins_ex_real_X, collector._bins_ey_real_X, collector._bins_ez_real_X,
            collector._bins_ex_real_Y, collector._bins_ey_real_Y, collector._bins_ez_real_Y
        )

        total_intensity += collector.finalize()

    total_raw_intensity = total_intensity / config.num_batches

    total_raw_intensity = apply_diffraction_block(
        theta_internal=theta_internal,
        total_raw_intensity=total_raw_intensity,
        radius_mm=particle.radius
    )

    if collector.num_phi_bins == 1:
        return np.interp(theta_requested, theta_internal, total_raw_intensity).astype(np.float32)
    else:
        result = np.zeros((collector.num_phi_bins, len(theta_requested)), dtype=np.float32)
        for p in range(collector.num_phi_bins):
            result[p, :] = np.interp(theta_requested, theta_internal, total_raw_intensity[p, :])
        return result
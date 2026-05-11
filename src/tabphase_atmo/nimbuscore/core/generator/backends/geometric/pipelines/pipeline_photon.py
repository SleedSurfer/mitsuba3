import numpy as np
import drjit as dr
from drjit.auto import Float, Complex2f, PCG32, UInt32
import copy

from ..tracer_core import trace_bounce
from ..emitters.photon_emitter import PhotonEmitter


def _trace_photon_chunk(active_rays, particle, ior_real_dr, ior_inv_dr, collector):
    # BOUNCE 0
    hit_mask, safe_t, surface_o, d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = trace_bounce(
        active_rays, particle, ior_real_dr
    )

    # Pure energy weights (No dr.sqrt required)
    R_weight = (dr.squared_norm(r_perp) + dr.squared_norm(r_para)) / Float(2.0)
    T_weight = (dr.squared_norm(t_perp) + dr.squared_norm(t_para)) / Float(2.0)

    # Branch A: Exit
    p0_rays = copy.copy(active_rays)
    p0_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
    p0_rays.weight = active_rays.weight * R_weight
    collector.accumulate(p0_rays, hit_mask)

    # Branch B: Refract IN
    active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
    active_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)
    active_rays.weight *= T_weight

    # BOUNCES 1 to 3
    for p in range(1, 4):
        hit_mask, safe_t, surface_o, d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = trace_bounce(
            active_rays, particle, ior_inv_dr
        )

        R_weight = (dr.squared_norm(r_perp) + dr.squared_norm(r_para)) / Float(2.0)
        T_weight = (dr.squared_norm(t_perp) + dr.squared_norm(t_para)) / Float(2.0)

        # Branch A: Refract OUT
        p_exit_rays = copy.copy(active_rays)
        p_exit_rays.direction = dr.select(hit_mask, d_refr, active_rays.direction)

        valid_trans = hit_mask & ~is_tir
        p_exit_rays.weight = dr.select(valid_trans, active_rays.weight * T_weight, Float(0.0))
        collector.accumulate(p_exit_rays, valid_trans)

        # Branch B: Reflect IN
        active_rays.origin = dr.select(hit_mask, surface_o, active_rays.origin)
        active_rays.direction = dr.select(hit_mask, d_refl, active_rays.direction)
        active_rays.weight *= R_weight

def _run_photon_pass(particle, collector, num_chunks, chunk_size, radius_mm, ior_real_dr, ior_inv_dr, seed_offset, seq_id):
    for c in range(num_chunks):
        idx = dr.arange(UInt32, chunk_size) + (c * chunk_size)
        rng = PCG32(size=chunk_size, initstate=idx * UInt32(seed_offset), initseq=seq_id)

        rays = PhotonEmitter.emit(chunk_size, target_radius_mm=radius_mm * 2.2, rng=rng)
        particle.randomize_orientations(chunk_size, rng=rng)

        _trace_photon_chunk(rays, particle, ior_real_dr, ior_inv_dr, collector)
        dr.eval(collector.bins_intensity)

    return collector.finalize()

def run_photon_pipeline(particle, collector, theta_internal, theta_requested, ior_real_dr, ior_inv_dr, num_phi_bins):
    radius_mm = particle.R
    chunk_size = 3_000_000
    num_chunks = 20

    total_raw_intensity = _run_photon_pass(
        particle, collector, num_chunks, chunk_size, radius_mm,
        ior_real_dr, ior_inv_dr, 2654435769, 1
    )

    if theta_internal is theta_requested:
        if num_phi_bins == 1:
            return total_raw_intensity.flatten().astype(np.float32)
        return total_raw_intensity.astype(np.float32)

    if num_phi_bins == 1:
        return np.interp(theta_requested, theta_internal, total_raw_intensity).astype(np.float32)
    else:
        result = np.zeros((num_phi_bins, len(theta_requested)), dtype=np.float32)
        for p in range(num_phi_bins):
            result[p, :] = np.interp(theta_requested, theta_internal, total_raw_intensity[p, :])
        return result
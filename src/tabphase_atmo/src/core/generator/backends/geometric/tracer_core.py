import drjit as dr
from drjit.auto import Float


def trace_bounce(active_rays, particle, ior_eval: Float):
    """
    The absolute bare-metal physics core. No wavefronts, no photons, just pure geometry and optics.
    Calculates the intersection point and the raw scattering parameters for the current bounce.
    """
    # 1. Find the intersection point
    hit_mask, t = particle.intersect(active_rays)

    # Anti-NaN: Prevent blowing up DrJit's autodiff graph on missed rays
    safe_t = dr.select(hit_mask, t, Float(0.0))

    # 2. Compute exact 3D surface hit point
    surface_o = active_rays.origin + active_rays.direction * safe_t

    # 3. Ask the particle to compute normals and run the Snell/Fresnel optics
    out = particle.scatter(active_rays, hit_mask, safe_t, ior_water=ior_eval)
    d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir = out

    return hit_mask, safe_t, surface_o, d_refl, d_refr, r_perp, r_para, t_perp, t_para, normals, is_tir
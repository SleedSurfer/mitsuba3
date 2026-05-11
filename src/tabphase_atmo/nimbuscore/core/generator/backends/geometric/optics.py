import drjit as dr
from drjit.llvm.ad import Float, Array3f, Complex2f


def apply_basis_rotation(Ex_X, Ey_X, Ex_Y, Ey_Y, basis_x, basis_y, d_in, n, d_out, coef_perp, coef_para):
    """
    Projects the dual Ex/Ey phasors onto the local scattering plane, applies
    Fresnel coefficients, and returns the new phasors alongside the new basis vectors.
    """
    s_raw = dr.cross(d_in, n)
    s_norm = dr.norm(s_raw)

    valid_s = s_norm > 1e-6
    s = dr.select(valid_s, s_raw / dr.maximum(s_norm, Float(1e-8)), basis_x)

    p_in = dr.cross(s, d_in)

    dot_x_s = dr.dot(basis_x, s)
    dot_y_s = dr.dot(basis_y, s)
    dot_x_p = dr.dot(basis_x, p_in)
    dot_y_p = dr.dot(basis_y, p_in)

    # Project X-Source
    E_perp_X = Ex_X * dot_x_s + Ey_X * dot_y_s
    E_para_X = Ex_X * dot_x_p + Ey_X * dot_y_p
    E_perp_out_X = E_perp_X * coef_perp
    E_para_out_X = E_para_X * coef_para

    # Project Y-Source
    E_perp_Y = Ex_Y * dot_x_s + Ey_Y * dot_y_s
    E_para_Y = Ex_Y * dot_x_p + Ey_Y * dot_y_p
    E_perp_out_Y = E_perp_Y * coef_perp
    E_para_out_Y = E_para_Y * coef_para

    p_out = dr.cross(s, d_out)

    return E_perp_out_X, E_para_out_X, E_perp_out_Y, E_para_out_Y, s, p_out


def compute_fresnel_and_scatter(d: Array3f, n: Array3f, ior_in: float, ior_out: float):
    """
    Computes Snell's Law and Complex Fresnel coefficients.
    """
    eta = Float(ior_in / ior_out)

    cos_theta_i_raw = -dr.dot(d, n)
    n_safe = dr.select(cos_theta_i_raw < 0.0, -n, n)
    cos_theta_i = dr.abs(cos_theta_i_raw)
    d_reflected = d + n_safe * (Float(2.0) * cos_theta_i)

    k = Float(1.0) - (eta * eta) * (Float(1.0) - cos_theta_i * cos_theta_i)
    is_tir = k < 0.0

    sqrt_k_real = dr.sqrt(dr.maximum(k, Float(0.0)))
    d_refracted = d * eta + n_safe * (eta * cos_theta_i - sqrt_k_real)

    cos_theta_t = Complex2f(
        dr.select(is_tir, Float(0.0), dr.sqrt(k)),
        dr.select(is_tir, dr.sqrt(-k), Float(0.0))
    )

    c_i = Complex2f(cos_theta_i, Float(0.0))
    n1 = Complex2f(Float(ior_in), Float(0.0))
    n2 = Complex2f(Float(ior_out), Float(0.0))

    r_perp = (n1 * c_i - n2 * cos_theta_t) / (n1 * c_i + n2 * cos_theta_t)
    r_para = (n2 * c_i - n1 * cos_theta_t) / (n2 * c_i + n1 * cos_theta_t)

    two_n1_ci = Complex2f(Float(2.0), Float(0.0)) * n1 * c_i
    t_perp = two_n1_ci / (n1 * c_i + n2 * cos_theta_t)
    t_para = two_n1_ci / (n2 * c_i + n1 * cos_theta_t)

    return d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, is_tir
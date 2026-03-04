import drjit as dr
from drjit.auto import Float, Array3f, Complex2f

def compute_fresnel_and_scatter(d: Array3f, n: Array3f, ior_in: float, ior_out: float):
    """
    Computes Snell's Law and Complex Fresnel coefficients.
    FIXED: Normal flipping for inside-out boundaries.
    """
    eta = Float(ior_in / ior_out)

    # ENFORCE NORMAL FLIP
    cos_theta_i_raw = -dr.dot(d, n)
    n_safe = dr.select(cos_theta_i_raw < 0.0, -n, n)
    cos_theta_i = dr.abs(cos_theta_i_raw) # Now strictly positive

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
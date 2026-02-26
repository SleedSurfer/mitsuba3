import drjit as dr
from drjit.auto import Float, Bool, Array3f, Complex2f


def compute_fresnel_and_scatter(d: Array3f, n: Array3f, ior_in: float, ior_out: float):
    """
    Computes Snell's Law and Complex Fresnel coefficients for amplitude splitting.
    """
    eta_val = ior_in / ior_out
    eta = Float(eta_val)

    # Cosine of the incident angle
    cos_theta_i = -dr.dot(d, n)

    # 1. Geometric Optics: Trajectories
    d_reflected = d + n * (Float(2.0) * cos_theta_i)

    # Snell's Law discriminant (k)
    k = Float(1.0) - (eta * eta) * (Float(1.0) - cos_theta_i * cos_theta_i)
    is_tir = k < 0.0

    # Refracted direction (dr.maximum prevents nan in TIR cases)
    sqrt_k_real = dr.sqrt(dr.maximum(k, Float(0.0)))
    d_refracted = d * eta + n * (eta * cos_theta_i - sqrt_k_real)

    # 2. Wave Optics: Complex Fresnel Coefficients
    # If TIR occurs, cos_theta_t becomes purely imaginary
    cos_theta_t = Complex2f(
        dr.select(is_tir, Float(0.0), dr.sqrt(k)),  # Real part
        dr.select(is_tir, dr.sqrt(-k), Float(0.0))  # Imaginary part
    )

    c_i = Complex2f(cos_theta_i, Float(0.0))
    n1 = Complex2f(Float(ior_in), Float(0.0))
    n2 = Complex2f(Float(ior_out), Float(0.0))

    # Reflection amplitudes: Perpendicular (s-pol) and Parallel (p-pol)
    r_perp = (n1 * c_i - n2 * cos_theta_t) / (n1 * c_i + n2 * cos_theta_t)
    r_para = (n2 * c_i - n1 * cos_theta_t) / (n2 * c_i + n1 * cos_theta_t)

    # Transmission amplitudes: Perpendicular and Parallel
    two_n1_ci = Complex2f(Float(2.0), Float(0.0)) * n1 * c_i
    t_perp = two_n1_ci / (n1 * c_i + n2 * cos_theta_t)
    t_para = two_n1_ci / (n2 * c_i + n1 * cos_theta_t)

    return d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, is_tir
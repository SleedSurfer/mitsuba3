import drjit as dr
from drjit.llvm import Float, UInt32, PCG32
import numpy as np
from scipy.special import j1
from scipy.special import airy
from .base import ScatteringBackend


class DrJitRaytracerBackend(ScatteringBackend):
    name = "drjit_raytracer_dep"

    def __init__(self, num_rays=10_000_000, max_bounces=6, suppress_p2=False):
        self.num_rays = num_rays
        self.max_bounces = max_bounces
        self.suppress_p2 = suppress_p2

    def apply_airy_lobe(self, theta_grid, energy, m, x, p):
        n = m.real
        cos_alpha_i = np.sqrt((n ** 2 - 1.0) / (p ** 2 - 1.0))
        alpha_i = np.arccos(cos_alpha_i)
        sin_alpha_i = np.sin(alpha_i)

        theta_d = 2.0 * alpha_i - 2.0 * p * np.arcsin(sin_alpha_i / n) + (p - 1.0) * np.pi
        theta_d = np.abs(theta_d % (2 * np.pi))
        if theta_d > np.pi: theta_d = 2 * np.pi - theta_d

        h = ((12.0 * sin_alpha_i) / (x ** 2 * cos_alpha_i ** 3)) ** (1 / 3.0)

        sign = -1.0 if p == 2 else 1.0
        z = sign * (theta_grid - theta_d) / h

        Ai, _, _, _ = airy(z)
        intensity_airy = (Ai ** 2)

        window = np.exp(-0.5 * ((theta_grid - theta_d) / 0.35) ** 2)
        intensity_airy *= window

        solid_angle_weight = 2.0 * np.pi * np.sin(theta_grid)
        integral = np.trapezoid(intensity_airy * solid_angle_weight, theta_grid)

        return (intensity_airy / (integral + 1e-12)) * energy

    def intensity_unpolarized(self, m: complex, x: float, mu: np.ndarray) -> np.ndarray:
        num_requested = len(mu)
        nyquist_limit = int(x * 4.0)
        internal_bins = max(8192, nyquist_limit, num_requested * 2)

        rng = PCG32(size=self.num_rays)
        u = rng.next_float32()
        alpha = dr.asin(dr.sqrt(u))
        n_real = Float(m.real)
        beta = dr.asin(dr.sin(alpha) / n_real)

        cos_alpha, cos_beta = dr.cos(alpha), dr.cos(beta)
        R_s = ((cos_alpha - n_real * cos_beta) / (cos_alpha + n_real * cos_beta)) ** 2
        R_p = ((n_real * cos_alpha - cos_beta) / (n_real * cos_alpha + cos_beta)) ** 2
        T_s, T_p = 1.0 - R_s, 1.0 - R_p

        histogram = dr.zeros(Float, internal_bins)
        # Separate energy bins for Airy treatment
        energy_p2 = dr.zeros(Float, 1)
        energy_p3 = dr.zeros(Float, 1)

        def add_to_hist(theta_val, energy_val):
            bin_float = theta_val * (Float(internal_bins - 1) / dr.pi)
            bin_idx = UInt32(bin_float + 0.5)
            bin_idx = dr.minimum(bin_idx, internal_bins - 1)
            dr.scatter_add(histogram, energy_val, bin_idx)

        # p=0
        add_to_hist(dr.pi - 2.0 * alpha, 0.5 * (R_s + R_p))

        e_s, e_p = T_s, T_p
        for p in range(1, self.max_bounces + 1):
            phi = 2.0 * Float(p) * beta - 2.0 * alpha + (Float(p) - 1.0) * dr.pi
            phi_wrapped = phi - dr.floor(phi * (0.5 / dr.pi) + 0.5) * (2.0 * dr.pi)
            theta = dr.abs(phi_wrapped)

            out_total = 0.5 * (e_s * T_s + e_p * T_p)

            # --- TARGETED EXTRACTION ---
            if p == 2:
                if not self.suppress_p2:
                    dr.scatter_add(energy_p2, out_total, UInt32(0))
            elif p == 3:
                dr.scatter_add(energy_p3, out_total, UInt32(0))
            else:
                add_to_hist(theta, out_total)

            e_s *= R_s
            e_p *= R_p

        # --- 3. POST-PROCESS ---
        theta_internal = np.linspace(0, np.pi, internal_bins)
        dtheta = np.pi / (internal_bins - 1)

        # Solid angle normalization
        theta_m = np.maximum(0.0, theta_internal - dtheta / 2.0)
        theta_p = np.minimum(np.pi, theta_internal + dtheta / 2.0)
        solid_angles = 2.0 * np.pi * (np.cos(theta_m) - np.cos(theta_p))

        intensity_base = (histogram.numpy() / self.num_rays) / np.maximum(solid_angles, 1e-12)

        diffraction = np.zeros_like(theta_internal)
        sin_theta = np.sin(theta_internal)
        mask = theta_internal < 1e-7
        diffraction[mask] = (x ** 4) / 4.0
        u_val = x * sin_theta[~mask]
        obliquity = ((1.0 + np.cos(theta_internal[~mask])) / 2.0) ** 2
        diffraction[~mask] = ((x * j1(u_val) / sin_theta[~mask]) ** 2) * obliquity

        p2_total = energy_p2.numpy()[0] / self.num_rays
        p3_total = energy_p3.numpy()[0] / self.num_rays

        intensity_p2 = self.apply_airy_lobe(theta_internal, p2_total, m, x, p=2)
        intensity_p3 = self.apply_airy_lobe(theta_internal, p3_total, m, x, p=3)

        total_intensity = intensity_base + diffraction + intensity_p2 + intensity_p3
        total_int = np.sum(total_intensity * solid_angles)
        normalized = total_intensity / (total_int + 1e-12)

        theta_requested = np.arccos(np.clip(mu, -1.0, 1.0))
        return np.interp(theta_requested, theta_internal, normalized)
import drjit as dr
from drjit.llvm import Float, UInt32, PCG32
import numpy as np
from scipy.special import airy
from scipy.signal import convolve
from .base import ScatteringBackend


class DrJitRaytracerBackend(ScatteringBackend):
    name = "drjit_raytracer"

    def __init__(self, num_rays=10_000_000, max_bounces=6, suppress_p2=False):
        self.num_rays = num_rays
        self.max_bounces = max_bounces
        self.suppress_p2 = suppress_p2

    def _generate_airy_kernel(self, num_bins, x_param):
        # z = (theta - theta_caustic) * x^(2/3)
        scale_factor = np.power(x_param, 2.0 / 3.0)
        angular_res_rad = np.pi / num_bins

        # Keep the window tight to preserve performance
        window_rad = np.radians(4.0)
        half_width_bins = int(window_rad / angular_res_rad)
        kernel_size = 2 * half_width_bins + 1

        theta_offsets = np.linspace(-window_rad, window_rad, kernel_size)
        z = theta_offsets * scale_factor

        ai_val, _, _, _ = airy(z)
        kernel = ai_val ** 2
        return kernel / (np.sum(kernel) + 1e-12)

    def intensity_unpolarized(self, m: complex, x: float, mu_np: np.ndarray) -> np.ndarray:
        TwoPi = 2.0 * dr.pi
        InvTwoPi = 1.0 / TwoPi
        num_bins = len(mu_np)

        # --- 1. GEOMETRIC TRACE ---
        rng = PCG32(size=self.num_rays)
        u = rng.next_float32()
        alpha = dr.asin(dr.sqrt(u))

        n_real = Float(m.real)
        beta = dr.asin(dr.sin(alpha) / n_real)

        cos_alpha, cos_beta = dr.cos(alpha), dr.cos(beta)
        rs = (cos_alpha - n_real * cos_beta) / (cos_alpha + n_real * cos_beta)
        rp = (n_real * cos_alpha - cos_beta) / (n_real * cos_alpha + cos_beta)
        R_val = 0.5 * (rs ** 2 + rp ** 2)
        T_val = 1.0 - R_val

        histogram = dr.zeros(Float, num_bins)

        # p=0 (Reflection)
        theta_0 = dr.pi - 2.0 * alpha
        dr.scatter_add(histogram, R_val, UInt32(theta_0 * (num_bins - 1) / dr.pi))

        # Internal Bounces
        energy = T_val
        for p in range(1, self.max_bounces + 1):
            # Standard GO deflection formula: Theta = 2(p*beta - alpha) + (p-1)*pi
            phi = 2.0 * Float(p) * beta - 2.0 * alpha + (Float(p) - 1.0) * dr.pi

            # Robust wrapping to [0, Pi]
            # This correctly maps any deflection into the scattering angle space
            phi_wrapped = phi - dr.floor(phi * (0.5 / dr.pi) + 0.5) * (2.0 * dr.pi)
            theta = dr.abs(phi_wrapped)

            energy_out = energy * T_val
            if self.suppress_p2 and p == 2:
                energy_out = Float(0.0)

            # Use a safer floor-based binning for your 8192 bins
            bin_idx = UInt32(theta * (Float(num_bins - 1) / dr.pi))
            bin_idx = dr.minimum(bin_idx, num_bins - 1)

            dr.scatter_add(histogram, energy_out, bin_idx)
            energy *= R_val

        # --- 2. POST-PROCESS ---
        theta_grid = np.linspace(0, np.pi, num_bins)
        # Convert energy density to intensity (1/sin factor)
        raw_geo = histogram.numpy() / self.num_rays
        intensity_geo = raw_geo / np.maximum(np.sin(theta_grid), 1e-7)

        # Apply Airy kernel (Smooths the 138-degree caustic and others)
        intensity_rainbow = convolve(intensity_geo, self._generate_airy_kernel(num_bins, x), mode='same')

        # --- 3. PURE GEOMETRIC NORMALIZATION ---
        # Ensure the phase function integrates to 1.0 over the sphere
        total_int = np.trapezoid(intensity_rainbow * np.sin(theta_grid), theta_grid) * 2 * np.pi

        return intensity_rainbow / (total_int + 1e-12)
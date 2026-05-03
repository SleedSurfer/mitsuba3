import drjit as dr
import mitsuba as mi
import numpy as np

# Ensure we are in the right variant for testing
mi.set_variant('llvm_spectral')


class vMFLobe:
    def __init__(self, mu: mi.Vector3f, kappa: mi.Float):
        self.mu = dr.normalize(mu)
        self.kappa = dr.maximum(kappa, 0.0)

    def pdf(self, omega: mi.Vector3f) -> mi.Float:
        cos_theta = dr.dot(self.mu, omega)
        is_uniform = self.kappa < 1e-4

        # Use safe_kappa to avoid NaN poisoning in the graph
        safe_kappa = dr.maximum(self.kappa, 1e-4)
        norm_factor = safe_kappa / (dr.two_pi * (1.0 - dr.exp(-2.0 * safe_kappa)))
        exp_term = dr.exp(safe_kappa * (cos_theta - 1.0))

        pdf_val = norm_factor * exp_term
        return dr.select(is_uniform, dr.inv_four_pi, pdf_val)

    def sample(self, sample2: mi.Point2f) -> tuple[mi.Vector3f, mi.Float]:
        xi_1, xi_2 = sample2.x, sample2.y

        # Clamp BEFORE division so Dr.Jit doesn't calculate 1.0/0.0 on the inactive lanes
        safe_kappa = dr.maximum(self.kappa, 1e-4)
        W = 1.0 + (1.0 / safe_kappa) * dr.log(xi_1 + (1.0 - xi_1) * dr.exp(-2.0 * safe_kappa))
        W = dr.select(self.kappa < 1e-4, 1.0 - 2.0 * xi_1, W)

        sin_theta = dr.safe_sqrt(1.0 - W * W)
        phi = dr.two_pi * xi_2
        s, c = dr.sincos(phi)

        local_dir = mi.Vector3f(sin_theta * c, sin_theta * s, W)
        frame = mi.Frame3f(self.mu)
        world_dir = frame.to_world(local_dir)

        return world_dir, self.pdf(world_dir)
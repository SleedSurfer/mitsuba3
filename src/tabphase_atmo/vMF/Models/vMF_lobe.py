import drjit as dr
import mitsuba as mi
import numpy as np

# Ensure we are in the right variant for testing
mi.set_variant('llvm_spectral')


class vMFLobe:
    def __init__(self, mu: mi.Vector3f, kappa: mi.Float):
        """
        mu: The central direction of the lobe (must be normalized)
        kappa: The concentration parameter (0 = uniform sphere, >100 = laser beam)
        """
        self.mu = dr.normalize(mu)
        self.kappa = dr.maximum(kappa, 0.0)  # Prevent negative kappa corruption

    def pdf(self, omega: mi.Vector3f) -> mi.Float:
        """
        Evaluates the PDF of the vMF distribution for a given direction.
        Standard math:  p = (kappa / (4 * pi * sinh(kappa))) * exp(kappa * dot(mu, omega))
        Stable math:    p = (kappa / (2 * pi * (1 - exp(-2*kappa)))) * exp(kappa * (dot(mu, omega) - 1))
        """
        cos_theta = dr.dot(self.mu, omega)

        # Handle the uniform sphere edge case to prevent division by zero
        is_uniform = self.kappa < 1e-4

        # Numerically stable evaluation
        norm_factor = self.kappa / (dr.two_pi * (1.0 - dr.exp(-2.0 * self.kappa)))
        exp_term = dr.exp(self.kappa * (cos_theta - 1.0))

        pdf_val = norm_factor * exp_term

        # Fallback to 1/(4*pi) if kappa is basically zero
        return dr.select(is_uniform, dr.inv_four_pi, pdf_val)

    def sample(self, sample2: mi.Point2f) -> tuple[mi.Vector3f, mi.Float]:
        """
        Samples a direction from the vMF lobe and returns (direction, pdf).
        """
        xi_1, xi_2 = sample2.x, sample2.y

        # Stable inversion of the CDF to find the cosine of the angle from mu
        # W = 1 + (1 / kappa) * ln(xi_1 + (1 - xi_1) * exp(-2 * kappa))
        W = 1.0 + (1.0 / self.kappa) * dr.log(xi_1 + (1.0 - xi_1) * dr.exp(-2.0 * self.kappa))

        # Uniform sphere fallback for W
        W = dr.select(self.kappa < 1e-4, 1.0 - 2.0 * xi_1, W)

        # Derive the sine and azimuthal angle
        sin_theta = dr.safe_sqrt(1.0 - W * W)
        phi = dr.two_pi * xi_2

        s, c = dr.sincos(phi)

        # Build the direction in local tangent space (Z is the forward axis)
        local_dir = mi.Vector3f(sin_theta * c, sin_theta * s, W)

        # Construct a coordinate frame around our mean direction and transform to world space
        frame = mi.Frame3f(self.mu)
        world_dir = frame.to_world(local_dir)

        return world_dir, self.pdf(world_dir)

    @classmethod
    def fit_to_data(cls, directions: mi.Vector3f, weights: mi.Float, active: mi.Bool = True) -> 'vMFLobe':
        """
        Estimates the optimal mu and kappa for a single lobe given a batch of weighted directions.
        """
        # Sum of all weights
        W = dr.sum(dr.select(active, weights, 0.0))

        # Weighted sum of direction vectors
        R_x = dr.sum(dr.select(active, directions.x * weights, 0.0))
        R_y = dr.sum(dr.select(active, directions.y * weights, 0.0))
        R_z = dr.sum(dr.select(active, directions.z * weights, 0.0))
        R = mi.Vector3f(R_x, R_y, R_z)

        # Prevent division by zero if there's no data
        valid_data = W > 1e-6
        r_vec = dr.select(valid_data, R / W, mi.Vector3f(0, 0, 1))

        # Length of the average vector (r_bar) determines our concentration
        r_bar = dr.norm(r_vec)

        # New mean direction is just the normalized average vector
        mu = dr.select(r_bar > 1e-6, r_vec / r_bar, mi.Vector3f(0, 0, 1))

        # Banerjee approximation for kappa in 3D: (r_bar * (3 - r_bar^2)) / (1 - r_bar^2)
        # Cap r_bar slightly below 1.0 to prevent infinite kappa blowouts
        r_bar_safe = dr.minimum(r_bar, 0.9999)
        r2 = r_bar_safe * r_bar_safe

        kappa = (r_bar_safe * (3.0 - r2)) / (1.0 - r2)
        kappa = dr.select(valid_data, kappa, 0.0)

        return cls(mu, kappa)
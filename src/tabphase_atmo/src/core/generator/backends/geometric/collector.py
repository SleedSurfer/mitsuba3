import drjit as dr
from drjit.auto import Float, Bool, Complex2f, Array3f, UInt32
import numpy as np

from .models.phasor_ray import PhasorRay
from .models.ray_patch import RayPatch


class CollectionSphere:
    """
    Collects exiting wavefront patches and bins them by scattering angle (μ = cos θ).
    Accumulates complex E-field amplitudes with proper phase, then computes intensity.
    """

    def __init__(self, mu_bins: np.ndarray, wavelength_nm: float):
        self.mu_bins_np = mu_bins
        self.num_bins = len(mu_bins)
        wavelength_mm = wavelength_nm / 1e6
        self.k = (2.0 * np.pi) / wavelength_mm

        # Pre-compute solid angles for each bin
        # μ goes from 1 to -1, so bin edges are:
        theta = np.arccos(mu_bins)
        dtheta = np.pi / (self.num_bins - 1)
        theta_lo = np.maximum(0, theta - dtheta / 2)
        theta_hi = np.minimum(np.pi, theta + dtheta / 2)
        # Solid angle = 2π(cos(θ_lo) - cos(θ_hi))
        self._solid_angles = 2.0 * np.pi * (np.cos(theta_lo) - np.cos(theta_hi))
        self._solid_angles = np.maximum(self._solid_angles, 1e-12)  # Avoid div by zero

        # Persistent accumulators
        self._bins_ex_real = None
        self._bins_ex_imag = None
        self._bins_ey_real = None
        self._bins_ey_imag = None
        self._total_patches = 0

    def reset(self):
        """Reset accumulators for a new collection pass."""
        self._bins_ex_real = dr.zeros(Float, self.num_bins)
        self._bins_ex_imag = dr.zeros(Float, self.num_bins)
        self._bins_ey_real = dr.zeros(Float, self.num_bins)
        self._bins_ey_imag = dr.zeros(Float, self.num_bins)
        self._total_patches = 0

    def _gather_ray_data(self, rays: PhasorRay, indices: UInt32):
        d_z = dr.gather(Float, rays.d.z, indices)
        l = dr.gather(Float, rays.l, indices)
        f = Float(dr.gather(type(rays.f), rays.f, indices))

        ex = Complex2f(
            dr.gather(Float, rays.Ex.real, indices),
            dr.gather(Float, rays.Ex.imag, indices)
        )
        ey = Complex2f(
            dr.gather(Float, rays.Ey.real, indices),
            dr.gather(Float, rays.Ey.imag, indices)
        )
        return d_z, l, f, ex, ey

    def _compute_phasor(self, l: Float, f: Float, ex: Complex2f, ey: Complex2f):
        phi = self.k * l - f * (np.pi / 2.0)
        two_pi = Float(2.0 * np.pi)
        phi = phi - dr.floor(phi / two_pi + 0.5) * two_pi
        rotation = Complex2f(dr.cos(phi), dr.sin(phi))
        return ex * rotation, ey * rotation

    def _mu_to_bin_index(self, mu: Float) -> UInt32:
        normalized = (1.0 - mu) / 2.0
        bin_float = normalized * Float(self.num_bins - 1)
        return UInt32(dr.clip(bin_float, 0.0, Float(self.num_bins - 1)))

    def accumulate(self, rays: PhasorRay, patches: RayPatch, patch_area: float):
        """
        Accumulates complex amplitudes from one wavefront into the bins.
        """
        if self._bins_ex_real is None:
            self.reset()

        num_patches = len(patches.v0)
        self._total_patches += num_patches

        # Gather and compute phasors for all 4 corners
        mu0, l0, f0, ex0, ey0 = self._gather_ray_data(rays, patches.v0)
        mu1, l1, f1, ex1, ey1 = self._gather_ray_data(rays, patches.v1)
        mu2, l2, f2, ex2, ey2 = self._gather_ray_data(rays, patches.v2)
        mu3, l3, f3, ex3, ey3 = self._gather_ray_data(rays, patches.v3)

        px0, py0 = self._compute_phasor(l0, f0, ex0, ey0)
        px1, py1 = self._compute_phasor(l1, f1, ex1, ey1)
        px2, py2 = self._compute_phasor(l2, f2, ex2, ey2)
        px3, py3 = self._compute_phasor(l3, f3, ex3, ey3)

        # Patch center μ
        mu_center = (mu0 + mu1 + mu2 + mu3) * 0.25
        area_weight = Float(0.25 * patch_area)
        # Area-Weighted Average E-Field!
        # E_contribution = E_avg * dA
        ex_avg = (px0 + px1 + px2 + px3) * area_weight
        ey_avg = (py0 + py1 + py2 + py3) * area_weight

        # Validity mask
        amp_threshold = Float(1e-6)
        amp0 = dr.abs(ex0) + dr.abs(ey0)
        amp1 = dr.abs(ex1) + dr.abs(ey1)
        amp2 = dr.abs(ex2) + dr.abs(ey2)
        amp3 = dr.abs(ex3) + dr.abs(ey3)

        # MUST BE VALID AND NOT NAN
        valid = (amp0 > amp_threshold) & (amp1 > amp_threshold) & \
                (amp2 > amp_threshold) & (amp3 > amp_threshold) & \
                dr.isfinite(mu_center)

        bin_idx = self._mu_to_bin_index(mu_center)

        # THE SAVIOR MASK - Do not remove 'valid'
        dr.scatter_add(self._bins_ex_real, ex_avg.real, bin_idx, valid)
        dr.scatter_add(self._bins_ex_imag, ex_avg.imag, bin_idx, valid)
        dr.scatter_add(self._bins_ey_real, ey_avg.real, bin_idx, valid)
        dr.scatter_add(self._bins_ey_imag, ey_avg.imag, bin_idx, valid)

    def finalize(self) -> np.ndarray:
        # Compute raw intensity: I = |E_total|^2
        intensity = (self._bins_ex_real ** 2 + self._bins_ex_imag ** 2 +
                     self._bins_ey_real ** 2 + self._bins_ey_imag ** 2)

        result = np.array(intensity)

        # Normalize by solid angle (Intensity per steradian)
        result /= self._solid_angles

        self.reset()
        return result
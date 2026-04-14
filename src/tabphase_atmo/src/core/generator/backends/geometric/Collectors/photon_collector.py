import drjit as dr
from drjit.auto import Float, UInt32, Complex2f, Bool
import numpy as np
import scipy.ndimage as ndimage


class PhotonCollectionSphere:
    def __init__(self, theta_bins: np.ndarray, num_phi_bins: int):
        self.num_theta_bins = len(theta_bins)
        self.num_phi_bins = num_phi_bins
        self.total_bins = self.num_theta_bins * self.num_phi_bins

        # Solid angle of each bin (used to normalize intensity at the end)
        theta = theta_bins
        dtheta = np.pi / (self.num_theta_bins - 1)
        dphi = (2.0 * np.pi) / self.num_phi_bins

        theta_lo = np.maximum(0, theta - dtheta / 2.0)
        theta_hi = np.minimum(np.pi, theta + dtheta / 2.0)
        self._solid_angles_1d = dphi * (np.cos(theta_lo) - np.cos(theta_hi))
        self._solid_angles_1d = np.maximum(self._solid_angles_1d, 1e-12)

        self.reset()

    def reset(self):
        self.bins_intensity = dr.zeros(Float, self.total_bins)

    def accumulate(self, rays, hit_mask: Bool):
        # 1. Get direction of surviving rays
        dx, dy, dz = rays.direction.x, rays.direction.y, rays.direction.z

        # 2. Convert to spherical coordinates
        theta = dr.acos(dr.clip(dz, -1.0, 1.0))
        phi = dr.atan2(dy, dx)

        # 3. Map to array indices (Using floor to prevent rounding artifacts)
        theta_coord = (theta / Float(np.pi)) * Float(self.num_theta_bins - 1)
        normalized_phi = (phi + Float(np.pi)) / Float(2.0 * np.pi)
        phi_coord = normalized_phi * Float(self.num_phi_bins)

        th_idx = dr.clip(UInt32(dr.floor(theta_coord)), 0, self.num_theta_bins - 1)
        phi_idx = UInt32(dr.floor(phi_coord)) % self.num_phi_bins

        flat_idx = phi_idx * self.num_theta_bins + th_idx

        # 4. Calculate pure energy (magnitude squared of the E field)
        energy = dr.squared_norm(rays.Ex) + dr.squared_norm(rays.Ey)

        # 5. Scatter Add (Atomic add on the GPU/CPU)
        dr.scatter_add(self.bins_intensity, energy, flat_idx, hit_mask)

    def finalize(self) -> np.ndarray:
        intensity_1d = np.array(self.bins_intensity)
        # Use the new naming
        intensity_2d = intensity_1d.reshape((self.num_phi_bins, self.num_theta_bins))

        # 1. Convert to Density FIRST
        omega_2d = np.tile(self._solid_angles_1d, (self.num_phi_bins, 1))
        density_raw = intensity_2d / (omega_2d + 1e-12)

        # 2. Blur the density (kills noise without creating black holes)
        target_blur_deg = 0.0
        deg_per_theta_bin = 180.0 / float(max(1, self.num_theta_bins - 1))
        deg_per_phi_bin = 360.0 / float(max(1, self.num_phi_bins))

        sigma_theta = max(1.0, target_blur_deg / deg_per_theta_bin)
        sigma_phi = max(1.0, target_blur_deg / deg_per_phi_bin)

        density_2d = ndimage.gaussian_filter(
            density_raw,
            sigma=(sigma_phi, sigma_theta),
            mode=['wrap', 'nearest']
        )

        self.reset()
        return density_2d

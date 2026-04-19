from src.core.config import BaseParticleConfig, HexagonalConfig, DropletConfig
import numpy as np


def compute_optical_weight(config: BaseParticleConfig, comp_item) -> float:
    """Converts user mass fraction (weight) to true scattering optical weight based on Area/Volume."""
    if isinstance(config, DropletConfig):
        shape_enum, weight, radius, variance = comp_item

        area_vol_ratio = 0.75 / max(radius, 1e-6)

    else:
        shape_enum, weight, c_axis, a_axis, variance = comp_item

        base_area = (3.0 * np.sqrt(3.0) / 2.0) * (a_axis ** 2)
        volume = base_area * c_axis

        s_total = (2.0 * base_area) + (6.0 * a_axis * c_axis)
        projected_area = s_total / 4.0

        area_vol_ratio = projected_area / max(volume, 1e-6)

    return weight * area_vol_ratio


def normalize_macroscopic_phase(phase_table: np.ndarray, mu: np.ndarray, num_phi_bins: int) -> np.ndarray:
    """Integrates and normalizes the final combined energy volume."""
    theta_linear = np.arccos(mu)

    if num_phi_bins > 1:
        d_phi = (2.0 * np.pi) / num_phi_bins
        theta_integrals = np.trapezoid(phase_table * np.sin(theta_linear), theta_linear, axis=2)
        global_integral = np.sum(theta_integrals * d_phi, axis=1, keepdims=True)
        return (phase_table / (global_integral[..., np.newaxis] + 1e-12)).astype(np.float32)
    else:
        global_integral = np.trapezoid(phase_table * np.sin(theta_linear), theta_linear, axis=1) * 2.0 * np.pi
        return (phase_table / (global_integral[:, np.newaxis] + 1e-12)).astype(np.float32)


def apply_similarity_truncation(phase_table, mu, num_phi_bins, threshold=50.0):
    """
    1. Normalize to unit-energy phase per wavelength.
    2. Back hemisphere (>90deg): clip at threshold, redistribute removed energy isotropically.
    3. Forward hemisphere (<90deg): clip at threshold, report removed energy fraction f.
    4. Renormalize for a valid 1.0 PDF per wavelength.
    """
    phase = normalize_macroscopic_phase(phase_table, mu, num_phi_bins)
    theta = np.arccos(mu)
    sin_theta = np.sin(theta)

    if num_phi_bins > 1:
        # phase layout: (W, P, T)
        forward_mask = (theta < (np.pi / 2.0))[None, None, :]   # (1,1,T)
        back_mask = ~forward_mask
        d_phi = (2.0 * np.pi) / num_phi_bins

        # Backscatter clipping and removed-energy integral
        back_excess = np.maximum(0.0, np.where(back_mask, phase - threshold, 0.0))
        back_integrals_theta = np.trapezoid(
            back_excess * sin_theta[None, None, :], theta, axis=2
        )  # (W, P)
        back_energy_loss = np.sum(back_integrals_theta * d_phi, axis=1)  # (W,)

        phase = np.where(back_mask, np.minimum(phase, threshold), phase)
        phase += (back_energy_loss / (4.0 * np.pi))[:, None, None]

        # Forward clipping and f-fraction integral
        forward_excess = np.maximum(0.0, np.where(forward_mask, phase - threshold, 0.0))
        f_integrals_theta = np.trapezoid(
            forward_excess * sin_theta[None, None, :], theta, axis=2
        )  # (W, P)
        f_fractions = np.sum(f_integrals_theta * d_phi, axis=1)  # (W,)

        phase = np.where(forward_mask, np.minimum(phase, threshold), phase)

    else:
        # phase layout: (W, T)
        forward_mask = (theta < (np.pi / 2.0))[None, :]          # (1,T)
        back_mask = ~forward_mask

        back_excess = np.maximum(0.0, np.where(back_mask, phase - threshold, 0.0))
        back_energy_loss = (
            np.trapezoid(back_excess * sin_theta[None, :], theta, axis=1) * 2.0 * np.pi
        )  # (W,)

        phase = np.where(back_mask, np.minimum(phase, threshold), phase)
        phase += (back_energy_loss / (4.0 * np.pi))[:, None]

        forward_excess = np.maximum(0.0, np.where(forward_mask, phase - threshold, 0.0))
        f_fractions = (
            np.trapezoid(forward_excess * sin_theta[None, :], theta, axis=1) * 2.0 * np.pi
        )  # (W,)

        phase = np.where(forward_mask, np.minimum(phase, threshold), phase)

    final_phase = normalize_macroscopic_phase(phase, mu, num_phi_bins)
    return final_phase, f_fractions

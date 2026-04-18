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
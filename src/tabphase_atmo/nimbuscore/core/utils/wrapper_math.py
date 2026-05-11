from nimbuscore.core.config import BaseParticleConfig, HexagonalConfig, DropletConfig, CuboctahedralConfig
import numpy as np


def compute_optical_weight(config: BaseParticleConfig, comp_item) -> float:
    """Converts user mass fraction (weight) to true scattering optical weight based on Area/Volume."""

    if isinstance(config, DropletConfig):
        weight = comp_item.weight
        radius = comp_item.radius_mean_um

        area_vol_ratio = 0.75 / max(radius, 1e-6)

    elif isinstance(config, CuboctahedralConfig):
        weight = comp_item.weight
        a_axis = comp_item.a_axis_um  # Distance from center to square face (H)

        # Exact mathematical volume and area of a cuboctahedron
        volume = (20.0 / 3.0) * (a_axis ** 3)
        s_total = (12.0 + 4.0 * np.sqrt(3.0)) * (a_axis ** 2)
        projected_area = s_total / 4.0

        area_vol_ratio = projected_area / max(volume, 1e-6)

    elif isinstance(config, HexagonalConfig):
        weight = comp_item.weight
        c_axis = comp_item.length_axis_um
        a_axis = comp_item.width_axis_um

        base_area = (3.0 * np.sqrt(3.0) / 2.0) * (a_axis ** 2)
        volume = base_area * c_axis

        s_total = (2.0 * base_area) + (6.0 * a_axis * c_axis)
        projected_area = s_total / 4.0

        area_vol_ratio = projected_area / max(volume, 1e-6)

    else:
        raise TypeError(f"Unknown config type passed to compute_optical_weight: {type(config)}")

    return weight * area_vol_ratio


def normalize_macroscopic_phase(phase_table: np.ndarray, mu: np.ndarray, num_phi_bins: int) -> np.ndarray:
    """
    Pure Intensity Normalization.
    Integrates Phase * sin(theta) over the sphere so the total energy = 1.0.
    """
    theta_linear = np.arccos(mu)
    sin_theta = np.sin(theta_linear)

    if num_phi_bins > 1:
        d_phi = (2.0 * np.pi) / num_phi_bins
        # 1. Integrate over theta: Phase * sin(theta) d_theta
        theta_integrals = np.trapezoid(phase_table * sin_theta, theta_linear, axis=2)

        # 2. Integrate over phi
        global_integral = np.sum(theta_integrals * d_phi, axis=1, keepdims=True)

        return (phase_table / (global_integral[..., np.newaxis] + 1e-12)).astype(np.float32)
    else:
        # 1. Integrate over theta: Phase * sin(theta) d_theta
        # 2. Multiply by 2*PI because phi is uniform
        global_integral = np.trapezoid(phase_table * sin_theta, theta_linear, axis=1) * (2.0 * np.pi)

        return (phase_table / (global_integral[:, np.newaxis] + 1e-12)).astype(np.float32)


def apply_similarity_truncation(phase_table, mu, num_phi_bins, threshold=50.0, forward_angle_deg=5.0):
    """
    1. Normalize to unit-energy phase per wavelength.
    2. Isolate ONLY the extreme forward peak (default: first 5 degrees).
    3. Clip the forward peak at the given threshold and calculate removed energy fraction (f).
    4. Leave everything else (halos, rainbows, glories) completely untouched.
    5. Renormalize for a valid 1.0 PDF per wavelength.
    """
    phase = normalize_macroscopic_phase(phase_table, mu, num_phi_bins)
    theta = np.arccos(mu)
    sin_theta = np.sin(theta)

    forward_angle_limit = np.radians(forward_angle_deg)

    if num_phi_bins > 1:
        # phase layout: (W, P, T)
        forward_mask = (theta < forward_angle_limit)[None, None, :]  # (1,1,T)
        d_phi = (2.0 * np.pi) / num_phi_bins

        # Forward clipping and f-fraction integral
        forward_excess = np.maximum(0.0, np.where(forward_mask, phase - threshold, 0.0))
        f_integrals_theta = np.trapezoid(
            forward_excess * sin_theta[None, None, :], theta, axis=2
        )  # (W, P)
        f_fractions = np.sum(f_integrals_theta * d_phi, axis=1)  # (W,)

        # Apply threshold ONLY to the forward peak
        phase = np.where(forward_mask, np.minimum(phase, threshold), phase)

    else:
        # phase layout: (W, T)
        forward_mask = (theta < forward_angle_limit)[None, :]  # (1,T)

        # Forward clipping and f-fraction integral
        forward_excess = np.maximum(0.0, np.where(forward_mask, phase - threshold, 0.0))
        f_fractions = (
                np.trapezoid(forward_excess * sin_theta[None, :], theta, axis=1) * 2.0 * np.pi
        )  # (W,)

        # Apply threshold ONLY to the forward peak
        phase = np.where(forward_mask, np.minimum(phase, threshold), phase)

    # Renormalize to ensure the remaining phase function integrates back to 1.0
    final_phase = normalize_macroscopic_phase(phase, mu, num_phi_bins)

    return final_phase, f_fractions

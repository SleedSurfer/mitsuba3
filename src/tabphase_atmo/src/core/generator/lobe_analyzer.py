"""
MIS (Multiple Importance Sampling) metadata analyzer for atmospheric phase functions.
Detects lobes (forward, rainbow, residual) and exports metadata for runtime sampling.
"""

import struct
import numpy as np
from scipy.signal import find_peaks

# Metadata format constants
MIS_MAGIC = b'MISD'  # 4 bytes:  MIS Data marker
MIS_VERSION = 1  # uint8

# Lobe type enumeration
LOBE_FORWARD = 0
LOBE_RAINBOW = 1
LOBE_RESIDUAL = 2
LOBE_GLORY = 3


def detect_lobes_for_wavelength(phase_1d, mu_vals):
    """Detect lobes using log-space on forward-masked data."""
    lobes = {}

    # --- FORWARD LOBE ---
    forward_idx = 0
    forward_val = phase_1d[forward_idx]
    forward_width = estimate_lobe_width(phase_1d, forward_idx, threshold=0.5)

    lobes['forward'] = {
        'mu': mu_vals[forward_idx],
        'amplitude': forward_val,
        'width': forward_width
    }

    # --- PREPARE MASKED LOG-SPACE DATA ---
    # Mask forward peak
    forward_mask = mu_vals > 0.9
    phase_masked = phase_1d.copy()
    phase_masked[forward_mask] = 1e-10  # Small value for log

    # Convert to log space
    phase_log = np.log10(phase_masked + 1e-10)

    # --- RAINBOW LOBE ---
    # Primary rainbow is ~138 deg (mu = -0.743)
    # Range: -0.85 to -0.55 covers primary and secondary comfortably
    rainbow_mask = (mu_vals >= -0.85) & (mu_vals <= -0.55)
    rainbow_indices = np.where(rainbow_mask)[0]

    if len(rainbow_indices) > 10:
        rainbow_log = phase_log[rainbow_indices]

        peaks, properties = find_peaks(
            rainbow_log,
            prominence=0.01,  # Very sensitive in log space
            distance=10
        )

        if len(peaks) > 0:
            # Take peak with highest prominence (not height)
            strongest_idx = peaks[np.argmax(properties['prominences'])]
            actual_idx = rainbow_indices[strongest_idx]

            lobes['rainbow'] = {
                'mu': mu_vals[actual_idx],
                'amplitude': phase_1d[actual_idx],
                'width': estimate_lobe_width(phase_1d, actual_idx, threshold=0.5)
            }
        else:
            lobes['rainbow'] = None
    else:
        lobes['rainbow'] = None

    # --- GLORY LOBE ---
    backscatter_mask = mu_vals <= -0.8
    backscatter_indices = np.where(backscatter_mask)[0]

    if len(backscatter_indices) > 5:
        backscatter_log = phase_log[backscatter_indices]

        peaks, properties = find_peaks(
            backscatter_log,
            prominence=0.03
        )

        if len(peaks) > 0:
            strongest_idx = peaks[np.argmax(properties['prominences'])]
            actual_idx = backscatter_indices[strongest_idx]

            lobes['glory'] = {
                'mu': mu_vals[actual_idx],
                'amplitude': phase_1d[actual_idx],
                'width': estimate_lobe_width(phase_1d, actual_idx, threshold=0.5)
            }
        else:
            lobes['glory'] = None
    else:
        lobes['glory'] = None

    return lobes


def estimate_lobe_width(phase_1d, peak_idx, threshold=0.5):
    """Estimate lobe width using FWHM."""
    peak_val = phase_1d[peak_idx]
    target = peak_val * threshold

    left = peak_idx
    while left > 0 and phase_1d[left] > target:
        left -= 1

    right = peak_idx
    while right < len(phase_1d) - 1 and phase_1d[right] > target:
        right += 1

    width_indices = right - left
    delta_mu = width_indices * (2.0 / len(phase_1d))

    return max(delta_mu, 0.01)


def width_to_kappa(width):
    """Convert lobe width (Δμ) to Gaussian concentration parameter κ."""
    sigma = width / 2.355  # FWHM to std deviation
    kappa = 1.0 / (2.0 * sigma ** 2 + 1e-8)
    return kappa


def analyze_lut_for_mis(phase_table, mu_vals, wavelengths):
    """
    Analyze a full LUT and extract MIS lobe parameters.
    CLAMPS KAPPA VALUES TO PREVENT FIREFLIES.
    """
    num_wavelengths = phase_table.shape[0]

    forward_lobes = []
    rainbow_lobes = []
    glory_lobes = []

    for wl_idx in range(num_wavelengths):
        phase_1d = phase_table[wl_idx, :]
        lobes_wl = detect_lobes_for_wavelength(phase_1d, mu_vals)

        if lobes_wl['forward']:
            forward_lobes.append(lobes_wl['forward'])
        if lobes_wl['rainbow']:
            rainbow_lobes.append(lobes_wl['rainbow'])
        if lobes_wl['glory']:
            glory_lobes.append(lobes_wl['glory'])

    # --- FORWARD LOBE ---
    forward_mu = np.mean([l['mu'] for l in forward_lobes])
    forward_amp = np.mean([l['amplitude'] for l in forward_lobes])
    forward_width = np.mean([l['width'] for l in forward_lobes])

    # CLAMP: 5000 is plenty sharp for a sun glare.
    # Prevents numerical instability if width is estimated as near-zero.
    forward_kappa = min(width_to_kappa(forward_width), 5000.0)

    forward_lobe_descriptor = {
        'type': LOBE_FORWARD,
        'mu_center': forward_mu,
        'kappa': forward_kappa,
        'amplitude': forward_amp,
        'wavelength_dependent': False
    }

    # --- RAINBOW LOBE (Operation Thicc Rainbow) ---
    rainbow_lobe_descriptor = None
    if len(rainbow_lobes) > 0:
        rainbow_mu_per_wl = np.array([l['mu'] for l in rainbow_lobes])

        # Calculate raw kappas from detected widths
        raw_kappas = np.array([width_to_kappa(l['width']) for l in rainbow_lobes])

        # CLAMP: Limit kappa to 600.0 (approx 2-3 degrees width).
        # This acts as a "Shotgun", forcing the sampler to cover the entire rainbow structure
        # (including ripples) rather than snipping a single interference spike.
        rainbow_kappa_per_wl = np.minimum(raw_kappas, 600.0)

        rainbow_amp_mean = np.mean([l['amplitude'] for l in rainbow_lobes])

        rainbow_lobe_descriptor = {
            'type': LOBE_RAINBOW,
            'mu_center': np.mean(rainbow_mu_per_wl),
            'kappa': np.mean(rainbow_kappa_per_wl), # Use the clamped mean
            'amplitude': rainbow_amp_mean,
            'wavelength_dependent': True,
            'mu_per_wavelength': rainbow_mu_per_wl,
            'kappa_per_wavelength': rainbow_kappa_per_wl
        }

    # --- RESIDUAL LOBE (uniform) ---
    residual_lobe_descriptor = {
        'type': LOBE_RESIDUAL,
        'mu_center': 0.0,
        'kappa': 0.1,
        'amplitude': 0.0,
        'wavelength_dependent': False
    }

    # --- COMPUTE MIXTURE WEIGHTS ---
    forward_importance = forward_amp * forward_width
    rainbow_importance = rainbow_amp_mean * np.mean([l['width'] for l in rainbow_lobes]) if rainbow_lobes else 0.0
    residual_importance = 0.1 * forward_importance

    total_importance = forward_importance + rainbow_importance + residual_importance

    weight_forward = min(forward_importance / total_importance, 0.85)
    weight_rainbow = max(rainbow_importance / total_importance, 0.10) if rainbow_lobes else 0.0
    weight_residual = residual_importance / total_importance

    # Renormalize
    total_weight = weight_forward + weight_rainbow + weight_residual
    weights = {
        'forward': weight_forward / total_weight,
        'rainbow': weight_rainbow / total_weight,
        'residual': weight_residual / total_weight
    }

    # Build lobe list
    lobes = [forward_lobe_descriptor]
    if rainbow_lobe_descriptor:
        lobes.append(rainbow_lobe_descriptor)
    lobes.append(residual_lobe_descriptor)

    return lobes, weights


def write_mis_metadata(f, lobes, weights):
    """
    Write MIS metadata to binary file.

    Args:
        f: file handle (opened in 'ab' mode)
        lobes: list of lobe descriptors from analyze_lut_for_mis()
        weights: dict of mixture weights
    """
    f.write(MIS_MAGIC)
    f.write(struct.pack('B', MIS_VERSION))
    f.write(struct.pack('I', len(lobes)))
    f.write(struct.pack('fff',
                        weights.get('forward', 0.0),
                        weights.get('rainbow', 0.0),
                        weights.get('residual', 0.0)
                        ))

    for lobe in lobes:
        f.write(struct.pack('B', lobe['type']))
        is_wl_dep = lobe.get('wavelength_dependent', False)
        f.write(struct.pack('B', 1 if is_wl_dep else 0))
        f.write(struct.pack('H', 0))  # Reserved
        f.write(struct.pack('fff',
                            lobe['mu_center'],
                            lobe['kappa'],
                            lobe['amplitude']
                            ))

        if is_wl_dep:
            mu_wl = lobe.get('mu_per_wavelength', np.array([]))
            kappa_wl = lobe.get('kappa_per_wavelength', np.array([]))

            f.write(struct.pack('I', len(mu_wl)))
            f.write(mu_wl.astype(np.float32).tobytes())
            f.write(kappa_wl.astype(np.float32).tobytes())
        else:
            f.write(struct.pack('I', 0))
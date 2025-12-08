"""
Verification suite for Atmospheric Phase Function Data.
Performs integrity checks and specialized plotting without altering generation logic.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import your existing loader
try:
    from visualize import load_binary_file
except ImportError:
    print("CRITICAL: Could not import 'load_binary_file' from 'visualize.py'.")
    sys.exit(1)


def check_normalization(phase_table, num_angles, num_wavelengths):
    """
    Integrates the phase function over the sphere to check energy conservation.
    Assumption: Data is p(cos_theta).
    Integral should be roughly constant across wavelengths if normalized.
    """
    print("\n--- INTEGRITY CHECK: ENERGY CONSERVATION ---")

    # Reconstruct mu grid (assuming linear as per your generation script)
    # WARNING: If you change generation to log-spacing, update this!
    mu = np.linspace(1, -1, num_angles)
    d_mu = np.abs(mu[1] - mu[0])  # Assuming uniform steps

    # Integral = Sum( p(mu) * d_mu ) * 2pi
    # (The 2pi comes from the azimuthal integral 0->2pi)

    integrals = []

    # We define a tolerance range.
    # Note: miepython returns intensity, not always normalized PDF.
    # We just want to see if it's STABLE across wavelengths.

    print(f"{'Wavelength':<15} | {'Integral (Sum * 2pi)':<25} | {'Status'}")
    print("-" * 55)

    for i in range(num_wavelengths):
        # Slice the table [Angle, Wavelength] - noticed your loader might transpose
        # Let's handle the shape safely based on your loader logic
        # shape is (num_wavelengths, num_angles) based on your load_binary_file

        p_theta = phase_table[i, :]

        # Simple Riemann sum (Trapezoidal would be better, but this highlights the grid issue)
        total_scattering = np.sum(p_theta) * d_mu * 2 * np.pi
        integrals.append(total_scattering)

        status = "OK" if i == 0 else ("STABLE" if np.isclose(total_scattering, integrals[0], rtol=0.1) else "DRIFTING")
        print(f"Channel {i:<7} | {total_scattering:.4f} {'':<20} | {status}")

    avg_integral = np.mean(integrals)
    print(f"\nAverage Integral: {avg_integral:.4f}")
    if avg_integral < 1.0:
        print("\n[CRITICAL WARNING] Integral is surprisingly low. \n"
              "Did you resolve the forward peak? (See 'Linear Sampling' critique).")
    elif avg_integral > 1000.0:
        print("\n[NOTE] Data appears to be raw Intensity, not Normalized PDF.\n"
              "You must normalize this in your Mitsuba shader (divide by Scalar).")


def plot_polar_log(phase_table, num_angles, num_wavelengths, min_wl, max_wl):
    """
    Generates a Polar Plot with Logarithmic Radius.
    Standard polar plots fail for Mie because Forward Scatter is 1000x brighter.
    """
    print("\n--- GENERATING POLAR PLOT ---")

    mu = np.linspace(1, -1, num_angles)
    theta = np.arccos(mu)  # 0 to Pi

    # Setup colors for wavelengths (Blue to Red)
    wavelengths = np.linspace(min_wl, max_wl, num_wavelengths)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

    # We only plot a few representative wavelengths to avoid clutter
    indices_to_plot = [0, num_wavelengths // 2, num_wavelengths - 1]
    colors = ['blue', 'green', 'red']
    labels = [f"{wavelengths[i]:.0f}nm" for i in indices_to_plot]

    for idx, color, label in zip(indices_to_plot, colors, labels):
        intensity = phase_table[idx, :]

        # Log scaling for plotting
        # Add epsilon to avoid log(0)
        log_intensity = np.log10(intensity + 1e-10)

        # We enforce a floor for the plot so it doesn't look weird
        lower_bound = np.percentile(log_intensity, 5)
        log_intensity = np.maximum(log_intensity, lower_bound)

        # Plot top half (0 to 180)
        ax.plot(theta, log_intensity, color=color, linewidth=1.5, label=label)

        # Mirror for bottom half (to make it look like a droplet scatter)
        ax.plot(-theta, log_intensity, color=color, linewidth=1.5, alpha=0.5)

    ax.set_theta_zero_location("N")  # Forward scatter up
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_title("Mie Scattering Polar Plot (Log Scale)\nTop=Forward, Bottom=Backward", va='bottom')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("verification_polar.png")
    print("Saved: verification_polar.png")
    plt.close()


def plot_rainbow_separation(phase_table, num_angles, num_wavelengths, min_wl, max_wl):
    """
    Zooms in on the 130-150 degree range to verify dispersion (The Rainbow).
    """
    print("\n--- GENERATING RAINBOW ZOOM PLOT ---")

    mu = np.linspace(1, -1, num_angles)
    angles_deg = np.arccos(mu) * 180.0 / np.pi

    # Filter for rainbow region
    mask = (angles_deg >= 130) & (angles_deg <= 150)
    zoom_angles = angles_deg[mask]

    fig, ax = plt.subplots(figsize=(10, 6))

    wavelengths = np.linspace(min_wl, max_wl, num_wavelengths)

    # Plot Blue (Start), Green (Mid), Red (End)
    indices = [0, num_wavelengths // 2, num_wavelengths - 1]
    colors = ['blue', 'green', 'red']

    for idx, color in zip(indices, colors):
        wl = wavelengths[idx]
        intensity = phase_table[idx, mask]

        # Normalize to peak of this region for comparison
        intensity_norm = intensity / np.max(intensity)

        ax.plot(zoom_angles, intensity_norm, color=color, label=f"{wl:.0f}nm")

    ax.set_xlabel("Scattering Angle (Degrees)")
    ax.set_ylabel("Normalized Intensity (Local)")
    ax.set_title("Rainbow Region Dispersion Check (130°-150°)")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.savefig("verification_rainbow_zoom.png")
    print("Saved: verification_rainbow_zoom.png")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_integrity.py <binary_file>")
        sys.exit(1)

    fpath = sys.argv[1]

    # Load using your existing logic
    # Note: load_binary_file returns (phase_table, num_angles, num_wavelengths, min, max)
    # The loader in your code reshapes it to (num_wavelengths, num_angles)
    data, n_ang, n_wl, min_w, max_w = load_binary_file(fpath)

    check_normalization(data, n_ang, n_wl)
    plot_polar_log(data, n_ang, n_wl, min_w, max_w)
    plot_rainbow_separation(data, n_ang, n_wl, min_w, max_w)
"""
visualize.py
Visualization module for Atmospheric Phase Function binary files.
Independent of config.py to ensure it truthfully represents the binary data.
"""

import sys
import struct
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator


def load_binary_file(filename: str):
    """
    Load phase function table directly from binary structure.
    Skips MIS metadata if present.

    Returns:
        tuple: (phase_table, num_angles, num_wavelengths, min_wvl, max_wvl)
        phase_table is (num_wavelengths, num_angles) oriented.
    """
    with open(filename, 'rb') as f:
        # 1. Header Validation
        magic = f.read(8)
        if magic != b'ATMPHASE':
            raise ValueError(f"Invalid file signature: {magic}")

        # 2. Metadata Read
        version = struct.unpack('<I', f.read(4))[0]
        num_angles = struct.unpack('<I', f.read(4))[0]
        num_wavelengths = struct.unpack('<I', f.read(4))[0]
        min_wavelength = struct.unpack('<f', f.read(4))[0]
        max_wavelength = struct.unpack('<f', f.read(4))[0]

        # 3. Data Read
        total_floats = num_angles * num_wavelengths
        data_flat = np.frombuffer(f.read(total_floats * 4), dtype=np.float32)

        # Reshape to (Angles, Wavelengths) then Transpose to (Wavelengths, Angles)
        phase_table = data_flat.reshape(num_angles, num_wavelengths).T

        # 4. Check for MIS metadata (optional, just for logging)
        current_pos = f.tell()
        remaining = f.read(4)
        if remaining == b'MISD':
            print(f"  [INFO] MIS metadata detected at offset {current_pos}")
        # Note: We don't need to parse it for visualization

    print(f"Loaded {filename}: {num_wavelengths} wvl x {num_angles} angles")
    return phase_table, num_angles, num_wavelengths, min_wavelength, max_wavelength


def visualize_binary_file(input_filename: str, output_image: str = None):
    """
    Generates a high-fidelity heatmap of the phase function.
    """
    # 1. Load Data
    data, n_angles, n_wvls, min_w, max_w = load_binary_file(input_filename)

    # 2. Prepare Axes
    phase_plot = data.T

    # Angles: 0 deg (Forward) to 180 deg (Back)
    mu = np.linspace(1, -1, n_angles)
    angles_deg = np.arccos(mu) * 180.0 / np.pi

    # Wavelengths
    wavelengths = np.linspace(min_w, max_w, n_wvls)

    # 3. Cut Forward Scattering
    cutoff_angle = 10.0
    valid_mask = angles_deg >= cutoff_angle

    phase_cut = phase_plot[valid_mask, :]
    angles_cut = angles_deg[valid_mask]

    # 4. Normalize (Per Wavelength)
    col_means = np.mean(phase_cut, axis=0)
    phase_norm = phase_cut / (col_means + 1e-12)

    # 5. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    X, Y = np.meshgrid(wavelengths, angles_cut)

    vmin = np.percentile(phase_norm, 1)
    vmax = np.percentile(phase_norm, 99)

    im = ax.pcolormesh(
        X, Y, phase_norm,
        shading='auto',
        cmap='turbo',
        norm=LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
    )

    # 6. Ticks & Labels
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    ax.grid(which='major', color='white', alpha=0.3, linestyle='-', linewidth=0.7)
    ax.grid(which='minor', color='white', alpha=0.1, linestyle=':', linewidth=0.5)

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Scattering Angle (degrees)', fontsize=12)
    ax.set_title(
        f'Mie Scattering Phase Function\n(Log Scale, Cutoff < {cutoff_angle}°)',
        fontsize=14, fontweight='bold'
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Intensity (Log)', fontsize=11)

    # 7. Output
    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_image}")
    else:
        plt.show()

    plt.close()


def plot_polar_log(
    phase_table: np.ndarray,
    num_angles: int,
    num_wavelengths: int,
    min_wl: float,
    max_wl: float,
    output_image: str | None = None,
):
    """
    Generates a Polar Plot with Logarithmic Radius.

    phase_table: (num_wavelengths, num_angles)
    """
    print("\n--- GENERATING POLAR PLOT ---")

    mu = np.linspace(1, -1, num_angles)
    theta = np.arccos(mu)  # 0..pi

    wavelengths = np.linspace(min_wl, max_wl, num_wavelengths)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

    # Plot representative wavelengths to avoid clutter
    indices_to_plot = [0, num_wavelengths // 2, num_wavelengths - 1]
    colors = ['blue', 'green', 'red']
    labels = [f"{wavelengths[i]:.0f}nm" for i in indices_to_plot]

    for idx, color, label in zip(indices_to_plot, colors, labels):
        intensity = phase_table[idx, :].astype(np.float64)

        log_intensity = np.log10(intensity + 1e-10)

        # floor to tame extreme log spikes (keeps shape but avoids ugly far-out tails)
        lower_bound = np.percentile(log_intensity, 5)
        log_intensity = np.maximum(log_intensity, lower_bound)

        # Top half (0..pi)
        ax.plot(theta, log_intensity, color=color, linewidth=1.5, label=label)
        # Mirror bottom half for a "droplet" look
        ax.plot(-theta, log_intensity, color=color, linewidth=1.5, alpha=0.5)

    ax.set_theta_zero_location("N")  # forward scatter up
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_title(
        "Scattering Polar Plot (Log Radius)\nTop=Forward, Bottom=Backward",
        va='bottom'
    )
    ax.legend(loc='lower right')

    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"Polar plot saved to: {output_image}")
    else:
        plt.show()

    plt.close()


def visualize_polar_plot(input_filename: str, output_image: str | None = None):
    """
    Loads a binary LUT and produces a polar plot.
    """
    phase_table, n_angles, n_wvls, min_w, max_w = load_binary_file(input_filename)
    plot_polar_log(
        phase_table=phase_table,
        num_angles=n_angles,
        num_wavelengths=n_wvls,
        min_wl=min_w,
        max_wl=max_w,
        output_image=output_image,
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <input.bin> [output.png]")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else None

    visualize_binary_file(in_file, out_file)
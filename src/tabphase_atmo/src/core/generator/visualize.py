"""
visualize.py
Visualization for Angle-Space Mie LUTS.
Includes 'Sun-Blocking' logic to fix the "Squished" polar plots.
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

def load_binary_file(filename: str):
    with open(filename, 'rb') as f:
        magic = f.read(8)
        if magic != b'ATMPHASE':
            raise ValueError(f"Invalid file signature: {magic}")
        version = struct.unpack('<I', f.read(4))[0]
        num_angles = struct.unpack('<I', f.read(4))[0]
        num_wavelengths = struct.unpack('<I', f.read(4))[0]
        min_wavelength = struct.unpack('<f', f.read(4))[0]
        max_wavelength = struct.unpack('<f', f.read(4))[0]
        total_floats = num_angles * num_wavelengths
        data_flat = np.frombuffer(f.read(total_floats * 4), dtype=np.float32)
        phase_table = data_flat.reshape(num_angles, num_wavelengths).T

    print(f"Loaded {filename}: {num_wavelengths} wvl x {num_angles} angles")
    return phase_table, num_angles, num_wavelengths, min_wavelength, max_wavelength

def visualize_binary_file(input_filename: str, output_image: str = None):
    data, n_angles, n_wvls, min_w, max_w = load_binary_file(input_filename)

    # Grid is Linear Angle (Theta)
    angles_deg = np.linspace(0, 180, n_angles)
    wavelengths = np.linspace(min_w, max_w, n_wvls)

    # Cut Forward Scattering for the Heatmap
    cutoff_idx = int(n_angles * (5.0 / 180.0)) # Skip first 5 degrees
    phase_cut = data.T[cutoff_idx:, :]
    angles_cut = angles_deg[cutoff_idx:]

    col_means = np.mean(phase_cut, axis=0)
    phase_norm = phase_cut / (col_means + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 10))
    X, Y = np.meshgrid(wavelengths, angles_cut)

    vmin = np.percentile(phase_norm, 1)
    vmax = np.percentile(phase_norm, 99)

    im = ax.pcolormesh(X, Y, phase_norm, shading='auto', cmap='turbo', norm=LogNorm(vmin=max(vmin, 1e-5), vmax=vmax))

    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Scattering Angle (degrees)')
    ax.set_title(f'Mie Phase Function (Heatmap, > 5°)')
    plt.colorbar(im, ax=ax, label='Normalized Intensity (Log)')

    if output_image: plt.savefig(output_image, dpi=150, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_polar_log(phase_table, num_angles, num_wavelengths, min_wl, max_wl, output_image=None):
    print("\n--- GENERATING POLAR PLOT ---")

    # Grid is Linear Angle (Theta)
    theta = np.linspace(0, np.pi, num_angles)
    wavelengths = np.linspace(min_wl, max_wl, num_wavelengths)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

    # --- THE FIX: Full Scale, No Clipping ---
    # Convert entire table to log to find the absolute true bounds
    log_all_data = np.log10(phase_table + 1e-12)

    # Floor: 1st percentile avoids the -12 math black holes
    vis_min = np.percentile(log_all_data, 1) - 0.2

    # Ceiling: The actual peak of the forward scattering (The Sun)
    vis_max = np.max(log_all_data)


    print(f"  [Vis] Scaling Range: {vis_min:.2f} to {vis_max:.2f} (Sun Included)")

    indices_to_plot = [0, num_wavelengths // 2, num_wavelengths - 1]
    colors = ['blue', 'green', 'red']
    labels = [f"{wavelengths[i]:.0f} nm" for i in indices_to_plot]

    for idx, color, label in zip(indices_to_plot, colors, labels):
        intensity = phase_table[idx, :].astype(np.float64)
        log_intensity = np.log10(intensity + 1e-12)

        # Let the data fly, absolutely NO clipping
        ax.plot(theta, log_intensity, color=color, linewidth=1.5, label=label)
        ax.plot(-theta, log_intensity, color=color, linewidth=1.5, alpha=0.5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)


    ax.set_rorigin(vis_min)

    ax.set_ylim(vis_min, vis_max)

    ax.set_title("Scattering Polar Plot (Log Radius)", va='bottom')
    ax.legend(loc='lower right')
    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def visualize_polar_plot(input_filename: str, output_image: str | None = None):
    phase_table, n_angles, n_wvls, min_w, max_w = load_binary_file(input_filename)
    plot_polar_log(phase_table, n_angles, n_wvls, min_w, max_w, output_image)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <input.bin> [output.png]")
        sys.exit(1)
    visualize_polar_plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
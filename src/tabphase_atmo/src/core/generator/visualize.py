"""
Visualization for Angle-Space Mie LUTS (V1 & V2).
Includes Anisotropic 'Unrolled Sky' and 'Sliced Polar' plots.
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

        if version == 1:
            num_angles = struct.unpack('<I', f.read(4))[0]
            num_phi_bins = 1
            num_wvls = struct.unpack('<I', f.read(4))[0]
        elif version == 2:
            num_angles = struct.unpack('<I', f.read(4))[0]
            num_phi_bins = struct.unpack('<I', f.read(4))[0]
            num_wvls = struct.unpack('<I', f.read(4))[0]
        else:
            raise ValueError(f"Unknown version: {version}")

        min_w = struct.unpack('<f', f.read(4))[0]
        max_w = struct.unpack('<f', f.read(4))[0]

        total_floats = num_angles * num_phi_bins * num_wvls
        data_flat = np.frombuffer(f.read(total_floats * 4), dtype=np.float32)

        # Invert the interleaving: Disk is (Theta, Phi, Wvl) -> Array is (Wvl, Phi, Theta)
        phase_interleaved = data_flat.reshape((num_angles, num_phi_bins, num_wvls))
        phase_table = phase_interleaved.transpose(2, 1, 0)

    print(f"[Vis] Loaded {filename} (v{version}): {num_wvls}wvl x {num_phi_bins}phi x {num_angles}theta")
    return phase_table, num_angles, num_phi_bins, num_wvls, min_w, max_w

def visualize_anisotropic(input_filename: str, output_image: str | None = None):
    data, n_angles, n_phis, n_wvls, min_w, max_w = load_binary_file(input_filename)

    theta_deg = np.linspace(0, 180, n_angles)
    theta_rad = np.linspace(0, np.pi, n_angles)
    phi_deg = np.linspace(0, 360, n_phis)
    wavelengths = np.linspace(min_w, max_w, n_wvls)

    # Grab the middle wavelength (usually ~550nm Green) for structural visualization
    mid_wvl_idx = n_wvls // 2
    wvl_slice = data[mid_wvl_idx, :, :]

    # --- PLOT SETUP ---
    fig = plt.figure(figsize=(18, 8))

    # Check if we actually have anisotropic data
    if n_phis > 1:
        # ==========================================
        # PANEL 1: UNROLLED SKY HEATMAP (Phi vs Theta)
        # ==========================================
        ax1 = fig.add_subplot(121)

        # Cut Forward Scattering for the Heatmap (skip first 5 degrees)
        cutoff_idx = int(n_angles * (5.0 / 180.0))
        slice_cut = wvl_slice[:, cutoff_idx:]
        theta_cut = theta_deg[cutoff_idx:]

        X, Y = np.meshgrid(theta_cut, phi_deg)

        vmin = np.percentile(slice_cut, 1)
        vmax = np.percentile(slice_cut, 99)

        im = ax1.pcolormesh(X, Y, slice_cut, shading='auto', cmap='turbo', norm=LogNorm(vmin=max(vmin, 1e-5), vmax=vmax))

        ax1.xaxis.set_major_locator(MultipleLocator(20))
        ax1.yaxis.set_major_locator(MultipleLocator(45))
        ax1.set_xlabel('Scattering Angle / Theta (degrees)')
        ax1.set_ylabel('Azimuth / Phi (degrees)')
        ax1.set_title(f'Anisotropic Unrolled Sky @ {wavelengths[mid_wvl_idx]:.0f}nm (>5°)')
        plt.colorbar(im, ax=ax1, label='Normalized Intensity (Log)')

        # ==========================================
        # PANEL 2: SLICED POLAR PLOT (Vertical vs Horizontal)
        # ==========================================
        ax2 = fig.add_subplot(122, projection='polar')

        slice_horizontal = wvl_slice[0, :] # Phi = 0
        slice_vertical = wvl_slice[n_phis // 4, :] # Phi = 90

        log_h = np.log10(slice_horizontal + 1e-12)
        log_v = np.log10(slice_vertical + 1e-12)

        vis_min = np.percentile(np.log10(wvl_slice + 1e-12), 1) - 0.2
        vis_max = np.max(np.log10(wvl_slice + 1e-12))

        ax2.plot(theta_rad, log_h, color='red', linewidth=1.5, label='Horizontal Slice (Φ=0°)')
        ax2.plot(-theta_rad, log_h, color='red', linewidth=1.5, alpha=0.5)

        ax2.plot(theta_rad, log_v, color='blue', linewidth=1.5, linestyle='--', label='Vertical Slice (Φ=90°)')
        ax2.plot(-theta_rad, log_v, color='blue', linewidth=1.5, linestyle='--', alpha=0.5)

        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_rlabel_position(45)
        ax2.set_rorigin(vis_min)
        ax2.set_ylim(vis_min, vis_max)

        ax2.set_title(f"Polar Anisotropy Check @ {wavelengths[mid_wvl_idx]:.0f}nm", va='bottom')
        ax2.legend(loc='lower right')

    else:
        # ==========================================
        # FALLBACK: STANDARD 1D SPHERICAL PLOTS
        # ==========================================
        print("[Vis] Detected isotropic data (num_phi_bins=1). Falling back to standard visualization.")

        ax1 = fig.add_subplot(121)
        cutoff_idx = int(n_angles * (5.0 / 180.0))
        phase_cut = data[:, 0, cutoff_idx:].T
        theta_cut = theta_deg[cutoff_idx:]

        col_means = np.mean(phase_cut, axis=0)
        phase_norm = phase_cut / (col_means + 1e-12)

        X, Y = np.meshgrid(wavelengths, theta_cut)
        vmin, vmax = np.percentile(phase_norm, 1), np.percentile(phase_norm, 99)

        im = ax1.pcolormesh(X, Y, phase_norm, shading='auto', cmap='turbo', norm=LogNorm(vmin=max(vmin, 1e-5), vmax=vmax))
        ax1.yaxis.set_major_locator(MultipleLocator(10))
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Scattering Angle (degrees)')
        ax1.set_title('Spherical Mie Phase Function (>5°)')
        plt.colorbar(im, ax=ax1, label='Normalized Intensity (Log)')

        ax2 = fig.add_subplot(122, projection='polar')
        indices = [0, n_wvls // 2, n_wvls - 1]
        colors = ['blue', 'green', 'red']

        log_all_data = np.log10(data[:, 0, :] + 1e-12)
        vis_min, vis_max = np.percentile(log_all_data, 1) - 0.2, np.max(log_all_data)

        for idx, color in zip(indices, colors):
            log_intensity = np.log10(data[idx, 0, :] + 1e-12)
            ax2.plot(theta_rad, log_intensity, color=color, linewidth=1.5, label=f"{wavelengths[idx]:.0f} nm")
            ax2.plot(-theta_rad, log_intensity, color=color, linewidth=1.5, alpha=0.5)

        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_rorigin(vis_min)
        ax2.set_ylim(vis_min, vis_max)
        ax2.set_title("Scattering Polar Plot (Log Radius)")
        ax2.legend(loc='lower right')

    plt.tight_layout()
    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <input.bin> [output.png]")
        sys.exit(1)
    visualize_anisotropic(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
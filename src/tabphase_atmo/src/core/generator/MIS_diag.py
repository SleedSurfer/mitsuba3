import struct
import numpy as np
import matplotlib.pyplot as plt


def load_and_visualize(filename):
    with open(filename, "rb") as f:
        # --- PARSE RAW LUT ---
        magic = f.read(8)
        if magic != b"ATMPHASE":
            raise ValueError("Not a valid ATMPHASE file bro.")

        version, res, channels = struct.unpack("<III", f.read(12))
        min_w, max_w = struct.unpack("<ff", f.read(8))

        # Read interleaved data and reshape to [Wavelengths, Angles]
        total_floats = res * channels
        raw_data = f.read(total_floats * 4)
        data_flat = np.frombuffer(raw_data, dtype=np.float32)
        phase_table = data_flat.reshape((res, channels)).T

        # --- PARSE MIS METADATA ---
        mis_magic = f.read(4)
        if mis_magic != b"MISD":
            print("No MIS metadata found, just rawdogging the LUT today.")
            return

        mis_version, num_lobes = struct.unpack("<BI", f.read(5))
        w_fwd, w_rbw, w_res = struct.unpack("<fff", f.read(12))

        print(f"--- MIS WEIGHTS ---\nForward: {w_fwd:.3f} | Rainbow: {w_rbw:.3f} | Residual: {w_res:.3f}\n")

        lobes = []
        for _ in range(num_lobes):
            l_type, is_wl_dep, _ = struct.unpack("<BBH", f.read(4))
            mu_c, kappa, amp = struct.unpack("<fff", f.read(12))

            lobe = {'type': l_type, 'mu': mu_c, 'kappa': kappa, 'amp': amp, 'is_wl_dep': is_wl_dep}

            if is_wl_dep:
                arr_len = struct.unpack("<I", f.read(4))[0]
                lobe['mu_wl'] = np.frombuffer(f.read(arr_len * 4), dtype=np.float32)
                lobe['kappa_wl'] = np.frombuffer(f.read(arr_len * 4), dtype=np.float32)

            lobes.append(lobe)

    # --- VISUALIZATION ---
    mu_vals = np.cos(np.linspace(0, np.pi, res))

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Heatmap
    ax1 = axes[0]
    im = ax1.imshow(np.log10(phase_table + 1e-10), aspect='auto',
                    extent=[mu_vals[-1], mu_vals[0], max_w, min_w], cmap='inferno')
    ax1.set_title("Raw Phase Function (Log10)")
    ax1.set_xlabel(r"Angle ($\mu$)")
    ax1.set_ylabel("Wavelength (nm)")
    ax1.invert_xaxis()  # 1 to -1 (Forward to Backward)
    fig.colorbar(im, ax=ax1, label="Log10 Intensity")

    # 2. MIS Overlay for middle wavelength
    ax2 = axes[1]
    mid_idx = channels // 2
    mid_wl = min_w + (max_w - min_w) * (mid_idx / channels)

    phase_1d = phase_table[mid_idx, :]
    ax2.plot(mu_vals, np.log10(phase_1d + 1e-10), color='white', label=f'Raw {mid_wl:.0f}nm')

    # Reconstruct von Mises-Fisher / Gaussian lobes
    for l in lobes:
        if l['is_wl_dep']:
            if len(l['mu_wl']) > mid_idx:
                mu_c = l['mu_wl'][mid_idx]
                kappa = l['kappa_wl'][mid_idx]
            else:
                print(f"Skipping empty wavelength array for lobe type {l['type']}")
                continue
        else:
            mu_c = l['mu']
            kappa = l['kappa']

        # Simple Gaussian approximation for the plot
        # kappa acts as 1 / (2 * sigma^2)
        y = amp * np.exp(-kappa * (mu_vals - mu_c) ** 2)

        name = ["Forward", "Rainbow", "Residual", "Glory"][l['type']]
        ax2.plot(mu_vals, np.log10(y + 1e-10), linestyle='--', label=f'{name} Lobe')

    ax2.set_title("MIS Lobe Overlay (Middle Wavelength)")
    ax2.set_xlabel(r"Angle ($\mu$)")
    ax2.set_ylabel("Log10 Intensity")
    ax2.invert_xaxis()
    ax2.legend()
    ax2.set_ylim(-5, np.max(np.log10(phase_1d)) + 1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_visualize("/home/speedlord/mitsuba3/src/tabphase_atmo/src/cache/hybrid/1000um_sphere_20pctVar_8b_brute_auto.bin")
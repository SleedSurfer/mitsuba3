import sys
import struct
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # The WSL savior
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def load_binary_file(filename: str):
    with open(filename, 'rb') as f:
        magic = f.read(8)
        if magic != b'ATMPHASE':
            raise ValueError(f"Invalid file signature: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        num_angles = struct.unpack('<I', f.read(4))[0]
        num_wavelengths = struct.unpack('<I', f.read(4))[0]
        min_w = struct.unpack('<f', f.read(4))[0]
        max_w = struct.unpack('<f', f.read(4))[0]

        print(f"Loaded {filename}: {num_angles} Angles, {num_wavelengths} Wavelengths")

        total_floats = num_angles * num_wavelengths
        data_flat = np.frombuffer(f.read(total_floats * 4), dtype=np.float32)

        if data_flat.size != total_floats:
            print(f"WARNING: Expected {total_floats} floats, got {data_flat.size}")

        phase_table = data_flat.reshape(num_wavelengths, num_angles)

    return phase_table, num_angles, num_wavelengths, min_w, max_w


def run_gui(filename):
    data, n_angles, n_wvls, min_w, max_w = load_binary_file(filename)

    theta = np.linspace(0, np.pi, n_angles)
    wavelengths = np.linspace(min_w, max_w, max(1, n_wvls))

    # Sanitize NaNs/Infs
    clean_data = np.nan_to_num(data, nan=1e-12, posinf=1e-12, neginf=1e-12)
    log_data = np.log10(np.clip(clean_data, 1e-12, None))

    abs_min = np.min(log_data)
    abs_max = np.max(log_data)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
    fig.patch.set_facecolor('#0b0e14')
    plt.subplots_adjust(bottom=0.25)

    init_wvl = 0
    init_zoom = abs_max

    line_main, = ax.plot(theta, log_data[init_wvl, :] - abs_min, color='#00ffcc', lw=2)
    line_sym, = ax.plot(-theta, log_data[init_wvl, :] - abs_min, color='#00ffcc', lw=2, alpha=0.3)

    def format_polar_axes(current_zoom):
        current_r_limit = current_zoom - abs_min

        # THE FIX: Lowered the hardcoded safety floor from 0.1 to 0.001
        safe_r_limit = max(0.001, current_r_limit)
        ax.set_rlim(0, safe_r_limit)

        ticks = np.linspace(0, safe_r_limit, 6)
        ax.set_rticks(ticks)

        tick_labels = ["$10^{" + f"{(t + abs_min):.1f}" + "}$" for t in ticks]
        ax.set_yticklabels(tick_labels, color='gray')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.grid(True, linestyle=':', color='white', alpha=0.2)
        ax.tick_params(colors='white')
        ax.set_facecolor('#0b0e14')

    format_polar_axes(init_zoom)
    ax.set_title(f"Mie Phase Function\nZoomed to 10^{init_zoom:.1f}", color='white', pad=30, fontsize=15)

    ax_color = '#1f2430'
    ax_wvl = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=ax_color)
    ax_zoom = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=ax_color)

    # THE FIX: Dropped the minimum zoom limit so you can scrub almost to the floor
    slider_wvl = Slider(ax_wvl, 'λ Index', 0, max(1, n_wvls - 1), valinit=init_wvl, valstep=1, color='#00ffcc')
    slider_zoom = Slider(ax_zoom, 'Zoom Max', abs_min + 0.05, abs_max, valinit=init_zoom, color='#ff00cc')

    slider_wvl.label.set_color('white')
    slider_wvl.valtext.set_color('white')
    slider_zoom.label.set_color('white')
    slider_zoom.valtext.set_color('white')

    def update(val):
        w_idx = int(slider_wvl.val)
        z_max = slider_zoom.val

        new_y = log_data[w_idx, :] - abs_min
        line_main.set_ydata(new_y)
        line_sym.set_ydata(new_y)

        format_polar_axes(z_max)
        wvl_val = wavelengths[w_idx] if n_wvls > 1 else min_w
        ax.set_title(f"Wavelength: {wvl_val:.1f} nm\nZoomed to 10^{z_max:.1f}", color='white', pad=30, fontsize=15)

        fig.canvas.draw_idle()

    slider_wvl.on_changed(update)
    slider_zoom.on_changed(update)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <path_to_bin_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    run_gui(file_path)
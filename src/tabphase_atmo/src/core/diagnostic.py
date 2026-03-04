"""
Iteration 0: Diagnostic Harness
--------------------------------
Loads an existing ATMPHASE binary LUT and produces:
  1. The raw phase function signal (linear and log scale)
  2. The FFT of that row (frequency content)
  3. A 2D heatmap of the full table (wavelength x angle)
  4. Rainbow region zoom
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

LUT_PATH = "/home/speedlord/mitsuba3/src/tabphase_atmo/src/cache/hybrid/1000um_sphere_20pctVar_8b_brute_auto.bin"

# Which rows to overlay on signal/fft/zoom plots
# -1 = auto (picks short/mid/long wavelength rows evenly)
OVERLAY_ROWS = [-1, -1, -1]

# ─────────────────────────────────────────────────────────────────────────────


def load_lut(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        phase_table : (num_wavelengths, num_angles) float32
        wavelengths : (num_wavelengths,)            float64  [nm]
        theta_deg   : (num_angles,)                 float64  [degrees]
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"ATMPHASE", f"Bad magic bytes: {magic!r}"

        version    = struct.unpack("<I", f.read(4))[0]
        num_angles = struct.unpack("<I", f.read(4))[0]
        num_wl     = struct.unpack("<I", f.read(4))[0]
        min_wl     = struct.unpack("<f", f.read(4))[0]
        max_wl     = struct.unpack("<f", f.read(4))[0]

        raw = np.frombuffer(f.read(num_angles * num_wl * 4), dtype=np.float32)

    # saved as (angles, wavelengths) interleaved — reverse the save transpose
    phase_table = raw.reshape((num_angles, num_wl)).T   # → (wl, angles)

    wavelengths = np.linspace(min_wl, max_wl, num_wl)
    theta_deg   = np.linspace(0.0, 180.0, num_angles)

    print(f"[Diag] Loaded : {Path(path).name}")
    print(f"[Diag]   Wavelengths : {num_wl}  ({min_wl:.0f}–{max_wl:.0f} nm)")
    print(f"[Diag]   Angles      : {num_angles}  (0–180°)")
    print(f"[Diag]   Version     : {version}")
    print(f"[Diag]   Value range : [{phase_table.min():.6f},  {phase_table.max():.6f}]")

    return phase_table, wavelengths, theta_deg


def resolve_rows(overlay_rows, num_wl):
    resolved = []
    defaults = [int(num_wl * 0.15), num_wl // 2, int(num_wl * 0.85)]
    for i, r in enumerate(overlay_rows):
        resolved.append(defaults[i] if r < 0 else int(np.clip(r, 0, num_wl - 1)))
    return resolved

def compare_luts(path_mono, path_poly, label_poly="polydisperse"):
    """
    Load two LUTs and produce a direct comparison:
      1. Signal overlay per wavelength row (full + rainbow zoom)
      2. FFT power spectrum overlay — shape comparison, not just peak
      3. Difference signal: poly - mono
      4. Ratio of FFT power spectra: shows exactly which frequencies were suppressed
    """
    import struct

    def _load(path):
        with open(path, "rb") as f:
            assert f.read(8) == b"ATMPHASE"
            _ = struct.unpack("<I", f.read(4))[0]
            num_angles = struct.unpack("<I", f.read(4))[0]
            num_wl     = struct.unpack("<I", f.read(4))[0]
            min_wl     = struct.unpack("<f", f.read(4))[0]
            max_wl     = struct.unpack("<f", f.read(4))[0]
            raw = np.frombuffer(f.read(num_angles * num_wl * 4), dtype=np.float32)
        pt  = raw.reshape((num_angles, num_wl)).T
        wls = np.linspace(min_wl, max_wl, num_wl)
        th  = np.linspace(0.0, 180.0, num_angles)
        return pt, wls, th

    pt_mono, wls, theta = _load(path_mono)
    pt_poly, _,   _     = _load(path_poly)

    num_wl  = len(wls)
    rows    = [int(num_wl * 0.15), num_wl // 2, int(num_wl * 0.85)]
    colors  = ["royalblue", "limegreen", "tomato"]

    deg_step   = theta[1] - theta[0]
    mask_full  = np.ones(len(theta), dtype=bool)
    mask_rain  = (theta >= 120.0) & (theta <= 150.0)
    theta_rain = theta[mask_rain]

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.suptitle(
        f"Mono vs Poly Comparison\n"
        f"  MONO: {Path(path_mono).name}\n"
        f"  POLY: {Path(path_poly).name}",
        fontsize=10, y=0.99
    )

    ax_full, ax_zoom, ax_diff   = axes[0]
    ax_fft_full, ax_fft_rain, ax_fft_ratio = axes[1]
    ax_fft_ratio2, ax_cum, ax_logdiff       = axes[2]

    for i, (r, col) in enumerate(zip(rows, colors)):
        wl   = wls[r]
        mono = pt_mono[r]
        poly = pt_poly[r]

        # ── Row 0: Signal comparison ──────────────────────────
        ax_full.plot(theta, mono, lw=0.6, color=col,
                     linestyle="--", label=f"mono {wl:.0f}nm")
        ax_full.plot(theta, poly, lw=0.6, color=col,
                     linestyle="-",  label=f"poly {wl:.0f}nm", alpha=0.7)

        ax_zoom.plot(theta_rain, mono[mask_rain], lw=0.8, color=col,
                     linestyle="--", label=f"mono {wl:.0f}nm")
        ax_zoom.plot(theta_rain, poly[mask_rain], lw=0.8, color=col,
                     linestyle="-",  label=f"poly {wl:.0f}nm", alpha=0.7)

        diff = poly - mono
        ax_diff.plot(theta, diff, lw=0.6, color=col, label=f"{wl:.0f}nm")

        # ── Row 1: FFT comparison ─────────────────────────────
        def _fft_detrend(sig, mask, deg=3):
            th_w = theta[mask]
            tr   = np.polyval(np.polyfit(th_w, sig[mask], deg), th_w)
            rip  = sig[mask] - tr
            N    = len(rip)
            fft  = np.fft.rfft(rip * np.hanning(N))
            freq = np.fft.rfftfreq(N, d=deg_step)
            return freq, np.abs(fft)

        # Full signal FFT
        freq_f, pwr_mono_f = _fft_detrend(mono, mask_full, deg=6)
        _,      pwr_poly_f = _fft_detrend(poly, mask_full, deg=6)
        ax_fft_full.plot(freq_f, pwr_mono_f, lw=0.6, color=col,
                         linestyle="--", alpha=0.8)
        ax_fft_full.plot(freq_f, pwr_poly_f, lw=0.6, color=col,
                         linestyle="-",  alpha=0.8, label=f"{wl:.0f}nm")

        # Rainbow window FFT
        freq_r, pwr_mono_r = _fft_detrend(mono, mask_rain, deg=3)
        _,      pwr_poly_r = _fft_detrend(poly, mask_rain, deg=3)
        ax_fft_rain.plot(freq_r, pwr_mono_r, lw=0.7, color=col,
                         linestyle="--", alpha=0.8)
        ax_fft_rain.plot(freq_r, pwr_poly_r, lw=0.7, color=col,
                         linestyle="-",  alpha=0.8, label=f"{wl:.0f}nm")

        # FFT power ratio: poly/mono — this is the transfer function of polydispersity
        eps = 1e-12
        ratio = pwr_poly_r / (pwr_mono_r + eps)
        ax_fft_ratio.plot(freq_r, ratio, lw=0.8, color=col, label=f"{wl:.0f}nm")

        # Same ratio on log scale
        ax_fft_ratio2.plot(freq_r, 20 * np.log10(ratio + eps),
                           lw=0.8, color=col, label=f"{wl:.0f}nm")

        # Cumulative power: how much energy survives as fn of frequency cutoff
        cum_mono = np.cumsum(pwr_mono_r ** 2)
        cum_poly = np.cumsum(pwr_poly_r ** 2)
        cum_mono /= cum_mono[-1] + eps
        cum_poly /= cum_poly[-1] + eps
        ax_cum.plot(freq_r, cum_mono, lw=0.6, color=col, linestyle="--")
        ax_cum.plot(freq_r, cum_poly, lw=0.6, color=col, linestyle="-",
                    label=f"{wl:.0f}nm")

        # Log diff in rainbow zoom
        safe_m = np.where(mono[mask_rain] > 0, mono[mask_rain], np.nan)
        safe_p = np.where(poly[mask_rain] > 0, poly[mask_rain], np.nan)
        ax_logdiff.plot(theta_rain,
                        np.log10(safe_p) - np.log10(safe_m),
                        lw=0.8, color=col, label=f"{wl:.0f}nm")

    # ── Formatting ────────────────────────────────────────────
    for ax, title, xlabel, ylabel in [
        (ax_full,       "Full Signal  (-- mono, — poly)",
                        "Angle (°)", "Phase fn"),
        (ax_zoom,       "Rainbow Zoom 120°–150°  (-- mono, — poly)",
                        "Angle (°)", "Phase fn"),
        (ax_diff,       "Difference  poly − mono",
                        "Angle (°)", "Δ Phase fn"),
        (ax_fft_full,   "Full FFT  (-- mono, — poly)",
                        "Freq (c/°)", "Amplitude"),
        (ax_fft_rain,   "Rainbow FFT  (-- mono, — poly)",
                        "Freq (c/°)", "Amplitude"),
        (ax_fft_ratio,  "FFT Power Ratio  poly/mono  ← THE KEY PLOT",
                        "Freq (c/°)", "Ratio"),
        (ax_fft_ratio2, "FFT Power Ratio  (dB)",
                        "Freq (c/°)", "dB"),
        (ax_cum,        "Cumulative Power  (-- mono, — poly)",
                        "Freq (c/°)", "Normalised cumulative power"),
        (ax_logdiff,    "Log Difference  log(poly) − log(mono)  in rainbow",
                        "Angle (°)", "Δ log₁₀"),
    ]:
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    # Reference lines
    ax_fft_ratio.axhline(1.0, color="white", lw=0.5, linestyle=":")
    ax_fft_ratio2.axhline(0.0, color="white", lw=0.5, linestyle=":")

    plt.tight_layout()
    out = "comparison_mono_vs_poly.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[Diag] Saved: {out}")


# ── run directly ──────────────────────────────────────────────────────────────
MONO_PATH = "/home/speedlord/mitsuba3/src/tabphase_atmo/src/cache/hybrid/1000um_sphere_0pctVar_8b_brute_auto.bin"
POLY_PATH = "/home/speedlord/mitsuba3/src/tabphase_atmo/src/cache/hybrid/1000um_sphere_20pctVar_8b_brute_auto.bin"

def plot_fft_rainbow_only(phase_table, wavelengths, theta_deg, rows, ax):
    """
    FFT restricted to the rainbow window (120-150 degrees) only.
    This isolates the supernumerary fringe frequency cleanly,
    away from the forward peak and smooth background.
    """
    from scipy.signal import windows

    deg_step = theta_deg[1] - theta_deg[0]
    mask     = (theta_deg >= 120.0) & (theta_deg <= 150.0)
    theta_w  = theta_deg[mask]
    dominant_freqs = {}

    for r in rows:
        sig    = phase_table[r][mask]
        wl     = wavelengths[r]

        # Detrend within window only
        trend  = np.polyval(np.polyfit(theta_w, sig, deg=3), theta_w)
        ripple = sig - trend

        N    = len(ripple)
        win  = windows.hann(N)
        fft  = np.fft.rfft(ripple * win)
        freq = np.fft.rfftfreq(N, d=deg_step)
        pwr  = np.abs(fft)

        ax.plot(freq, pwr, lw=0.8, label=f"{wl:.1f} nm")

        peak_idx = np.argmax(pwr[1:]) + 1
        dominant_freqs[wl] = freq[peak_idx]
        ax.axvline(freq[peak_idx], lw=0.5, alpha=0.5, linestyle="--")
        ax.text(freq[peak_idx], pwr[peak_idx],
                f" {freq[peak_idx]:.4f}", fontsize=6, va="bottom")

    ax.set_title("FFT — Rainbow Window Only (120°–150°)")
    ax.set_xlabel("Spatial Frequency (cycles / degree)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    print(f"[Diag] Rainbow-window dominant fringe frequencies:")
    for wl, freq in dominant_freqs.items():
        print(f"[Diag]   {wl:.1f} nm  →  ω_ref = {freq:.5f} cycles/degree")

    return dominant_freqs


def plot_signals(phase_table, wavelengths, theta_deg, rows, ax_lin, ax_log):
    for r in rows:
        sig = phase_table[r]
        lbl = f"{wavelengths[r]:.1f} nm"
        ax_lin.plot(theta_deg, sig, lw=0.8, label=lbl)
        safe = np.where(sig > 0, sig, np.nan)
        ax_log.plot(theta_deg, np.log10(safe), lw=0.8, label=lbl)

    for ax, title, ylabel in [
        (ax_lin, "Signal — Linear",    "Phase Function (normalized)"),
        (ax_log, "Signal — Log Scale", "log₁₀(Phase Function)"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Scattering Angle (degrees)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)


def plot_fft(phase_table, wavelengths, theta_deg, rows, ax, title="FFT of Detrended Signal"):
    deg_step = theta_deg[1] - theta_deg[0]
    dominant_freqs = {}

    for r in rows:
        sig    = phase_table[r]
        wl     = wavelengths[r]

        # Remove slow trend (geometric background) with a polynomial
        trend  = np.polyval(np.polyfit(theta_deg, sig, deg=6), theta_deg)
        ripple = sig - trend

        N    = len(ripple)
        fft  = np.fft.rfft(ripple * np.hanning(N))
        freq = np.fft.rfftfreq(N, d=deg_step)
        pwr  = np.abs(fft)

        ax.plot(freq, pwr, lw=0.8, label=f"{wl:.1f} nm")

        # dominant frequency — skip DC bin
        peak_idx = np.argmax(pwr[1:]) + 1
        dominant_freqs[wl] = freq[peak_idx]
        ax.axvline(freq[peak_idx], lw=0.5, alpha=0.5, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Spatial Frequency (cycles / degree)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    return dominant_freqs


def plot_rainbow_lognorm_zoom(phase_table, wavelengths, theta_deg, rows, ax):
    """
    Plots a log-normal zoomed-in rainbow visualization for phase function values.

    Args:
        phase_table (ndarray): 2D array (wavelengths x angles) of phase function values.
        wavelengths (ndarray): Array of wavelengths corresponding to rows in `phase_table`.
        theta_deg (ndarray): Array of scattering angles corresponding to columns in `phase_table`.
        rows (list): Indices of wavelength rows to visualize.
        ax (Axes): Matplotlib Axes object to draw the plot.
    """
    # Log-normal transform: clip data to avoid log(0) errors
    clipped_phase_table = np.maximum(phase_table, 1e-12)
    log_phase_table = np.log10(clipped_phase_table)

    # Select the angular range of interest (122–144 degrees)
    mask = (theta_deg >= 122.0) & (theta_deg <= 144.0)

    # Plot each selected row (corresponding to specific wavelengths)
    for r in rows:
        ax.plot(theta_deg[mask], log_phase_table[r][mask],
                lw=0.8, label=f"{wavelengths[r]:.1f} nm")

    # Add plot details
    ax.set_title("Rainbow Log-Normal Zoom (122°–144°)")
    ax.set_xlabel("Scattering Angle (degrees)")
    ax.set_ylabel("log₁₀(Phase Function)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


def plot_rainbow_zoom(phase_table, wavelengths, theta_deg, rows, ax):
    mask = (theta_deg >= 100.0) & (theta_deg <= 170.0)
    for r in rows:
        ax.plot(theta_deg[mask], phase_table[r][mask],
                lw=0.8, label=f"{wavelengths[r]:.1f} nm")
    ax.set_title("Rainbow + Glory Skirt Zoom (100°–170°)")
    ax.set_xlabel("Scattering Angle (degrees)")
    ax.set_ylabel("Phase Function")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


def plot_instantaneous_freq(phase_table, wavelengths, theta_deg, rows, ax):
    """
    Plots the instantaneous frequency of the detrended signal via Hilbert.
    This is a rough read — will be noisy where signal amplitude is low,
    but gives us a first look at the chirp rate map ω(θ).
    """
    from scipy.signal import hilbert

    deg_step = theta_deg[1] - theta_deg[0]

    for r in rows:
        sig   = phase_table[r]
        trend = np.polyval(np.polyfit(theta_deg, sig, deg=6), theta_deg)
        ripple = sig - trend

        analytic   = hilbert(ripple)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq  = np.abs(np.diff(inst_phase) / (2.0 * np.pi * deg_step))

        ax.plot(theta_deg[:-1], inst_freq, lw=0.7,
                label=f"{wavelengths[r]:.1f} nm", alpha=0.8)

    ax.set_title("Instantaneous Frequency ω(θ)  [crude Hilbert — expect noise]")
    ax.set_xlabel("Scattering Angle (degrees)")
    ax.set_ylabel("Frequency (cycles / degree)")
    ax.set_ylim(0, 0.5)   # clip runaway noise at low-amplitude regions
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


def main():
    phase_table, wavelengths, theta_deg = load_lut(LUT_PATH)
    num_wl = len(wavelengths)

    rows = resolve_rows(OVERLAY_ROWS, num_wl)
    print(f"[Diag] Overlay rows  : {rows}")
    print(f"[Diag] Wavelengths   : {[f'{wavelengths[r]:.1f}nm' for r in rows]}")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"LUT Diagnostic — {Path(LUT_PATH).name}", fontsize=13, y=0.98)

    ax_lin   = fig.add_subplot(3, 2, 1)
    ax_log   = fig.add_subplot(3, 2, 2)
    ax_fft   = fig.add_subplot(3, 2, 3)
    ax_heat  = fig.add_subplot(3, 2, 4)
    ax_zoom  = fig.add_subplot(3, 2, 5)
    ax_ifreq = fig.add_subplot(3, 2, 6)

    plot_signals(phase_table, wavelengths, theta_deg, rows, ax_lin, ax_log)

    dominant_freqs = plot_fft(phase_table, wavelengths, theta_deg, rows, ax_fft)
    print(f"[Diag] Dominant fringe frequencies:")
    for wl, freq in dominant_freqs.items():
        print(f"[Diag]   {wl:.1f} nm  →  ω_ref = {freq:.5f} cycles/degree")

    plot_rainbow_lognorm_zoom(phase_table, wavelengths, theta_deg, rows, ax_heat)
    plot_rainbow_zoom(phase_table, wavelengths, theta_deg, rows, ax_zoom)
    plot_fft_rainbow_only(phase_table, wavelengths, theta_deg, rows, ax_ifreq)
    #plot_instantaneous_freq(phase_table, wavelengths, theta_deg, rows, ax_ifreq)

    plt.tight_layout()
    out = Path(LUT_PATH).stem + "_diagnostic.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[Diag] Saved: {out}")


if __name__ == "__main__":
    main()
    compare_luts(MONO_PATH, POLY_PATH)
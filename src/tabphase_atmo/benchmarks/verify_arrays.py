import numpy as np
import matplotlib.pyplot as plt


def verify_raw_arrays(your_bin_path: str, mieplot_txt_path: str):
    # 1. Load Your Engine's Data
    from nimbuscore.core.generator.visualize import load_binary_file
    data, n_angles, n_phis, n_wvls, min_w, max_w = load_binary_file(your_bin_path)

    mid_wvl_idx = n_wvls // 2
    # .copy() is required because np.frombuffer creates a read-only view
    your_intensity = data[mid_wvl_idx, 0, :].copy()
    your_theta_deg = np.linspace(0, 180, n_angles)

    # 2. Load MiePlot's ground truth
    print(f"Loading MiePlot data from {mieplot_txt_path}...")
    # Skipping 23 lines to avoid the header; max_rows 18001 captures 0.00 to 180.00
    mie_data = np.loadtxt(mieplot_txt_path, skiprows=23, max_rows=18000, usecols=(0, 3))
    mie_theta_raw = mie_data[:, 0]
    mie_intensity_raw = mie_data[:, 1]

    # 3. Interpolate MiePlot to match NimbusCore resolution
    print(f"Interpolating MiePlot array ({len(mie_theta_raw)} bins) to match NimbusCore ({n_angles} bins)...")
    mie_intensity_mapped = np.interp(your_theta_deg, mie_theta_raw, mie_intensity_raw)

    # --- 3.5 Strict Spherical Normalization ---
    # Patch 0-degree singularity if present
    if your_intensity[0] < 1e-10:
        your_intensity[0] = your_intensity[1]

    # Calculate spherical integral to convert raw intensity to PDF
    # Integral = 2*pi * integral_0^pi ( I(theta) * sin(theta) d_theta )
    d_theta = np.radians(your_theta_deg[1] - your_theta_deg[0])
    solid_angle_weights = 2 * np.pi * np.sin(np.radians(your_theta_deg))

    your_integral = np.sum(your_intensity * solid_angle_weights) * d_theta
    mie_integral = np.sum(mie_intensity_mapped * solid_angle_weights) * d_theta

    # Normalize MiePlot data to be a valid rendering PDF
    mie_pdf = mie_intensity_mapped / mie_integral

    # --- 3.6 Numerical Validation Metrics ---
    your_log = np.log10(your_intensity + 1e-12)
    mie_log = np.log10(mie_pdf + 1e-12)

    # Mean Absolute Error in Log Space (Distance between curves)
    mae_log = np.mean(np.abs(your_log - mie_log))

    # Pearson Correlation Coefficient (Checks if the ripple 'shape' matches)
    correlation = np.corrcoef(your_intensity, mie_pdf)[0, 1]

    # Percentage Similarity (1 - Normalized Root Mean Square Error)
    rmse = np.sqrt(np.mean((your_intensity - mie_pdf) ** 2))
    norm_rmse = rmse / (np.max(mie_pdf) - np.min(mie_pdf))
    accuracy_pct = (1 - norm_rmse) * 100

    print(f"\n" + "=" * 40)
    print(f"   VALIDATION REPORT: NimbusCore vs MiePlot")
    print(f"=" * 40)
    print(f"Wavelength:         {wavelengths[mid_wvl_idx] if 'wavelengths' in locals() else 'Mid-Wvl'} nm")
    print(f"Angular Samples:    {n_angles}")
    print(f"NimbusCore Integral:{your_integral:.6f} (PDF Target: 1.0)")
    print(f"-" * 40)
    print(f"Correlation (R):    {correlation:.6f} (Shape Match)")
    print(f"Mean Log Error:     {mae_log:.4f} orders of mag")
    print(f"ESTIMATED ACCURACY: {accuracy_pct:.2f} %")
    print(f"=" * 40 + "\n")

    # --- 4. Plot ---

    # Increase font sizes globally for this script
    plt.rcParams.update({
        'font.size': 14,            # base font size
        'axes.titlesize': 18,      # title font size
        'axes.labelsize': 16,      # x/y label size
        'xtick.labelsize': 14,     # x tick labels
        'ytick.labelsize': 14,     # y tick labels
        'legend.fontsize': 12,
    })

    plt.figure(figsize=(14, 7))

    # Plot Ground Truth (Thick Blue)
    plt.plot(your_theta_deg, mie_log,
             color='blue', linewidth=5, alpha=0.5, label="MiePlot (Normalizovaný)")

    # Plot Engine (Thin Dashed Red)
    plt.plot(your_theta_deg, your_log,
             color='red', linewidth=1.5, linestyle='--', label="Nimbus")

    # Set axis labels / title with explicit font sizes (redundant but explicit)
    plt.xlabel("Uhol rozptylu (stupňe)", fontsize=16)
    plt.ylabel("Intenzita (Log10)", fontsize=16)
    plt.title(f"Numerická validácia ({accuracy_pct:.2f}% Presnosť)", fontsize=18)

    # Make ticks large and readable
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    # Save both PDF and PNG for compatibility
    plt.savefig("mievalidation.pdf", bbox_inches='tight')
    plt.savefig("mievalidation.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Ensure n_wvls and other metadata are accessible if needed for the title
    verify_raw_arrays(
        "/home/speedlord/mitsuba3/src/tabphase_atmo/benchmarks/cache/miepython/droplet_8192ang-360az-1wl_sphe10um-100w-0pvar.bin",
        "/home/speedlord/mitsuba3/src/tabphase_atmo/benchmarks/plotcompare/plotcompare.txt"
    )

import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

# Set variant
mi.set_variant('llvm_ad_spectral')

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
PLUGIN_FILENAME = "/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/cache/150um_mean_10um_std_32bins_1024ang_cornell_rainbow.bin"


def audit_math():
    print("--- 📉 PDF vs EVAL AUDITOR: STARTED ---")

    # 1. Load
    try:
        phase_func = mi.load_dict({
            'type': 'atmosphericphase',
            'filename': PLUGIN_FILENAME,
            'use_mis' : False
        })
    except Exception as e:
        print(f"💀 FATAL: {e}")
        return

    # 2. Create a dense grid of angles
    num_points = 2000
    theta_vals = dr.linspace(mi.Float, 0, dr.pi, num_points)

    # 3. Construct Geometry
    wi = mi.Vector3f(0, 0, 1)

    # wo rotates around Y axis based on theta
    wo_x = dr.sin(theta_vals)
    wo_y = dr.zeros(mi.Float, num_points)
    wo_z = dr.cos(theta_vals)
    wo = mi.Vector3f(wo_x, wo_y, wo_z)

    # Context setup
    ctx = mi.PhaseFunctionContext(None)
    mei = mi.MediumInteraction3f()
    mei.wi = wi

    # --- CRITICAL FIX 1: Broadcasting ---
    # Don't pass a list of 4 items. Pass a SINGLE scalar Spectrum.
    # Dr.Jit will broadcast this (1 -> 2000) correctly.
    mei.wavelengths = mi.Spectrum(0.550)

    # 4. Evaluate
    # eval_pdf returns (Spectrum, Float)
    result_pair = phase_func.eval_pdf(ctx, mei, wo)
    eval_spec = result_pair[0]  # This is a Spectrum (likely 4 channels)
    pdf_val = result_pair[1]  # This is a Float

    # --- CRITICAL FIX 2: Slicing ---
    # We only want to plot ONE channel (e.g., channel 0).
    # In llvm_ad_spectral, Spectrum is usually [r, g, b, alpha] or similar structure.
    # We access the first component.
    eval_val_scalar = eval_spec[0]

    # 5. Convert to Numpy
    angles_deg = np.rad2deg(theta_vals.numpy())
    y_eval = eval_val_scalar.numpy()  # Now shape (2000,)
    y_pdf = pdf_val.numpy()  # Now shape (2000,)

    # 6. Plotting
    plt.figure(figsize=(12, 8))

    # Subplot 1: The Curves
    plt.subplot(2, 1, 1)
    plt.title(f"The Truth vs. The Lie (Eval vs PDF)\nPlugin: {PLUGIN_FILENAME}")
    plt.plot(angles_deg, y_eval, label='Phase Function (Eval)', color='black', linewidth=2)
    plt.plot(angles_deg, y_pdf, label='Sampling Probability (PDF)', color='red', linestyle='--', linewidth=2)
    plt.yscale('log')
    plt.ylabel("Value (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # Subplot 2: The Ratio
    plt.subplot(2, 1, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = y_eval / y_pdf

    plt.plot(angles_deg, ratio, color='blue', linewidth=1)
    plt.axhline(1.0, color='red', linestyle='--', label="Perfect Match (1.0)")
    plt.title("Weight Ratio (Eval / PDF)")
    plt.xlabel("Scattering Angle (Degrees)")
    plt.ylabel("Ratio")
    plt.yscale('log')  # Log scale helps see if it's off by factor of 10 or 100
    plt.grid(True, alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig("math_audit.png")
    print("✅ Audit complete. Saved to math_audit.png")
    plt.show()


if __name__ == "__main__":
    audit_math()
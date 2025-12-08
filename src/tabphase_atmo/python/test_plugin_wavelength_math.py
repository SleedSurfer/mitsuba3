import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

# Set variant (Must be spectral)
mi.set_variant('llvm_ad_spectral')

from atmospheric import create_atmospheric_phase


def test_plugin_math():
    print("--- PHASE FUNCTION LIE DETECTOR ---")

    # 1. Generate Config (50um for distinct rainbow)
    phase_element = create_atmospheric_phase(
        radius_mean_um=50.0,
        radius_std_um=2.0,
        num_angles=4096,
        num_wavelengths=32,
        note="lie_detector",
        force_regen=False
    )

    # 2. Instantiate Plugin (The Correct Way)
    print(f"Loading plugin from: {phase_element['filename']}")
    try:
        phase_func = mi.load_dict({
            'type': 'atmosphericphase',
            'filename': phase_element['filename']
        })
    except Exception as e:
        print(f"FATAL: Could not load plugin. Did you compile it? {e}")
        return

    # 3. Setup Test Data
    # Angles: 0 to 180 degrees
    num_steps = 1000
    angles_deg = np.linspace(0, 180, num_steps)
    # CosTheta: 1.0 (Forward) to -1.0 (Backward)
    cos_theta_arr = np.cos(np.deg2rad(angles_deg))

    # Wavelengths: Blue, Green, Red
    test_wvls = [450.0, 550.0, 650.0]
    colors = ['blue', 'green', 'red']

    print(f"Sampling {num_steps} angles for {len(test_wvls)} wavelengths...")

    plt.figure(figsize=(12, 7))

    for i, wvl_val in enumerate(test_wvls):
        # --- VECTORIZED QUERY ---

        # 1. Create Wavefront of size 'num_steps'
        # We broadcast the single wavelength to match the number of angles
        wavs = mi.Spectrum(wvl_val)

        # 2. Directions
        # Wi = Forward Z (0,0,1)
        # Wo = Rotated vector
        wi = mi.Vector3f(0, 0, 1)

        ct = mi.Float(cos_theta_arr)
        st = dr.sqrt(1.0 - ct ** 2)
        wo = mi.Vector3f(st, 0, ct)

        # 3. Mock Interaction
        mei = mi.MediumInteraction3f()
        mei.wi = wi
        mei.wavelengths = wavs

        # 4. Context (Sampler not needed for eval)
        ctx = mi.PhaseFunctionContext(None)

        # 5. Evaluate Plugin
        # val is the Phase Function Value, pdf is the sampling probability
        val, pdf = phase_func.eval_pdf(ctx, mei, wo)

        # Download data
        y_values = val[0].numpy().flatten()

        # Plot (Log Scale because Mie scattering is extreme)
        plt.semilogy(angles_deg, y_values, label=f"{int(wvl_val)}nm", color=colors[i], linewidth=1.5)

    # --- ANALYSIS & PLOT STYLING ---
    plt.title(
        "Plugin Verification: 50µm Water Droplets\n(If these lines overlap exactly, Spectral Interpolation is BROKEN)")
    plt.xlabel("Scattering Angle (Degrees)")
    plt.ylabel("Phase Function Value (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()

    # Highlight Key Features
    plt.axvline(0, color='k', linestyle='--', alpha=0.5, label="Forward (0°)")
    plt.axvline(138, color='purple', linestyle='--', alpha=0.8, label="Rainbow (~138°)")
    plt.axvline(180, color='grey', linestyle='--', alpha=0.5, label="Glory (180°)")

    output_img = "plugin_verification_plot.png"
    plt.savefig(output_img)
    print(f"✅ Saved verification plot to {output_img}")
    plt.show()


if __name__ == "__main__":
    test_plugin_math()
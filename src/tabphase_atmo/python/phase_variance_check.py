import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

# Set variant (Must match your plugin compilation)
mi.set_variant('llvm_ad_spectral')

# ---------------------------------------------------------
# SETUP: Point this to your generated .bin file
# ---------------------------------------------------------
PLUGIN_FILENAME = "cache/150um_mean_10um_std_32bins_1024ang_cornell_rainbow.bin"
RADIUS_UM = 20.0  # Just for labelling, logic comes from the bin file

def hunt_fireflies():
    print("--- 🔫 VARIANCE HUNTER: STARTED ---")

    # 1. Load the Plugin
    print(f"Loading plugin: {PLUGIN_FILENAME}")
    try:
        phase_func = mi.load_dict({
            'type': 'atmosphericphase',
            'filename': PLUGIN_FILENAME
        })
    except Exception as e:
        print(f"💀 FATAL: Could not load plugin. {e}")
        return

    # 2. The Setup
    # We simulate 1,000,000 rays hitting a particle
    n_samples = 1_000_000
    print(f"Simulating {n_samples} scattering events...")

    # Dummy Interaction (Forward incident ray)
    wi = mi.Vector3f(0, 0, 1)

    # We test just ONE wavelength (Green 550nm) to isolate geometric variance
    # If your Green channel sampling is perfect, weights should be exactly 1.0 here.
    # If they aren't, your 'reverse' logic or distribution creation is slightly off.
    wavs = mi.Spectrum(550.0)

    mei = mi.MediumInteraction3f()
    mei.wi = wi
    mei.wavelengths = wavs

    # Random number generator
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(0, n_samples)

    ctx = mi.PhaseFunctionContext(sampler)

    # 3. HAMMER THE SAMPLE METHOD
    # This calls your C++ sample() implementation
    wo, weight_spec, pdf = phase_func.sample(ctx, mei, sampler.next_1d(), sampler.next_2d())

    # 4. Data Extraction
    # Calculate scattering angle theta for every sample
    # wo.z is cos(theta) because wi is (0,0,1)
    cos_theta = wo.z
    angles_deg = np.rad2deg(np.arccos(cos_theta.numpy().flatten()))

    # Extract weights (Energy throughput)
    # We only care about the channel we simulated (index 0)
    weights = weight_spec[0].numpy().flatten()
    pdfs = pdf.numpy().flatten()

    # 5. The Analysis
    print("--- ANALYSIS ---")
    max_weight = np.max(weights)
    avg_weight = np.mean(weights)

    print(f"Average Weight: {avg_weight:.4f} (Should be close to 1.0)")
    print(f"Max Weight:     {max_weight:.4f} (This is your firefly brightness)")

    # Filter for 'Fireflies' (Weights > 5x average)
    firefly_threshold = 5.0
    firefly_mask = weights > firefly_threshold
    firefly_angles = angles_deg[firefly_mask]
    firefly_weights = weights[firefly_mask]

    print(f"Fireflies detected: {len(firefly_weights)} ({len(firefly_weights)/n_samples*100:.2f}%)")

    # 6. Plotting
    plt.figure(figsize=(14, 8))

    # Subplot 1: All samples
    plt.subplot(2, 1, 1)
    plt.title(f"Monte Carlo Weights vs Scattering Angle (Max: {max_weight:.1f})")
    plt.scatter(angles_deg, weights, alpha=0.1, s=1, c='blue', label='Valid Sample')

    # Highlight Fireflies
    if len(firefly_weights) > 0:
        plt.scatter(firefly_angles, firefly_weights, alpha=0.8, s=10, c='red', label='FIREFLIES')

    plt.axhline(1.0, color='k', linestyle='--', label="Ideal Weight (1.0)")
    plt.ylabel("MC Weight (Value / PDF)")
    plt.yscale('log') # Log scale because fireflies are HUGE
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # Subplot 2: Zoom on Forward Peak (0 - 5 degrees)
    plt.subplot(2, 1, 2)
    plt.title("Zoom: Forward Scattering Peak (0° - 5°)")

    mask_zoom = angles_deg < 5.0
    plt.scatter(angles_deg[mask_zoom], weights[mask_zoom], alpha=0.3, s=5, c='orange')
    plt.axhline(1.0, color='k', linestyle='--')
    plt.xlabel("Scattering Angle (Degrees)")
    plt.ylabel("MC Weight")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("variance_analysis.png")
    print("✅ Plot saved to variance_analysis.png")
    plt.show()

if __name__ == "__main__":
    hunt_fireflies()
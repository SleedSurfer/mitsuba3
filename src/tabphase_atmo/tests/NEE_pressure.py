import numpy as np
import struct
import argparse
import matplotlib.pyplot as plt


def load_atmphase(filename):
    with open(filename, "rb") as f:
        magic = f.read(8)
        if magic != b"ATMPHASE":
            raise ValueError(f"Invalid file signature: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        num_angles = struct.unpack('<I', f.read(4))[0]
        num_wavelengths = struct.unpack('<I', f.read(4))[0]
        min_wavelength = struct.unpack('<f', f.read(4))[0]
        max_wavelength = struct.unpack('<f', f.read(4))[0]

        total_floats = num_angles * num_wavelengths
        data_flat = np.frombuffer(f.read(total_floats * 4), dtype=np.float32)

        phase_table = data_flat.reshape(num_angles, num_wavelengths)

    return phase_table, num_angles, num_wavelengths


def build_cdf_cpp_replica(data, res, channels):
    """
    Replicates the C++ constructor's trapezoidal integration over mu.
    """
    cdf = np.zeros((res, channels), dtype=np.float64)
    norms = np.zeros(channels, dtype=np.float64)

    d_theta = np.pi / (res - 1)
    thetas = np.arange(res) * d_theta
    mus = np.cos(thetas)

    for c in range(channels):
        integral = 0.0
        cdf[0, c] = 0.0
        for i in range(1, res):
            d_mu = mus[i - 1] - mus[i]
            p0 = data[i - 1, c]
            p1 = data[i, c]
            integral += 0.5 * (p0 + p1) * d_mu
            cdf[i, c] = integral

        norm = float(integral)
        if norm <= 0: norm = 1.0
        norms[c] = norm
        cdf[:, c] /= norm
        cdf[-1, c] = 1.0

    return cdf, norms


# --------------------------------------------------------------------------------
# TEST 1: Index-First Logic (The "Fixed" Version)
# --------------------------------------------------------------------------------

def run_diagnostic(data, cdf, norms, res, channel_idx, n_samples=200000, seed=42):
    np.random.seed(seed)
    print(f"\n[Center-Bin Fixed] Diagnostic for Channel {channel_idx}...")

    u = np.random.rand(n_samples)
    idx = np.searchsorted(cdf[:, channel_idx], u, side='right')
    idx0 = np.clip(idx - 1, 0, res - 1).astype(int)

    # --- THE FIX: Sample the CENTER of the bin, not the edge ---
    # Old: theta_sample = idx0 * d_theta
    # New: (idx0 + 0.5) puts us safely away from the floor() cliff
    d_theta = np.pi / (res - 1)
    theta_sample = (idx0 + 0.5) * d_theta

    # -----------------------------------------------------------

    mu_sample = np.cos(theta_sample).astype(np.float64)

    # Round-trip reconstruction
    theta_recon = np.arccos(np.clip(mu_sample, -1.0, 1.0))
    idx_recon = np.floor(theta_recon * (res - 1) / np.pi).astype(int)
    idx_recon = np.clip(idx_recon, 0, res - 1)

    raw_sample = data[idx0, channel_idx].astype(np.float64)
    raw_eval = data[idx_recon, channel_idx].astype(np.float64)
    norm = norms[channel_idx].astype(np.float64)

    # Note: We divide by 2pi here to simulate the "Correct" PDF unit
    # In the final C++ we will remove this to suppress NEE, but for
    # variance testing we keep the units consistent.
    pdf_sample = (raw_sample / norm) * (1.0 / (2.0 * np.pi))
    pdf_sample = np.maximum(pdf_sample, 1e-30)

    weights = raw_eval / pdf_sample

    mean_w = np.mean(weights)
    var_w = np.var(weights)
    expected = norm * 2.0 * np.pi

    print(f"Mean Weight: {mean_w:.8f} (Expected: {expected:.8f})")
    print(f"Variance:    {var_w:.6e}")

    if var_w < 1e-20:
        print("RESULT: PERFECT STABILITY.")
    else:
        print("RESULT: STILL DRIFTING.")

def weight_distribution_test(data, cdf, norms, res, channel_idx):
    print(f"\n[Index-First] Distribution Test for Channel {channel_idx}...")
    N_SAMPLES = 1000000
    u = np.random.rand(N_SAMPLES)

    # 1. Simulate Phase Sampling (Index-First)
    cdf_col = cdf[:, channel_idx]
    idx0 = np.searchsorted(cdf_col, u, side='right') - 1
    idx0 = np.clip(idx0, 0, res - 1)

    raw_values = data[idx0, channel_idx]
    norm = norms[channel_idx]

    # Phase PDF & Weight
    pdf = (raw_values / norm) * (1.0 / (2.0 * np.pi))
    value = raw_values
    weights = value / pdf

    # 2. Simulate NEE (Light Sampling)
    u_nee = np.random.rand(N_SAMPLES)
    mu_nee = 2.0 * u_nee - 1.0
    theta_nee = np.arccos(mu_nee)
    idx_nee = np.floor(theta_nee * (res - 1) / np.pi).astype(int)

    val_nee = data[idx_nee, channel_idx]
    weights_nee = val_nee * (4.0 * np.pi)  # Value / (1/4pi)

    # Reporting
    print(f"Phase Weight Range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    print(f"NEE Weight Range:   [{np.min(weights_nee):.4f}, {np.max(weights_nee):.4f}]")

    plt.figure(figsize=(10, 5))
    plt.hist(weights_nee, bins=100, log=True, color='blue', alpha=0.7, label='NEE Weights')
    plt.axvline(np.mean(weights), color='red', linestyle='dashed', label='Phase Sampling Weight')
    plt.title(f"[Index-First] Weight Distribution (Channel {channel_idx})")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency (Log Scale)")
    plt.legend()
    plt.show()


# --------------------------------------------------------------------------------
# TEST 2: Original Logic (Interpolation + Drift)
# --------------------------------------------------------------------------------

def lookup_interpolated_python(data, angle_idx, channel_idx, res):
    """ Helper to simulate linear interpolation lookup """
    a0 = np.floor(angle_idx).astype(int)
    a1 = np.minimum(a0 + 1, res - 1)
    t = angle_idx - a0
    v0 = data[a0, channel_idx]
    v1 = data[a1, channel_idx]
    return (1.0 - t) * v0 + t * v1


def run_original_logic_test(data, cdf, norms, res, channel_idx):
    print(f"\n[Original Logic] Drift & Interpolation Test for Channel {channel_idx}...")
    norm = norms[channel_idx]
    N_SAMPLES = 1000000
    u = np.random.rand(N_SAMPLES).astype(np.float32)

    # --- 1. SAMPLE STEP (Linear CDF Inversion) ---
    cdf_col = cdf[:, channel_idx]
    idx_lo = np.searchsorted(cdf_col, u, side='right') - 1
    idx_lo = np.clip(idx_lo, 0, res - 2)

    c0 = cdf_col[idx_lo]
    c1 = cdf_col[idx_lo + 1]
    denom = np.maximum(c1 - c0, 1e-12)
    s = (u - c0) / denom
    s = np.clip(s, 0.0, 1.0)

    # Original logic calculates float index
    angle_idx_sample = idx_lo.astype(np.float32) + s
    theta_sample = angle_idx_sample * (np.pi / (res - 1))

    # --- 2. DRIFT SIMULATION (The "Round Trip") ---
    # Simulate: Theta -> Vector -> Dot -> Theta
    # This checks if float precision causes us to evaluate a different spot than we sampled
    mu_sample = np.cos(theta_sample).astype(np.float32)
    theta_recon = np.arccos(np.clip(mu_sample, -1.0, 1.0))
    angle_idx_recon = theta_recon * ((res - 1) / np.pi)

    # --- 3. EVALUATE ---
    # Sample PDF: Calculated at the EXACT sample point (idx_sample)
    val_sample_pt = lookup_interpolated_python(data, angle_idx_sample, channel_idx, res)
    pdf_sample = val_sample_pt / norm

    # Eval Value: Calculated at the RECONSTRUCTED point (idx_recon)
    val_eval = lookup_interpolated_python(data, angle_idx_recon, channel_idx, res)

    # Weight = Value / PDF
    weights = val_eval / pdf_sample

    # --- 4. NEE Comparison ---
    u_nee = np.random.rand(N_SAMPLES)
    mu_nee = 2.0 * u_nee - 1.0
    theta_nee = np.arccos(mu_nee)
    idx_nee = theta_nee * ((res - 1) / np.pi)
    val_nee = lookup_interpolated_python(data, idx_nee, channel_idx, res)
    weights_nee = val_nee * (4.0 * np.pi)

    print(f"Phase Weight Mean: {np.mean(weights):.6f} (Target: {norm:.6f})")
    print(f"Phase Weight Var:  {np.var(weights):.6e}")

    plt.figure(figsize=(10, 5))
    plt.hist(weights_nee, bins=100, log=True, color='blue', alpha=0.5, label='NEE (Light Sampling)')
    plt.hist(weights, bins=100, log=True, color='red', alpha=0.7, label='Phase Sampling (Original)')
    plt.axvline(norm, color='black', linestyle='dashed', label='Ideal Norm')
    plt.title(f"[Original Logic] Variance Analysis (Channel {channel_idx})")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency (Log)")
    plt.legend()
    plt.show()


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # Hardcoded path as provided
    path = "/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/benchmarks/cache/miepython/0um_mean_0um_std_64bins_4096ang_glory_020_003.bin"

    data, res, channels = load_atmphase(path)
    cdf, norms = build_cdf_cpp_replica(data, res, channels)

    test_indices = [0, channels // 2, channels - 1]

    print("========================================")
    print("TEST SUITE 1: Index-First (Proposed Fix)")
    print("========================================")
    for c_idx in test_indices:
        run_diagnostic(data, cdf, norms, res, c_idx)

    # Uncomment to see Index-First histograms
    # for c_idx in test_indices:
    #     weight_distribution_test(data, cdf, norms, res, c_idx)

    print("\n========================================")
    print("TEST SUITE 2: Original Logic (Drift Check)")
    print("========================================")
    for c_idx in test_indices:
        run_original_logic_test(data, cdf, norms, res, c_idx)
import struct
import numpy as np


def validate_lut_energy(filename):
    print(f"--- VALIDATING: {filename} ---")

    with open(filename, "rb") as f:
        # 1. Read Header Explicitly (28 Bytes Total)
        magic = f.read(8)
        if magic != b"ATMPHASE":
            print(f"❌ ERROR: Invalid Magic Number: {magic}")
            return

        version = struct.unpack("<I", f.read(4))[0]
        res = struct.unpack("<I", f.read(4))[0]
        chans = struct.unpack("<I", f.read(4))[0]
        min_w = struct.unpack("<f", f.read(4))[0]
        max_w = struct.unpack("<f", f.read(4))[0]

        print(f"  Header Info: Res={res}, Chans={chans}, Ver={version}")

        # 2. Calculate Exact Data Size
        # each float is 4 bytes
        expected_bytes = res * chans * 4

        # 3. Read ONLY the Phase Data
        buffer = f.read(expected_bytes)

        if len(buffer) != expected_bytes:
            print(f"❌ ERROR: Unexpected End of File. Expected {expected_bytes} bytes, got {len(buffer)}")
            return

        # 4. Convert to Numpy
        data = np.frombuffer(buffer, dtype=np.float32).reshape(res, chans)

    # --- INTEGRATION CHECK ---
    # Reconstruct the Linear Grid
    theta = np.linspace(0, np.pi, res)

    # Check the middle wavelength channel
    mid_idx = chans // 2
    phase_slice = data[:, mid_idx]

    # Integrate p(theta) * sin(theta) * 2pi
    # We use sin(theta) because we are integrating over the solid angle sphere
    # 1. Reconstruct Cosine Grid
    mu = np.cos(theta)

    # 2. Integrate over d_mu (Trapz expects x to be increasing, so flip it)
    # mu is 1 -> -1 (decreasing). We want integral from -1 to 1 or just abs value.
    integral = np.abs(np.trapezoid(phase_slice, x=mu)) * 2 * np.pi

    print(f"  Calculated Integral (Cosine Method): {integral:.6f}")

    print(f"  Calculated Integral (Mid-Channel): {integral:.6f}")

    if abs(integral - 1.0) > 0.05:
        print("❌ CRITICAL FAILURE: Normalization is BROKEN.")
        print("   -> The phase function does not sum to 1.0.")
    else:
        print("✅ INTEGRITY PASS: Energy is conserved (within 5% tolerance).")


if __name__ == "__main__":
    # Update this path to your actual binary file
    validate_lut_energy("scenes/cache/miepython/300um_mean_6um_std_64bins_8192ang_radius_300.00.bin")
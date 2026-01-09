import struct
import numpy as np
import sys


def inspect_binary(filename):
    print(f"--- INSPECTING: {filename} ---")

    with open(filename, 'rb') as f:
        # 1. HEADER CHECK
        magic = f.read(8)
        if magic != b'ATMPHASE':
            print(f"❌ FATAL: Invalid Magic Header. Found: {magic}")
            return

        version = struct.unpack('<I', f.read(4))[0]
        num_angles = struct.unpack('<I', f.read(4))[0]
        num_wavelengths = struct.unpack('<I', f.read(4))[0]
        min_wvl = struct.unpack('<f', f.read(4))[0]
        max_wvl = struct.unpack('<f', f.read(4))[0]

        print(f"✅ Magic: {magic}")
        print(f"version: {version}")
        print(f"num_angles (Resolution): {num_angles}")
        print(f"num_wavelengths (Channels): {num_wavelengths}")
        print(f"Wavelength Range: {min_wvl:.1f}nm - {max_wvl:.1f}nm")

        expected_floats = num_angles * num_wavelengths
        print(f"Expected Data Size: {expected_floats} floats ({expected_floats * 4 / 1024 / 1024:.2f} MB)")

        # 2. DATA CHECK
        raw_data = f.read()
        float_count = len(raw_data) // 4

        if float_count != expected_floats:
            print(f"❌ SIZE MISMATCH: Header expects {expected_floats} floats, but found {float_count}.")
            return

        data = np.frombuffer(raw_data, dtype=np.float32)

        # 3. INTERLEAVING VERIFICATION
        print("\n--- DATA LAYOUT SAMPLE ---")

        # Angle 0
        print(f"Angle Index 0 (Forward, cos=1.0):")
        start_idx = 0
        end_idx = start_idx + num_wavelengths
        block_0 = data[start_idx:end_idx]
        print(f"  First 5 wavelengths: {block_0[:5]}")
        print(f"  Min Value in block: {np.min(block_0):.4e}")
        print(f"  Max Value in block: {np.max(block_0):.4e} (Should be Massive)")

        # Angle N-1
        print(f"\nAngle Index {num_angles - 1} (Backward, cos=-1.0):")
        start_idx = (num_angles - 1) * num_wavelengths
        end_idx = start_idx + num_wavelengths
        block_n = data[start_idx:end_idx]
        print(f"  First 5 wavelengths: {block_n[:5]}")
        print(f"  Max Value in block: {np.max(block_n):.4e}")

        # Mid angle
        mid_idx = num_angles // 2
        print(f"\nAngle Index {mid_idx} (Side, cos~0.0):")
        start_idx = mid_idx * num_wavelengths
        end_idx = start_idx + num_wavelengths
        block_mid = data[start_idx:end_idx]
        print(f"  Max Value in block: {np.max(block_mid):.4e} (Should be lower)")

        # 4. Explicit reshape check: [angle, wavelength]
        data_2d = data.reshape((num_angles, num_wavelengths))
        print("\n--- 2D INDEX CHECK (angle, wavelength) ---")
        print(f"data_2d[0, 0] (Angle 0, Wvl 0): {data_2d[0, 0]:.6e}")
        print(f"data_2d[-1, 0] (Angle {num_angles-1}, Wvl 0): {data_2d[-1, 0]:.6e}")


if __name__ == "__main__":
    target_file = "../cache/miepython/150um_mean_10um_std_32bins_1024ang_cornell_rainbow.bin"
    inspect_binary(target_file)
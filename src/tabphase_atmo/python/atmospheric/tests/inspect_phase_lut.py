import struct
import numpy as np
import os


def binary_to_txt(bin_filename, txt_filename):
    if not os.path.exists(bin_filename):
        print(f"Error: File '{bin_filename}' not found.")
        return

    file_size = os.path.getsize(bin_filename)

    with open(bin_filename, 'rb') as f:
        # 1. Read Header
        magic = f.read(8)
        if magic != b'ATMPHASE':
            print(f"Error: Bad magic number: {magic}")
            return

        version = struct.unpack('<I', f.read(4))[0]
        num_angles = struct.unpack('<I', f.read(4))[0]
        num_wavelengths = struct.unpack('<I', f.read(4))[0]
        min_wl = struct.unpack('<f', f.read(4))[0]
        max_wl = struct.unpack('<f', f.read(4))[0]

        print(f"--- Header Info ---")
        print(f"  Version: {version}")
        print(f"  Resolution (Angles): {num_angles}")
        print(f"  Channels (Wavelengths): {num_wavelengths}")
        print(f"  Range: {min_wl:.2f}nm - {max_wl:.2f}nm")

        # 2. Read Data
        # We calculate exactly how many bytes of float data to read.
        # This safely stops before the "MIS metadata" footer.
        count = num_angles * num_wavelengths
        data_bytes = f.read(count * 4)

        if len(data_bytes) != count * 4:
            print(f"Error: Expected {count * 4} bytes of data, got {len(data_bytes)}. File might be truncated.")
            return

        # Convert bytes to float array
        data = np.frombuffer(data_bytes, dtype=np.float32)

        # Reshape: Your writer used `phase_table.T.flatten()`
        # Assuming phase_table was (Wavelengths, Angles), .T is (Angles, Wavelengths)
        # So we reshape to (Angles, Wavelengths)
        table = data.reshape((num_angles, num_wavelengths))

        # 3. Critical Analysis
        min_val = np.min(table)
        max_val = np.max(table)
        negatives = table[table < 0]
        nans = table[np.isnan(table)]
        infs = table[np.isinf(table)]

        print(f"\n--- Data Analysis ---")
        print(f"  Global Max: {max_val:.6e}")
        print(f"  Global Min: {min_val:.6e}")

        if len(nans) > 0:
            print(f"  [CRITICAL] Found {len(nans)} NaNs!")

        if len(infs) > 0:
            print(f"  [CRITICAL] Found {len(infs)} Infinite values!")

        if len(negatives) > 0:
            print(f"  [FAIL] Found {len(negatives)} NEGATIVE values.")
            print(f"  First 5 violations:")
            neg_indices = np.argwhere(table < 0)
            for i in range(min(5, len(neg_indices))):
                ang_idx, wl_idx = neg_indices[i]
                val = table[ang_idx, wl_idx]

                # Approximate physical values for context
                # Assuming linear spacing for mu from 1 to -1
                mu = 1.0 - 2.0 * (ang_idx / (num_angles - 1))
                wl = min_wl + (max_wl - min_wl) * (wl_idx / (num_wavelengths - 1)) if num_wavelengths > 1 else min_wl

                print(f"    - Index [A:{ang_idx}, W:{wl_idx}] (mu≈{mu:.3f}, wl≈{wl:.1f}nm) = {val}")
        else:
            print("  [SUCCESS] 0 Negative values found in binary data.")

        # 4. Export to CSV for manual inspection
        print(f"\n--- Exporting ---")
        print(f"Writing to {txt_filename}...")
        with open(txt_filename, 'w') as txt:
            txt.write(f"# Header: Res={num_angles}, Ch={num_wavelengths}, Range=[{min_wl}, {max_wl}]\n")
            # Create a header row for wavelengths
            wl_headers = np.linspace(min_wl, max_wl, num_wavelengths)
            txt.write("AngleIdx," + ",".join([f"{w:.1f}nm" for w in wl_headers]) + "\n")

            for r in range(num_angles):
                row_data = table[r]
                # Format to scientific notation for precision
                row_str = ",".join([f"{x:.6e}" for x in row_data])
                txt.write(f"{r},{row_str}\n")

        print("Done.")


# --- RUN IT HERE ---
# Replace with your actual filename
binary_to_txt("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/tests/scenes/cache/miepython/4um_mean_0um_std_64bins_8192ang_glory.bin", "atmosphere_dump.csv")
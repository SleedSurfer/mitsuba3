"""
test_generation.py
Validates the generator using the Config as the source of truth.
"""
import os
from config import MieConfig
from generate import generate_mie_table, save_binary_file
# Assuming visualize.py is unchanged
from visualize import visualize_binary_file

def test_generation():
    print("=" * 60)
    print("Atmospheric Phase Function: Configuration Test")
    print("=" * 60)

    # 1. Load Configuration
    # You can override defaults here if you want a fast test
    # e.g., cfg = MieConfig(num_wavelengths=16, num_angles=2048)
    cfg = MieConfig()

    print(f"Testing Config: {cfg.output_filename}")
    print(f"  Wavelengths: {cfg.num_wavelengths}")
    print(f"  Angles:      {cfg.num_angles}")
    print(f"  Grid Steps:  {cfg.num_samples}")

    # 2. Generate
    print("\nRunning Generator...")
    phase_table = generate_mie_table(cfg)

    # 3. Save
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{cfg.output_filename}.bin")

    save_binary_file(output_path, phase_table, cfg)

    # 4. Verify Integrity (The 'Audit')
    if os.path.exists(output_path):
        actual_size = os.path.getsize(output_path)
        expected_size = cfg.get_expected_file_size()

        print(f"\nVerification:")
        print(f"  Actual Size:   {actual_size:,} bytes")
        print(f"  Expected Size: {expected_size:,} bytes")

        if actual_size == expected_size:
            print("  [PASS] File structure is correct.")
        else:
            print("  [FAIL] Size mismatch!")
            return

    # 5. Visualize
    print(f"\nGenerating Graph...")
    graph_path = os.path.join("graphs", f"{cfg.output_filename}.png")
    os.makedirs("graphs", exist_ok=True)
    visualize_binary_file(output_path, graph_path)
    print(f"  Graph saved to {graph_path}")

if __name__ == '__main__':
    test_generation()
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr

# Adjust this import to match your actual package name
# Assuming your package structure is like: my_renderer/generator/...
# If you are running this from inside the package, change to relative imports.

from .wrapper import create_atmospheric_phase


def test_raytracer_explicit():
    print("=== TEST: Dr.Jit Raytracer Backend Integration ===")

    # 1. Force Dr.Jit Backend Check
    # We want to know if we are burning GPU or CPU cycles.
    if dr.has_backend(dr.JitBackend.CUDA):
        print(f"✅ [Dr.Jit] CUDA Backend Available. Expecting GPU execution.")
    elif dr.has_backend(dr.JitBackend.LLVM):
        print(f"⚠️ [Dr.Jit] CUDA not found. Falling back to LLVM (CPU AVX).")
    else:
        print(f"❌ [Dr.Jit] No JIT backend found! This will be slow/broken.")

    # 2. Define Parameters for a "Rainbow-Optimized" Droplet
    # Large radius (1.5mm) to ensure Geometric Optics is valid.
    # Low std_dev to make the primary rainbow sharp.
    params = {
        "radius_mean_um": 220.0,  # 1.5mm (Huge drop)
        "radius_std_um": 60.0,  # Very uniform size
        "num_angles": 360,  # High res to see the spike
        "num_wavelengths": 8,  # Keep it fast
        "backend": "jit_traced",  # <--- EXPLICIT CALL
        "force_regen": True,  # Force it to run
        "note": "help"
    }

    # params = {
    #         "radius_mean_um": 50.0,  # 1.5mm (Huge drop)
    #         "radius_std_um": 2.0,  # Very uniform size
    #         "num_angles": 1024,  # High res to see the spike
    #         "num_wavelengths": 16,  # Keep it fast
    #         "backend": "jit_traced",  # <--- EXPLICIT CALL
    #         "force_regen": True,  # Force it to run
    #         "note": "test_small_droplets"
    # }

    print(f"\n--- Invoking Wrapper with backend='{params['backend']}' ---")

    try:
        result = create_atmospheric_phase(**params)
        filepath = result['filename']
        print(f"✅ [Success] Wrapper finished. Output: {filepath}")
    except Exception as e:
        print(f"❌ [Failure] Wrapper crashed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_raytracer_explicit()
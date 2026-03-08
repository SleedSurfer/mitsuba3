import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.core import MieConfig
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


    params = MieConfig(radius_mean_um=120.0,
                       variance=0.0,
                       num_angles=8192,
                       num_wavelengths=10,
                       brute_force_integration=True,
                       note="backend_visual_differences_hybrid")

    try:
        result = create_atmospheric_phase(params,backend="auto",force_regen=True)
        filepath = result['filename']
        print(f"✅ [Success] Wrapper finished. Output: {filepath}")
    except Exception as e:
        print(f"❌ [Failure] Wrapper crashed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_raytracer_explicit()
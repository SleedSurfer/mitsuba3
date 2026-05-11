import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from nimbuscore.core.config import BackendType, Particle, DropletShape
import drjit as dr

# Adjust this import to match your actual package name
# Assuming your package structure is like: my_renderer/generator/...
# If you are running this from inside the package, change to relative imports.

from .gen_manager import create_atmospheric_phase


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

    config = Particle.droplet(radius_mean_um=300,
                              variance=0.0,
                              composition=[(DropletShape.SPHERE,1.0)],
                              backend=BackendType.DRJIT,
                              num_angles=2048,
                              num_wavelengths=32,
                              num_phi_bins=360
                              )

    try:
        result = create_atmospheric_phase(config,force_regen=True,)
        filepath = result['filename']
        print(f"✅ [Success] Wrapper finished. Output: {filepath}")
    except Exception as e:
        print(f"❌ [Failure] Wrapper crashed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_raytracer_explicit()
import pytest
import drjit as dr
from drjit.auto import Float, Complex2f  # Swap to llvm or auto if needed
import math
import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric.ray_emitter import GridEmitter


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture
def grid_data():
    """
    Spawns a 10x10 ray grid spanning 2.0 mm.
    This fixture gets injected into any test that requests 'grid_data'.
    """
    rays, patches = GridEmitter.emit(grid_res=10, width_mm=2.0)
    return rays, patches


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------
def test_void_propagation(grid_data):
    rays, _ = grid_data
    distance_to_move = 15.0  # mm

    # Move rays forward
    rays.o += rays.d * distance_to_move

    # Update optical path (index of refraction of vacuum is 1.0)
    rays.l += distance_to_move * 1.0

    assert dr.allclose(rays.l, Float(15.0)), "Optical path tracking is cooked."


def test_destructive_interference():
    # Wave 1: Amplitude 1, Phase 0 (Real=1, Imag=0)
    w1 = Complex2f(1.0, 0.0)

    # Wave 2: Amplitude 1, Phase PI (Real=-1, Imag=0 -> perfectly out of phase)
    w2 = Complex2f(math.cos(math.pi), math.sin(math.pi))

    interference = w1 + w2

    assert dr.allclose(dr.abs(interference), Float(0.0),
                       atol=1e-6), "Wave math is borked. Destructive interference failed."


def test_phasor_initialization(grid_data):
    rays, _ = grid_data

    assert dr.allclose(dr.real(rays.Ex), 1.0), "Ex real part initialization failed."
    assert dr.allclose(dr.imag(rays.Ex), 0.0), "Ex imaginary part initialization failed."
    assert dr.allclose(dr.real(rays.Ey), 1.0), "Ey real part initialization failed."
    assert dr.allclose(dr.imag(rays.Ey), 0.0), "Ey imaginary part initialization failed."


def test_grid_boundaries(grid_data):
    rays, _ = grid_data

    # We requested a 2.0 mm width. Rays should strictly span -1.0 to 1.0.
    assert dr.allclose(dr.min(rays.o.x), -1.0, atol=1e-5), "Grid min boundary is fucked."
    assert dr.allclose(dr.max(rays.o.x), 1.0, atol=1e-5), "Grid max boundary is fucked."


def test_patch_indexing(grid_data):
    rays, patches = grid_data

    num_rays = 10 * 10
    # The highest index in any patch MUST be less than the total number of rays
    assert dr.all(patches.v3 < num_rays), "Patch memory contains an out-of-bounds pointer."

    # A 10x10 grid should produce exactly 81 patches
    assert len(patches.v0) == 81, f"Expected 81 patches, got {len(patches.v0)}."
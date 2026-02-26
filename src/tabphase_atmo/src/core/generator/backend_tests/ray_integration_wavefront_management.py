import pytest
import numpy as np
import drjit as dr
from drjit.auto import Float, Bool
import sys
import os
import math

from src.core.generator.backends.go_grid import DrJitRaytracerBackend
from src.core.generator.backends.geometric.collector import CollectionSphere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your actual backend path (adjust if your folder structure differs)




def test_master_bounce_loop():
    print("\n--- Running Backend Pipeline Integration Test ---")

    # 1. Setup a low-res backend (100x100 grid = 10,000 rays) for speed
    backend = DrJitRaytracerBackend(grid_res=100, particle_shape="sphere")

    # Dummy parameters for the contract
    m = complex(1.333, 0.0)  # Water IOR
    x = 1000.0  # Size parameter
    mu = np.linspace(1, -1, 10)  # 10 dummy angles

    # 2. Fire the engine
    _ = backend.intensity_unpolarized(m, x, mu)

    # 3. Interrogate the results
    wavefronts = backend._last_wavefronts

    # Check A: Did we get exactly 4 bounces?
    assert len(wavefronts) == 4, f"Pipeline failed. Expected 4 exiting wavefronts, got {len(wavefronts)}."

    print("\nWavefront Manifest:")
    for rays, patches, name in wavefronts:
        print(f"  - {name} | Rays: {len(rays.l)} | Patches: {len(patches.v0)}")

    # Extract specific bounces
    p0_rays, _, p0_name = wavefronts[0]
    p1_rays, _, p1_name = wavefronts[1]
    p2_rays, _, p2_name = wavefronts[2]  # Primary Rainbow
    p3_rays, _, p3_name = wavefronts[3]  # Secondary Rainbow

    # Check B: Focal Line Physics
    # (Checking if the particle contract handed out the right pi/2 phase shifts)
    assert dr.all(p0_rays.f == 0), "p=0 (External Reflection) should have 0 focal lines."
    assert dr.all(p1_rays.f == 0), "p=1 (Direct Transmission) should have 0 focal lines."
    assert dr.all(p2_rays.f == 1), "p=2 (Primary Rainbow) should have 1 focal line."
    assert dr.all(p3_rays.f == 2), "p=3 (Secondary Rainbow) should have 2 focal lines."

    print("[PASS] Focal line phase shifts applied perfectly across all bounces.")

    # Check C: Optical Path Length (l) validation
    # The optical path should strictly increase with every internal bounce
    # We grab the max optical path of the dead-center ray to verify the water traversal
    l0 = dr.max(p0_rays.l)
    l1 = dr.max(p1_rays.l)
    l2 = dr.max(p2_rays.l)
    l3 = dr.max(p3_rays.l)

    assert l1 > l0, "Optical path did not increase going inside the droplet!"
    assert l2 > l1, "Optical path did not increase on internal bounce 1!"
    assert l3 > l2, "Optical path did not increase on internal bounce 2!"

    print("[PASS] Optical path lengths (l) properly accumulating physical distance.")
    print("[PASS] Master Bounce Loop is structurally flawless.")


def test_phase_coherence():
    print("\n--- Running Phase Coherence Sanity Check ---")

    # 1. Setup: Use a real wavelength (550nm) and a standard droplet
    wavelength_nm = 550.0
    collector = CollectionSphere(mu_bins=np.linspace(1, -1, 100), wavelength_nm=wavelength_nm)

    backend = DrJitRaytracerBackend(grid_res=4000)  # Small grid
    m = complex(1.333, 0.0)
    x = 500.0  # x = 2pi*r/lambda

    # Run the simulation to get wavefronts
    _ = backend.intensity_unpolarized(m, x, np.linspace(1, -1, 10))
    wavefronts = backend._last_wavefronts

    # Let's look at the p=1 wavefront (Direct Transmission)
    rays, patches, name = wavefronts[1]
    hit_mask_p = dr.gather(Bool, rays.Ex.real != 0.0, patches.v0)
    # 2. Extract Phase Data for a few central patches
    # We'll just grab the 0th field of the gathered complex results (representing the first few patches)
    def get_phasor_stats(r_idx):
        # Gather 'l' as before
        l = dr.gather(Float, rays.l, r_idx)

        # FIX: Just use the type of rays.f directly for the gather
        # then cast the result to Float for the math
        f_gathered = dr.gather(type(rays.f), rays.f, r_idx)
        f = Float(f_gathered)

        # Phase = k*l - f*pi/2
        phi = collector.k * l - f * (math.pi / 2.0)

        # Gather Amplitudes
        # Note: If Ex is a Dr.Jit Complex type, you might need to gather
        # real/imag separately if the linter complains
        ex_real = dr.gather(Float, dr.real(rays.Ex), r_idx)
        ex_imag = dr.gather(Float, dr.imag(rays.Ex), r_idx)
        amp = dr.sqrt(ex_real ** 2 + ex_imag ** 2)

        return phi, amp

    phi0, amp0 = get_phasor_stats(patches.v0)

    # 3. Rational Verification
    # A: Amplitude check. For p=1, t_perp * t_para should be roughly 0.98 * 0.98 ~ 0.96
    mean_amp = dr.mean(amp0)
    print(f"Mean Exit Amplitude (p=1): {mean_amp[0]:.4f}")
    assert mean_amp[0] > 0.5, "Energy loss is too high. Transmission logic is cooked."

    # B: Phase Delta check.
    # Neighboring rays in a smooth wavefront should have very similar phases.
    # If the phase delta is > pi, our k-factor or l-units are causing aliasing.
    phi0, amp0 = get_phasor_stats(patches.v0)
    phi1, _ = get_phasor_stats(patches.v1)
    phase_delta = dr.abs(phi1 - phi0)
    valid_phase_delta = dr.select(hit_mask_p, phase_delta, 0.0)
    max_delta = dr.max(phase_delta)
    max_delta_val = dr.max(valid_phase_delta)[0]
    delta_l = dr.abs(dr.gather(Float, rays.l, patches.v1) - dr.gather(Float, rays.l, patches.v0))
    print(f"Distance delta between rays: {delta_l[0]} mm")

    print(f"Max Phase Delta between adjacent rays: {max_delta_val:.4f} rad")

    # Inside your test_phase_coherence
    l_subset = dr.gather(Float, rays.l, dr.arange(dr.uint32_array_t(Float), 10))
    print(f"Sample Path Lengths: {l_subset}")

    # Also check the total range
    l_min = dr.min(rays.l)[0]
    l_max = dr.max(rays.l)[0]
    print(f"Path Range: {l_min:.6f} to {l_max:.6f} mm")
    # If this is way above pi (3.14), we have an aliasing problem.
    # In a 50x50 grid on a droplet, it should be quite small.
    assert max_delta_val < math.pi, "Phase aliasing detected! Adjacent rays are too far apart for the wavelength."

    print("[PASS] Phase coherence verified. Wavefront is smooth and units are aligned.")

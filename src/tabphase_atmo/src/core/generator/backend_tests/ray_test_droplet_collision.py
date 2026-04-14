import drjit as dr
from drjit.auto import Float, Bool, Array3f
import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particles.spherical.sphere import SphericalParticle
from geometric.models.phasor_ray import PhasorRay


def test_sphere_intersection():
    print("\n--- Running Droplet Collision Tests ---")

    droplet = SphericalParticle(radius_mm=1.0)

    rays = dr.zeros(PhasorRay, 3)

    # Ray 0: Dead center
    # Ray 1: Edge Collision
    # Ray 2: Whiffed (x=2.0)
    rays.origin = Array3f(
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        [-10.0, -10.0, -10.0]
    )

    # headings
    rays.direction = Array3f(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    )

    hit_mask, t = droplet.intersect(rays)

    assert hit_mask[0] == True, "Bullseye ray somehow missed the droplet."
    assert dr.allclose(t[0], 9.0), f"Bullseye hit at wrong distance: {t[0]} instead of 9.0"

    # 5. Verify the Graze (Ray 1)
    # Starts at -10, grazes the equator at z=0. Distance must be exactly 10.0 mm.
    assert hit_mask[1] == True, "Grazing ray missed the edge."
    assert dr.allclose(t[1], 10.0), f"Grazing ray hit at wrong distance: {t[1]} instead of 10.0"

    # 6. Verify the Whiff (Ray 2)
    # Fired at x=2.0, sphere ends at x=1.0. Must be a complete miss.
    assert hit_mask[2] == False, "Whiffing ray somehow hit the droplet. Geometry is cooked."

def test_sphere_scattering():
    print("\n--- Running Optics & Scattering Tests ---")

    droplet = SphericalParticle(radius_mm=1.0)
    rays = dr.zeros(PhasorRay, 3)

    # 0: Dead center, 1: Edge graze, 2: Miss
    rays.origin = Array3f([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [-10.0, -10.0, -10.0])
    rays.direction = Array3f([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    hit_mask, t = droplet.intersect(rays)

    # Scatter!
    out = droplet.scatter(rays, hit_mask, t, ior_water=1.333)
    d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, normals = out

    # Verify Bullseye (Ray 0) Fresnel Amplitude
    # n1=1.0, n2=1.333. Reflection ampl = (1 - 1.333)/(1 + 1.333) ≈ -0.1427
    expected_r = (1.0 - 1.333) / (1.0 + 1.333)

    assert dr.allclose(dr.real(r_perp)[0], expected_r, atol=1e-3), "Bullseye perpendicular reflectance is wrong."
    assert dr.allclose(dr.real(r_para)[0], -expected_r, atol=1e-3), "Bullseye parallel reflectance is wrong."

    # Verify Graze (Ray 1) Fresnel Amplitude
    assert dr.allclose(dr.real(r_perp)[1], -1.0, atol=1e-2), "Graze perpendicular reflectance should be -1.0."

    print("\n[PASS] Fresnel Complex Amplitudes matched expected boundaries.")
    print(f"Ray 0 (Center) Reflectance Amplitude: {dr.real(r_perp)[0]:.4f}")
    print(f"Ray 1 (Graze)  Reflectance Amplitude: {dr.real(r_perp)[1]:.4f}")
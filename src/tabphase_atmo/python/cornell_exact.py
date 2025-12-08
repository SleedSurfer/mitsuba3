import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_spectral')

from atmospheric import create_atmospheric_phase


def render_cornell():
    print("--- Rendering Cornell Box (The Rational Fix) ---")

    # Keeping your custom mist, but noting that the original used isotropic.
    # If this looks weird, switch to an isotropic phase function.
    phase_mist = create_atmospheric_phase(
        radius_mean_um=150.0,
        radius_std_um=10.0,
        num_angles=1024, num_wavelengths=32,
        note="cornell_rainbow", force_regen=False
    )

    # Helper to ingest those raw XML matrices without headache
    def T(flat_list):
        return mi.ScalarTransform4f(mi.ScalarMatrix4f(flat_list))

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'volpathmis',
            'max_depth': 64,
        },
        'sensor': {
            'type': 'perspective',
            'fov': 19.5,
            'to_world': T([
                -1, 0, 1.50996e-007, -1.05697e-006,
                0, 1, 0, 1,
                -1.50996e-007, 0, -1, 7,
                0, 0, 0, 1
            ]),
            'sampler': {
                'type': 'independent',
                'sample_count': 256
            },
            'film': {
                'type': 'hdrfilm',
                'width': 1024,
                'height': 1024,
                'pixel_format': 'rgb',
                'rfilter': {'type': 'tent'}
            },
        },

        # --- MATERIALS ---
        'LeftWallBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.63, 0.065, 0.05]}},
        'RightWallBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.091]}},
        'WhiteBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}},
        'LightBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.0, 0.0, 0.0]}},

        'SphereBSDF': {
            'type': 'dielectric',
            'int_ior': 1.5,
            'ext_ior': 1.0
        },

        'Floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 0]).rotate([1, 0, 0], -90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'WhiteBSDF'}
        },
        'Ceiling': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 2, 0]).rotate([1, 0, 0], 90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'WhiteBSDF'}
        },
        'BackWall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 1, -1]).scale(1),
            'bsdf': {'type': 'ref', 'id': 'WhiteBSDF'}
        },
        'RightWall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([1, 1, 0]).rotate([0, 1, 0], -90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'RightWallBSDF'}
        },
        'LeftWall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([-1, 1, 0]).rotate([0, 1, 0], 90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'LeftWallBSDF'}
        },

        # --- SPHERE ---
        'Sphere': {
            'type': 'sphere',
            'radius': 0.3,
            'center': [-0.22827, 1.2, 0.152505],
            'bsdf': {'type': 'ref', 'id': 'SphereBSDF'}
        },

        # --- LIGHT: THE PINPOINT ---
        'Light': {
            'type': 'rectangle',
            'to_world': T([
                -0.0025, -1.91069e-015, 4.37114e-008, -0.005,
                -2.18557e-010, 2.18557e-008, -0.5, 1.98,
                0, -0.002, -8.74228e-011, -0.03,
                0, 0, 0, 1
            ]),
            'bsdf': {'type': 'ref', 'id': 'LightBSDF'},
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [541127, 381972, 127324]
                }
            }
        },

        # --- FOG BOX ---
        'FogBox': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.scale(2.5),
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 1.0,
                'albedo': 0.99,
                'phase': phase_mist
            }
        }
    }

    scene = mi.load_dict(scene_dict)
    img = mi.render(scene)

    #img = dr.clip(img, 0.0, 100.0)

    filename = "cornell_exact_rainbow_fix_sigmat_1.exr"
    mi.util.write_bitmap(filename, img)
    print(f"Saved {filename}. OPEN IN TEV.")


if __name__ == "__main__":
    render_cornell()
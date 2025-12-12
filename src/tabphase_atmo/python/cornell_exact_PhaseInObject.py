import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_spectral')

from atmospheric import create_atmospheric_phase


def render_cornell():
    print("--- Rendering Cornell Box (The Rational Fix) ---")

    # phase_mist = create_atmospheric_phase(
    #     radius_mean_um=10.0,
    #     radius_std_um=2.0,
    #     num_angles=1024, num_wavelengths=32,
    #     note="test", force_regen=False
    # )
    phase_mist = create_atmospheric_phase(
        radius_mean_um=150.0,
        radius_std_um=10.0,
        num_angles=1024, num_wavelengths=32,
        note="cornell_rainbow", force_regen=False
    )

    def T(flat_list):
        return mi.ScalarTransform4f(mi.ScalarMatrix4f(flat_list))

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'volpathmis',
            'max_depth':  64,
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
                'type':  'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'pixel_format':  'rgb',
                'rfilter': {'type': 'tent'}
            },
        },

        # --- MATERIALS ---
        'LeftWallBSDF': {'type': 'diffuse', 'reflectance':  {'type': 'rgb', 'value': [0.63, 0.065, 0.05]}},
        'RightWallBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.091]}},
        'WhiteBSDF':  {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}},

        'Floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 0]).rotate([1, 0, 0], -90).scale(1),
            'bsdf': {'type':  'ref', 'id':  'WhiteBSDF'}
        },
        'Ceiling': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 2, 0]).rotate([1, 0, 0], 90).scale(1),
            'bsdf': {'type':  'ref', 'id':  'WhiteBSDF'}
        },
        'BackWall': {
            'type':  'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 1, -1]).scale(1),
            'bsdf': {'type': 'ref', 'id': 'WhiteBSDF'}
        },
        'RightWall': {
            'type': 'rectangle',
            'to_world':  mi.ScalarTransform4f.translate([1, 1, 0]).rotate([0, 1, 0], -90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'RightWallBSDF'}
        },
        'LeftWall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([-1, 1, 0]).rotate([0, 1, 0], 90).scale(1),
            'bsdf':  {'type': 'ref', 'id': 'LeftWallBSDF'}
        },

        # --- SPHERE ---
        'Sphere': {
            'type': 'sphere',
            'radius': 0.3,
            'center': [-0.22827, 1.2, 0.152505],
            'bsdf': {
                'type': 'dielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0
            },
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 3.0,
                'albedo': 0.99,
                'phase': phase_mist
            }
        },

        # --- LIGHT: Standard Cornell Box ceiling light ---
        'Light': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f. translate([0, 1.99, 0]).rotate([1, 0, 0], 90).scale([0.25, 0.25, 1]),
            'emitter': {
                'type':  'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [18.4, 15.6, 8.0]  # Typical Cornell box light intensity
                }
            }
        },
    }

    scene = mi.load_dict(scene_dict)
    img = mi.render(scene)

    filename = "MistInSphere_test_Baseline.exr"
    mi.util.write_bitmap(filename, img)
    print(f"Saved {filename}. OPEN IN TEV.")


if __name__ == "__main__":
    render_cornell()
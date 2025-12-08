import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_spectral')

from atmospheric import create_atmospheric_phase

def render_cornell():
    print("--- Rendering Cornell Box: ---")

    # Your custom Mie scattering LUT
    '''phase_mist = create_atmospheric_phase(
        radius_mean_um=10.0, radius_std_um=2, num_angles=1024, num_wavelengths=32,
        note="cornell_shafts", force_regen=False
    )'''

    phase_mist = create_atmospheric_phase(
        radius_mean_um=150.0,
        radius_std_um=10.0,
        num_angles=1024, num_wavelengths=32,
        note="cornell_rainbow", force_regen=False
    )

    def T(m):
        return mi.ScalarTransform4f(mi.ScalarMatrix4f(m))

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'volpathmis',
            'max_depth': 16,
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
                # Rainbows require A LOT of samples to resolve cleanly.
                # I'm setting 1024. If it's too slow, try 512.
                # If you want a wallpaper, go 4096.
                'sample_count': 256
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512, # Kept at 512 so you can actually see the definition
                'height': 512,
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
            'type': 'roughdielectric',
            'int_ior': 'bk7', # Standard glass (Dispersion is inherent in Mitsuba spectral)
            'ext_ior': 1.0,
            'alpha': 0.1
        },

        # --- GEOMETRY ---
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
            # Centered it vertically a bit better for the "Caustic Cone"
            'center': [-0.22827, 0.85, 0.152505],
            'bsdf': {'type': 'ref', 'id': 'SphereBSDF'}
        },

        # --- LIGHT: THE PINPOINT ---
        'Light': {
            'type': 'rectangle', # Change from sphere to rectangle
            'to_world': mi.ScalarTransform4f.translate([0, 1.99, 0])
                                            .rotate([1, 0, 0], 90)
                                            .scale([0.5, 0.5, 1.0]),
            'bsdf': {'type': 'ref', 'id': 'LightBSDF'},
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'd65',
                    'scale': 100.0 # You can lower the scale since the area is larger
                }
            }
        },

        # --- FOG BOX ---
        # 'FogBox': {
        #     'type': 'cube',
        #     'to_world': mi.ScalarTransform4f.scale(2.5),
        #     'bsdf': {'type': 'null'},
        #     'interior': {
        #         'type': 'homogeneous',
        #         'sigma_t': 1.0,
        #         'albedo': 1.0,
        #         'phase': phase_mist
        #     }
        # }
    }

    scene = mi.load_dict(scene_dict)
    print("Rendering ...")
    img = mi.render(scene)

    img = dr.clip(img, 0.0, 100.0)

    filename = "cornell_testing_clear.exr"
    mi.util.write_bitmap(filename, img)
    print(f"Saved {filename}. OPEN IN TEV.")

if __name__ == "__main__":
    render_cornell()
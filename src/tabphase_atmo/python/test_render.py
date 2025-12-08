import mitsuba as mi
mi.set_variant("llvm_ad_spectral")

from atmospheric import create_atmospheric_phase
def render_halo_test():
    print("Initializing Scene...")

    # 1. Create the Phase Function Config
    phase_element = create_atmospheric_phase(
        radius_mean_um=10.0,
        radius_std_um=1.5,
        num_angles=1024,
        num_wavelengths=16,
        note="test_halo",
        force_regen=False
    )

    # 2. Build the Scene
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'volpath',
            'max_depth': 32,
            'sample_limit':  500
        },
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 5],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 2048
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'pixel_format': 'rgb',
                # Critical: If you want to use the 'rfilter' (reconstruction filter)
                # to smooth edges, it happens here, but default is fine.
            },
        },
        'foggy_sphere': {
            'type': 'sphere',
            'center': [0, 0, 0],
            'radius': 1.0,
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 2.0,
                'albedo': 0.99,
                'phase': phase_element
            }
        },
        'back_light': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, -3]).scale(2.0),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'uniform', 'value': 6.0}
            }
        }
    }

    scene = mi.load_dict(scene_dict)

    print("Rendering...")
    img = mi.render(scene)

    output_filename = "halo_test.exr"
    print(f"Saving result to '{output_filename}'")
    mi.util.write_bitmap(output_filename, img)



if __name__ == "__main__":
    render_halo_test()
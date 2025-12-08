import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_spectral')
from atmospheric import create_atmospheric_phase


def render_beams_main():
    print("--- Rendering: Forward Scattering (The Money Shot) ---")

    # 1. Force Regen to ensure data is pristine
    phase_mist = create_atmospheric_phase(
        radius_mean_um=10.0, radius_std_um=1.5,
        num_angles=1024, num_wavelengths=16,
        note="godrays",
        force_regen=False  # <--- PARANOIA CHECK
    )

    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': 8},
        'sensor': {
            'type': 'perspective',
            'fov': 40,
            # Camera: At Z=10, looking at Z=0 (The Bars)
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 12],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {'type': 'independent', 'sample_count': 64},  # Fast render
            'film': {'type': 'hdrfilm', 'width': 512, 'height': 512, 'pixel_format': 'rgb'},
        },

        # 1. The Volume
        'fog': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.scale(20.0),
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 0.5,  # Thicker fog
                'albedo': 0.99,
                'phase': phase_mist
            }
        },

        # 2. The Blocker (Black Silhouettes)
        # We make them pure black so only the beams shine
        'Bar1': {'type': 'cube', 'to_world': mi.ScalarTransform4f.translate([-1.5, 0, 0]).scale([0.4, 5, 0.1]),
                 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': 0.0}}},
        'Bar2': {'type': 'cube', 'to_world': mi.ScalarTransform4f.translate([0, 0, 0]).scale([0.4, 5, 0.1]),
                 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': 0.0}}},
        'Bar3': {'type': 'cube', 'to_world': mi.ScalarTransform4f.translate([1.5, 0, 0]).scale([0.4, 5, 0.1]),
                 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': 0.0}}},

        # 3. The Light (Behind the bars, aiming at camera)
        'sun': {
            'type': 'spot',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 2, -5],  # Behind bars
                target=[0, 0, 10],  # Aiming at camera
                up=[0, 1, 0]
            ),
            'intensity': {'type': 'spectrum', 'value': 1000.0},  # Lower intensity is fine for forward scatter
            'cutoff_angle': 60
        }
    }

    scene = mi.load_dict(scene_dict)
    print("Rendering...")
    img = mi.render(scene)

    #img = dr.clip(img, 0.0, 100.0)

    mi.util.write_bitmap("beams_main.exr", img)
    print("Saved beams_main.exr")


if __name__ == "__main__":
    render_beams_main()
import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_spectral')

from atmospheric import create_atmospheric_phase


def render_scene(name, scene_dict, spp=128):
    print(f"--- Rendering: {name} ---")
    scene = mi.load_dict(scene_dict)
    img = mi.render(scene, spp=spp)

    # Auto-exposure for saving (simple mean normalization)
    # This ensures we don't save a pitch black image if the light is far away
    mean_val = dr.mean(img)
    if mean_val > 0:
        img = img / (mean_val * 2.0)  # Scale to mid-grey roughly

    filename = f"{name}.exr"
    mi.util.write_bitmap(filename, img)
    print(f"Saved {filename}")


def run_suite():
    # ---------------------------------------------------------
    # TEST 1: THE RAINBOW (Back-Scattering)
    # ---------------------------------------------------------
    # Physics: Light is BEHIND the camera. We look at the fog.
    # Expectation: A faint colored arc or circle center screen.
    print("Generating LUT for Rainbow (50um droplets)...")
    phase_rainbow = create_atmospheric_phase(
        radius_mean_um=50.0,  # Large drops for spectral separation
        radius_std_um=2.0,  # Narrow distribution for sharper colors
        num_angles=2048,  # High angular res for the bow
        num_wavelengths=32,  # More bins for color accuracy
        note="rainbow_grade"
    )

    scene_rainbow = {
        'type': 'scene',
        'integrator': {'type': 'volpathmis', 'max_depth': 8},
        'sensor': {
            'type': 'perspective',
            'fov': 60,  # Wide FOV to catch the bow (approx 42 deg)
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0],
                target=[0, 0, 1],  # Looking +Z
                up=[0, 1, 0]
            ),
            'sampler': {'type': 'independent', 'sample_count': 256},
            'film': {'type': 'hdrfilm', 'width': 512, 'height': 512, 'pixel_format': 'rgb'},
        },
        # A wall of mist in front of us
        'mist_wall': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 10]).scale([20, 20, 1]),
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 1.0,  # Thinner fog to let light in and out
                'albedo': 0.99,
                'phase': phase_rainbow
            }
        },
        # The Sun (Behind the camera)
        'sun': {
            'type': 'directional',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, -10],  # Behind camera
                target=[0, 0, 1],  # Shining forward (+Z)
                up=[0, 1, 0]
            ),
            'irradiance': {'type': 'uniform', 'value': 20000.0}
        }
    }

    # ---------------------------------------------------------
    # TEST 2: THE SEARCHLIGHT (Side-Scattering)
    # ---------------------------------------------------------
    # Physics: A spotlight cutting through fog perpendicular to view.
    # Expectation: Beam visibility should fall off based on phase function side-scatter lobes.

    # We reuse the 10um "Mist" phase for a softer look
    phase_mist = create_atmospheric_phase(
        radius_mean_um=10.0,
        radius_std_um=1.5,
        num_angles=1024,
        num_wavelengths=16,
        note="beam_grade"
    )

    scene_beam = {
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': 8},
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 10, 0],  # Top-down-ish side view
                target=[0, 0, 0],
                up=[0, 0, 1]
            ),
            'sampler': {'type': 'independent', 'sample_count': 128},
            'film': {'type': 'hdrfilm', 'width': 512, 'height': 512, 'pixel_format': 'rgb'},
        },
        # Large volume filling the scene
        'volume_box': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.scale(5.0),
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 0.5,
                'albedo': 0.99,
                'phase': phase_mist
            }
        },
        # Spotlight aiming horizontally
        'spotlight': {
            'type': 'spot',
            # FIX: Use static look_at with explicit origin and target
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-4, 0, 0],
                target=[4, 0, 0],
                up=[0, 1, 0]
            ),
            'cutoff_angle': 20,
            'beam_width': 15,
            'intensity': {'type': 'uniform', 'value': 500.0}
        }
    }

    # Execute
    render_scene("test_rainbow", scene_rainbow, spp=512)  # Rainbows need samples
    render_scene("test_beam", scene_beam, spp=128)


if __name__ == "__main__":
    run_suite()
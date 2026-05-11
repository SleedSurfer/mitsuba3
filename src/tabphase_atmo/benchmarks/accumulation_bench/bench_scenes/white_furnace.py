import mitsuba as mi
from nimbuscore import create_atmospheric_phase

def get_scene(config, cloud_phase):
    """
    White Furnace Validation Scene.
    The camera stares at a 2-meter sphere of your medium, bathed in 1.0 radiance from all sides.
    """
    # 1. Bake Phase Function
    phase_dict = create_atmospheric_phase(cloud_phase, up_vector=(0.0, 0.0, 1.0), force_regen=False)

    # 2. Camera: Staring straight down the Y axis at the origin
    cam_transform = mi.ScalarTransform4f.look_at(
        origin=[0.0, -5.0, 0.0], # 5 meters back
        target=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0]
    )

    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            # Crank max_depth. If your plugin leaks 1% of energy per scatter,
            # you won't see it clearly at depth 2. At depth 64, it compounds into obvious bullshit.
            "max_depth": 64,
            # Push Russian Roulette out of the way so it doesn't add variance to your debugging
            "rr_depth": 65
        },
        "sensor": {
            "type": "perspective",
            "fov": 35.0,
            "to_world": cam_transform,
            "sampler": {"type": "independent", "sample_count": config.get('batch_size', 1024)},
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 512), # Keep resolution small for fast iteration
                "height": config.get('res_h', 512),
                "pixel_format": "rgb",
            }
        },

        # --- THE FURNACE ---
        # No directional lights. Just an omnidirectional blast of 1.0.
        "env_constant": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}
        },

        # --- THE TEST SUBJECT ---
        "test_sphere": {
            "type": "sphere",
            "center": [0.0, 0.0, 0.0],
            "radius": 2.0, # 2 meters across
            "bsdf": {"type": "null"}, # Absolutely critical: no surface reflection.
            "interior": {
                "type": "homogeneous",
                "sigma_t": 1.0, # High enough to force multiple scattering events
                "albedo": 1.0,  # CRITICAL: 1.0 means zero absorption.
                "phase": phase_dict,
            }
        }
    }
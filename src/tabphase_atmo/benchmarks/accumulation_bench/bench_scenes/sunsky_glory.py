import math
import mitsuba as mi

from nimbuscore import create_atmospheric_phase


def get_scene(config, cloud_phase, sun_elevation_deg=168.0):
    """
    Antisolar 'Glory' setup using the constant envmap + directional sun
    to avoid the 'looking at the ground' black-hole effect.
    """
    # 1. Bake Phase Function
    phase_dict = create_atmospheric_phase(cloud_phase, up_vector=(0.0, 0.0, 1.0), force_regen=False)

    # 2. Camera: Looking directly at the antisolar point
    # At 168 deg elevation, the sun is behind and 'above' the horizon (from a certain POV),
    # so we look 'down' into the slab.
    distance = 50.0
    sun_elev_rad = math.radians(sun_elevation_deg)

    # We calculate the direction the sun is coming FROM
    sun_dir = [
        0.0,
        math.cos(sun_elev_rad),
        math.sin(sun_elev_rad)
    ]

    # Camera is placed along the sun's rays, looking back at the origin
    cam_origin = [0.0, distance * sun_dir[1], distance * sun_dir[2]]

    cam_transform = mi.ScalarTransform4f.look_at(
        origin=cam_origin,
        target=[0.0, 0.0, 0.0],
        up=(0.0, 0.0, 1.0)
    )

    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 2,
            "rr_depth": 5
        },
        "sensor": {
            "type": "perspective",
            "fov": 15.0,  # Tightening for better glory resolution
            "to_world": cam_transform,
            "sampler": {"type": "independent", "sample_count": config.get('batch_size', 1024)},
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 1920),
                "height": config.get('res_h', 1080),
                "pixel_format": "rgb",
            }
        },

        # --- THE "DEFAULT ASS" ENVMAP COMBO ---
        # This keeps the 'ground' from being black
        "env_constant": {
            "type": "constant",
            # Darkened the blue a bit more for that 'teeny smidge' vibe
            "radiance": {"type": "rgb", "value": [0.01, 0.03, 0.12]}
        },
        "sun_directional": {
            "type": "directional",
            "irradiance": {"type": "rgb", "value": [3.0, 3.0, 3.0]},
            "direction": [-sun_dir[0], -sun_dir[1], -sun_dir[2]]  # Sun points TOWARD origin
        },

        # --- THE SLAB ---
        "rain_slab": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate([0, 0, 0]) @
                        mi.ScalarTransform4f.scale([500.0, 20.0, 500.0]),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.02,
                "albedo": 1.0,
                "phase": phase_dict,
            },
        }
    }
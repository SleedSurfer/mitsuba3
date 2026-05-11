import math
import mitsuba as mi

from nimbuscore import create_atmospheric_phase


def get_scene(config, cloud_phase, sun_elevation_deg=168.0):
    """
    Flipped for Corona viewing (Forward Scattering).
    The camera now stares at the sun through the slab.
    """
    # 1. Bake Phase Function (Ensure your phase model supports strong forward peaks!)
    phase_dict = create_atmospheric_phase(cloud_phase, up_vector=(0.0, 0.0, 1.0), force_regen=False)

    # 2. Camera: Looking DIRECTLY AT the sun
    # Sun is at 168 elevation (pointing down-ish). We place cam at origin,
    # looking up the sun's vector.
    sun_elev_rad = math.radians(sun_elevation_deg)
    sun_dir = [0.0, math.cos(sun_elev_rad), math.sin(sun_elev_rad)]

    # Look from origin toward where the sun is coming from
    cam_transform = mi.ScalarTransform4f.look_at(
        origin=[0.0, 0.0, 0.0],
        target=[sun_dir[0], sun_dir[1], sun_dir[2]],
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
            "fov": 15.0, # Corona is SMALL. You need a long lens to see the rings.
            "to_world": cam_transform,
            "sampler": {"type": "independent", "sample_count": config.get('batch_size', 2048)},
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 1920),
                "height": config.get('res_h', 1080),
                "pixel_format": "rgb",
            }
        },

        # # Add this to your scene dict
        # "occluder": {
        #     "type": "sphere",  # A sphere is easier to position than a disk
        #     "center": [sun_dir[0] * 2, sun_dir[1] * 2, sun_dir[2] * 2],  # 2 meters away along the sun line
        #     "radius": 0.08,  # Adjust this to perfectly cover the directional sun's angular size
        #     "bsdf": {
        #         "type": "diffuse",
        #         "reflectance": {"type": "rgb", "value": 0.0}  # Vantablack mode
        #     }
        # },

        # --- THE DARKER SKY COMBO ---
        "env_constant": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.005, 0.01, 0.05]} # Darker for contrast
        },
        "sun_directional": {
            "type": "directional",
            "irradiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}, # Bright sun to pierce the slab
            "direction": [-sun_dir[0], -sun_dir[1], -sun_dir[2]]
        },

        # --- THE SLAB (Positioned in front of cam) ---
        "rain_slab": {
            "type": "cube",
            # We use look_at to rotate the slab so its flat face is perfectly perpendicular to the camera ray.
            # Origin is the center of the slab (pushed 30 meters out to clear the camera).
            # Target is the camera (origin).
            "to_world": mi.ScalarTransform4f.look_at(
                            origin=[30.0 * sun_dir[0], 30.0 * sun_dir[1], 30.0 * sun_dir[2]],
                            target=[0.0, 0.0, 0.0],
                            up=[0.0, 0.0, 1.0]
                        ) @ mi.ScalarTransform4f.scale([100.0, 100.0, 5.0]), # Z is now the thin axis pointing at us
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.005,
                "albedo": 0.99,
                "phase": phase_dict,
            }
        }
    }
import math
import mitsuba as mi
from ..utils import get_asset_path, generate_cloud_grid
from src.core.wrapper import create_atmospheric_phase


def get_scene(config, phase):
    # --- 1. MATCH YOUR BAKE ELEVATION ---
    sun_elevation_deg = 20.0
    sun_elev_rad = math.radians(sun_elevation_deg)

    # --- 2. Z-UP SUN VECTOR ---
    # Elevation is the angle above the XY horizon.
    sun_dir_y = math.cos(sun_elev_rad)
    sun_dir_z = math.sin(sun_elev_rad)
    sun_direction = [0.0, sun_dir_y, sun_dir_z]

    # --- 3. THE "GROUND" OBSERVER CAMERA ---
    # We place the camera beneath the cloud slab (negative Z)
    # and look directly along the sun vector to keep it dead center.
    cam_origin = [
        -5.0 * sun_direction[0],
        -5.0 * sun_direction[1],
        -5.0 * sun_direction[2]
    ]
    cam_target = [0.0, 0.0, 0.0]

    # --- 4. Z-UP PHASE FUNCTION ---
    # We MUST tell your phase function generator that gravity is now on the Z-axis,
    # otherwise your plates will be oriented sideways relative to the sky dome.
    phase_dict = create_atmospheric_phase(phase, up_vector=(0.0, 0.0, 1.0), force_regen=True)

    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 3,  # Bumped from 2 to 3 for juicier multi-scattering
        },
        "sensor": {
            "type": "perspective",
            "fov": 100.0,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=cam_origin,
                target=cam_target,
                up=(0.0, 0.0, 1.0),  # Camera up is now global Z
            ),
            "sampler": {"type": "independent", "sample_count": config.get('batch_size', 1024)},
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 1024),
                "height": config.get('res_h', 1024),
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            }
        },

        # --- NATIVELY Z-UP SUNSKY ---
        "sky": {
            "type": "sunsky",
            "sun_direction": sun_direction,
            "turbidity": 2.5,  # 2.5 = crisp high-altitude air.
            "sun_scale": 1.0,  # Tweak this if the sun burns out your halos
            "sky_scale": 1.0,
        },

        "cloud_slab": {
            "type": "cube",
            # In a Z-up world, this makes a wide horizontal plate in the sky (X, Y)
            # that is thin vertically (Z).
            "to_world": mi.ScalarTransform4f.scale([30.0, 30.0, 0.5]),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.03,
                "albedo": 0.98,
                "phase": phase_dict,
            },
        },
    }
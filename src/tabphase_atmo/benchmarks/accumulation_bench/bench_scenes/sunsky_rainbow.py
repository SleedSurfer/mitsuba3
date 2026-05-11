import math
import mitsuba as mi
import time

from ..utils import (
    get_asset_path,
    create_sunsky_emitter,
    generate_cumulus_grid,
    assemble_volume_dict,
    create_sun_aligned_camera,
    visualize_grid_seeds
)
from nimbuscore.core.gen_manager import create_atmospheric_phase
from nimbuscore.core.config import *


def create_anti_sun_camera(sun_elevation_deg, distance=5.0):
    """
    Puts the camera in front of the origin, looking AT the origin,
    with the sun shining from behind the camera. (Rainbow viewing mode!)
    """
    sun_elev_rad = math.radians(sun_elevation_deg)
    sun_dir_y = math.cos(sun_elev_rad)
    sun_dir_z = math.sin(sun_elev_rad)

    cam_origin = [
        0.0,
        distance * sun_dir_y,  # Positive distance instead of negative!
        distance * sun_dir_z
    ]

    return mi.ScalarTransform4f.look_at(
        origin=cam_origin,
        target=[0.0, 0.0, 0.0],
        up=(0.0, 0.0, 1.0)
    )


def get_scene(config, cloud_phase,isbrute = False, sun_elevation_deg=130.0):
    # 1. Bake BOTH phase functions
    print("Baking Cloud Phase...")
    start_time = time.perf_counter()
    phase_dict = create_atmospheric_phase(cloud_phase, up_vector=(0.0, 0.0, 1.0), force_regen=True,masochist_mode=isbrute,threshold=1.0)
    end_time = time.perf_counter()
    print(f"Cloud Phase baked in {end_time - start_time:.4f} seconds..")
    base_w = config.get('res_w', 3840)
    base_h = config.get('res_h', 2160)

    # # Anchor point shifted left to catch the apex, starting near the absolute top
    # offset_x = int(base_w * 0.45)  # 35% across
    # offset_y = int(base_h * 0.25)  # 2% down
    #
    # # Bounding box stretches to the right edge and drops deep below the 50% horizon
    # crop_w = int(base_w * 0.35)  # Covers the remaining 65% of the width
    # crop_h = int(base_h * 0.35)  # Pulls the bottom edge down to 80% of the total frame height
    offset_x = int(base_w * 0.55)
    offset_y = int(base_h * 0.28)

    # Keep the same bounding box size unless you want it to scale with the shift
    crop_w = int(base_w * 0.35)
    crop_h = int(base_h * 0.35)
    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 2,
            "rr_depth": 5
        },
        "sensor": {
            "type": "perspective",
            "fov": 90.0,
            "to_world": create_sun_aligned_camera(20, distance=50.0)@mi.ScalarTransform4f.translate([0.0, 0.0, 0.0]),
            "sampler": {"type": "independent", "sample_count": config.get('batch_size', 1024)},
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 1024),
                "height": config.get('res_h', 1024),
                "pixel_format": "rgb",
                # Injected dynamic crop parameters
                "crop_width": crop_w,
                "crop_height": crop_h,
                "crop_offset_x": offset_x,
                "crop_offset_y": offset_y,
            }
        },


        # The open sky environment
        "sky": create_sunsky_emitter(
            sun_elevation_deg=168, #148
            use_direct_vector=True,
            turbidity=2.5,
            sun_scale=4
        ),


        "rain_slab": {
            "type": "cube",
            # Scale Z by 10 (20m tall bounding box) and lift by 10.
            # Sits perfectly from Z=0 to Z=20. Safe distance from the cloud!
            "to_world": mi.ScalarTransform4f.translate([0, 0, 0]) @ mi.ScalarTransform4f.scale(
                [500.0, 10.0, 500.0]),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.02,
                "albedo": 1.0,
                "phase": phase_dict,
            },
        }
    }
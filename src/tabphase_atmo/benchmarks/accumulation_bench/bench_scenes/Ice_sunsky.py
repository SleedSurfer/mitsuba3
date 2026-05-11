import mitsuba as mi
import time
from ..utils import (
    get_asset_path,
    create_sunsky_emitter,
    generate_cirrus_grid,
    assemble_volume_dict,
    create_sun_aligned_camera
)
from nimbuscore.core.gen_manager import create_atmospheric_phase


def get_scene(config, phase, sun_elevation_deg=20.0):
    # 1. Bake the phase function
    start_time = time.perf_counter()
    phase_dict = create_atmospheric_phase(phase, up_vector=(0.0, 0.0, 1.0), force_regen=True)
    end_time = time.perf_counter()
    print(f"Cloud Phase baked in {end_time - start_time:.4f} seconds..")

    # 2. Bake the physical cloud
    raw_grid = generate_cirrus_grid(
        res=128,
        density_multiplier=0.02,
        wind_stretch=16.0
    )
    cloud_volume_dict = assemble_volume_dict(
        density_grid=raw_grid,
        phase_dict=phase_dict,
        bounds=(2000.0, 2000.0, 5.0)
    )

    # 3. Assemble the scene payload
    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 2,
        },
        "my_atmo_medium": {
            "type": "homogeneous",
            "sigma_t": 0.004,
            "albedo": 0.98,
            "phase": phase_dict
        },
        "sensor": {
            "type": "perspective",
            "fov":100,
            "medium": {
                "type": "ref",
                "id": "my_atmo_medium"
            },
            "to_world": create_sun_aligned_camera(sun_elevation_deg, distance=5.0),
            "sampler": {"type": "independent", "sample_count": config.get('batch_size', 1024)},
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 1024),
                "height": config.get('res_h', 1024),
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            }
        },

        "sky": create_sunsky_emitter(
            sun_elevation_deg=sun_elevation_deg,
            use_direct_vector=True,
            turbidity=2.5,
            sun_scale=2.5

        ),

        "atmosphere_boundary": {
            "type": "sphere",
            "radius": 100.0,  # 5km radius should be plenty
            "to_world": mi.ScalarTransform4f.translate([0, 0, 0]),  # Center it exactly on the camera
            "bsdf": {"type": "null"},  # Perfectly transparent
            "interior": {
                "type": "ref",
                "id": "my_atmo_medium"
            }
        },
        #"cloud_volume": cloud_volume_dict,
    }
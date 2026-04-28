import mitsuba as mi
from ..utils import get_asset_path, generate_cloud_grid
from src.core.wrapper import create_atmospheric_phase


def get_scene(config,phase):

    light_mod = 0.5
    # --- PHASE FUNCTION ---
    phase_dict = create_atmospheric_phase(phase,up_vector=(0.0, 1.0, 0.0),force_regen=False)

    sun_direction = [0, 0, -1]

    # return {
    #     "type": "scene",
    #     "integrator": {
    #         "type": "volpath",
    #         "max_depth": 2,
    #     },
    #     "sensor": {
    #         "type": "perspective",
    #         "fov": 110.0,
    #         "to_world": mi.ScalarTransform4f.look_at(
    #             origin=(0, 0, 3),
    #             target=(0, 0, 0),
    #             up=(0, 1, 0),
    #         ),
    #         "sampler": {"type": "independent", "sample_count": config.get('batch_size', 1024)},
    #         "film": {
    #             "type": "hdrfilm",
    #             "width": config.get('res_w', 1024),
    #             "height": config.get('res_h', 1024),
    #             "pixel_format": "rgb",
    #             "rfilter": {"type": "box"}
    #         }
    #     },
    #
    #     "rect_light": {
    #         "type": "rectangle",
    #         # Placed at Z=5 (behind the camera at Z=3), aiming dead center at the clouds
    #         "to_world": mi.ScalarTransform4f.look_at(
    #             origin=(0, 0, 5),
    #             target=(0, 0, 0),
    #             up=(0, 1, 0)
    #         ),
    #         "bsdf": {"type": "null"},
    #         "emitter": {
    #             "type": "area",
    #             "radiance": {"type": "rgb", "value": [50.0, 50.0, 50.0]},
    #         },
    #     },
    #
    #     "cloud_slab": {
    #         "type": "cube",
    #         "to_world": mi.ScalarTransform4f.scale([15.0, 15.0, 0.5]),
    #         "bsdf": {"type": "null"},
    #         "interior": {
    #             "type": "homogeneous",
    #             "sigma_t": 0.03,
    #             "albedo": 0.98,
    #             "phase": phase_dict,
    #         },
    #     },
    # }
    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 2
        },
        "sensor": {
            "type": "perspective",
            "fov": 110.0,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3),
                target=(0, 0, 0),
                up=(0, 1, 0),
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

        "sun": {
            "type": "directional",
            "direction": sun_direction,
            "irradiance": {"type": "rgb", "value": [50.0,50.0,50.0]},
            # "irradiance": {
            #     "type": "spectrum",
            #     "value": [
            #         (380.0, 1.00),
            #         (400.0, 1.00),  # violet
            #         (430.0, 1.0),
            #         (450.0, 1.0),
            #     ],
            # },
        },

        "cloud_slab": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.scale([15.0, 15.0, 0.5]),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.03,
                "albedo": 0.98,
                "phase": phase_dict,
            },
        },
    }
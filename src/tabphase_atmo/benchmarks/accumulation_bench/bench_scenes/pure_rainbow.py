import mitsuba as mi
from ..utils import get_asset_path, get_phase_plugin
import mitsuba as mi
from ..utils import get_asset_path, get_phase_plugin, generate_cloud_grid


def get_scene(config):

    light_mod = 1.0

    # --- PHASE FUNCTION ---

    phase_dict = get_phase_plugin(
        radius_mean=1800,
        radius_std=800,
        num_angles=8192,
        num_wavelengths=64,
        note="rain_test"
    )

    sun_direction = [0, 0, -1]

    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": -1,
        },
        "sensor": {
            "type": "perspective",
            "fov": 110.0,  # Wide FOV to catch the 42° rainbow radius
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
        # Directional light (The Sun)
        "sun": {
            "type": "directional",
            "direction": sun_direction,
            "irradiance": {"type": "rgb", "value": 50.0*light_mod},
        },
        # The Cloud Slab
        "cloud_slab": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.scale([15.0, 15.0, 0.5]),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.1,
                "albedo": 0.98,
                "phase": phase_dict,
            },
        },
    }
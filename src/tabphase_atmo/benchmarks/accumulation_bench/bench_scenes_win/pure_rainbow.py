import mitsuba as mi

from nimbuscore.core import create_atmospheric_phase
from ..utils import generate_cloud_grid

def get_scene(config):
    phase_dict = create_atmospheric_phase(
        radius_mean_um=15.0,
        radius_std_um=5.0,
        num_angles=4096,
        num_wavelengths=64,
        note="mie_cloud",
    )

    # Base transform for the physical grid space
    base_transform = mi.ScalarTransform4f.translate([-4, -3, 5]).scale(5.0)

    cloud_medium = {
        "type": "heterogeneous",
        "albedo": 0.98,
        "phase": phase_dict,
        "sigma_t": {
            "type": "gridvolume",
            "grid": generate_cloud_grid(res=64, density_multiplier=15.0, seed=123914),
            "to_world": base_transform,
            "filter_type": "trilinear"
        }
    }

    scene_dict = {
        "type": "scene",

        # 1. The Engine
        "integrator": {
            "type": "volpath",
            "max_depth": 16#-1,
            #"rr_depth": 99999
        },

        # 2. The Camera (Looking East)
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 15, 2],
                target=[0, 0, 5],
                up=[0, 0, 1]
            ),
            "sampler": {
                "type": "independent",
                "sample_count": config['batch_size']
            },
            "film": {
                "type": "hdrfilm",
                "width": config['res_w'],
                "height": config['res_h'],
                "pixel_format": "rgb",
            }
        },

        # 3. The Lighting (ACTUALLY Golden Hour now)
        "emitter": {
            "type": "sunsky",
            "latitude": 49.1951,
            "longitude": 16.6068,
            "timezone": 1.0,
            "month": 5,
            "day": 23,
            "hour": 6.5,           # 4:30 PM (Sun sets in the West, behind camera)
       },

        # 4. The Floor
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.scale(100.0),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.15, 0.15, 0.15]
                }
            }
        },

        # 5. The Cloud Container (Geometry Fixed)
        "cloud_bbox": {
            "type": "cube",
            "to_world": base_transform @ mi.ScalarTransform4f.translate([0.5, 0.5, 0.5]).scale(0.5),
            "interior": cloud_medium,
            "bsdf": {"type": "null"}
        }
    }

    return scene_dict
import mitsuba as mi
import numpy as np
from nimbuscore.core.config import Particle, DropletComposition, DropletShape, BackendType
from nimbuscore import create_atmospheric_phase

MESH_DIR = "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/scenes/mars"

def get_exact_blender_camera(loc, rot_deg):
    rx, ry, rz = np.radians(rot_deg)
    mat_x = np.array([[1, 0, 0, 0], [0, np.cos(rx), -np.sin(rx), 0], [0, np.sin(rx), np.cos(rx), 0], [0, 0, 0, 1]])
    mat_y = np.array([[np.cos(ry), 0, np.sin(ry), 0], [0, 1, 0, 0], [-np.sin(ry), 0, np.cos(ry), 0], [0, 0, 0, 1]])
    mat_z = np.array([[np.cos(rz), -np.sin(rz), 0, 0], [np.sin(rz), np.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    rot_mat = mat_z @ mat_y @ mat_x
    blender_matrix = rot_mat.copy()
    blender_matrix[:3, 3] = loc

    # Coordinate fixes for Blender -> Mitsuba
    init_rot = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    basis_change = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    return mi.ScalarTransform4f(basis_change @ (blender_matrix @ init_rot))

def get_scene(config, phase=None, sun_elevation_deg=20.0):
    # Camera transforms from your screenshot
    loc = [-26.925, -9.4475, 6.4279]
    rot = [-65,180,-40]

    pdict = create_atmospheric_phase(phase, (0, 1, 0), force_regen=True, threshold=1.0)

    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 32,  # Note: 2 means only single-scattering. Bump to 4-6 if you want realistic cloud glow
            "rr_depth": 10
        },

        # --- 1. DEFINE GLOBAL MEDIUM ---
        "my_atmo_medium": {
            "type": "homogeneous",
            "sigma_t": 0.004,
            "albedo": 0.98,
            "phase": pdict
        },

        "sensor": {
            "type": "perspective",
            "focal_length": "16mm",
            "to_world": get_exact_blender_camera(loc, rot),
            "sampler": {
                "type": "independent",
                "sample_count": config.get('batch_size', 4096)
            },
            # Tell the camera it is starting INSIDE the medium
            "medium": {
                "type": "ref",
                "id": "my_atmo_medium"
            },
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 2160),
                "height": config.get('res_h', 3240),
                "pixel_format": "rgb",
                "component_format": "float32"
            }
        },

        # --- 2. THE ATMOSPHERE BOUNDARY ---
        # A massive invisible sphere that holds the medium
        "atmosphere_boundary": {
            "type": "sphere",
            "radius": 100.0,  # 5km radius should be plenty
            "to_world": mi.ScalarTransform4f.translate(loc),  # Center it exactly on the camera
            "bsdf": {"type": "null"},  # Perfectly transparent
            "interior": {
                "type": "ref",
                "id": "my_atmo_medium"
            }
        },

        # --- EMITTERS ---
        "envmap": {
            "type": "envmap",
            "filename": "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/envmaps/passendorf_snow_4k.exr",
            "to_world": mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=0.0)
        }
    }

    return scene_dict
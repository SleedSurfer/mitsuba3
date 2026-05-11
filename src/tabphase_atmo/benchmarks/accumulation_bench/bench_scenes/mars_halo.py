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
    rot = [90.722, 0.53753, -82.437]

    pdict = create_atmospheric_phase(phase, (0, 1, 0), force_regen=False,threshold=1.0)

    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 32,
            "rr_depth": 10
        },
        "sensor": {
            "type": "perspective",
            "fov_axis": "smaller",
            "fov": 37.5,
            "to_world": get_exact_blender_camera(loc, rot),
            "sampler": {
                "type": "independent",
                "sample_count": config.get('batch_size', 4096)
            },
            "medium": {
                "type": "homogeneous",
                "sigma_t": 0.0006,
                "albedo": 0.78,
                "phase": pdict
            },
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 2160),
                "height": config.get('res_h', 3240),
                "pixel_format": "rgb",
                "component_format": "float32",
                #"rfilter": {"type": "box"}
            }
        },

        # --- MATERIALS ---
        "mars_pbr": {
            "type": "twosided",
            "bsdf": {
                "type": "normalmap",
                "normalmap": {
                    "type": "bitmap",
                    "filename": f"{MESH_DIR}/textures/mars_normal.png",
                    "raw": True
                },
                "bsdf": {
                    "type": "principled",
                    "base_color": {
                        "type": "bitmap",
                        "filename": f"{MESH_DIR}/textures/Mars_color.png"
                    },
                    "roughness": {
                        "type": "bitmap",
                        "filename": f"{MESH_DIR}/textures/mars_roughness.png",
                        "raw": True
                    },
                    "specular": 0.0,
                    "metallic": 0.0
                }
            }
        },

        # --- EMITTERS ---
        "sky_ambient": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [x*3 for x in[0.776, 0.482, 0.361]]}
        },
        "sun": {
            "type": "directional",
            # The exact inverse of the disk's origin coordinates
            # Vector from [500, 100, -250] to [0, 0, 0] -> [-500, -100, 250]
            # Normalized down so it's cleaner:
            "direction": [-1.0, -0.2, 0.5],
            "irradiance": {
                "type": "rgb",
                "value": [0.06*x for x in [12.0, 11.5, 9.5]]
            }
        },
        "sun_disk": {
            "type": "disk",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[500, 100, -250],  # Push it way out into the "sky"
                target=[0, 0, 0],
                up=[0, 1, 0]
            ).scale(2.0),  # Small scale = sharp sun
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [500, 480, 400]}
            }
        },
        # --- BASE GEOMETRY ---
        "mars_surface": {
            "type": "ply",
            "filename": f"{MESH_DIR}/meshes/Plane.ply",
            "bsdf": {"type": "ref", "id": "mars_pbr"}
        },
        # "crystals":{
        #     "type": "sphere",
        #     "radius": 600.0,
        #     "to_world": mi.ScalarTransform4f.translate([0.0, 0.0, 0.0]),
        #     "bsdf": {"type": "null"},
        #     "interior": {
        #         "type": "homogeneous",
        #         "sigma_t": 3.0,
        #         "albedo": 1.0,
        #     }
        # }
    }

    # --- DYNAMIC ROCK LOADING ---
    for i in range(11):
        name = "Icosphere" if i == 0 else f"Icosphere_{i:03d}"
        scene_dict[f"obj_{name}"] = {
            "type": "ply",
            "filename": f"{MESH_DIR}/meshes/{name}.ply",
            "bsdf": {"type": "ref", "id": "mars_pbr"}
        }

    return scene_dict
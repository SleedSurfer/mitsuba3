import mitsuba as mi
import pyopenvdb as vdb
import numpy as np


def get_scene(config):
    vdb_path = "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/scenes/wdas_cloud/wdas_cloud_eighth.vdb"

    # ==========================================
    # --- THE INLINE VDB MAGIC YOU ASKED FOR ---
    # ==========================================
    # 1. Read the density grid from the OpenVDB file natively
    grid = vdb.read(vdb_path, gridname='density')

    # 2. Find the exact bounding box of the active voxels
    v0, v1 = grid.evalActiveVoxelBoundingBox()
    ijk0 = np.array(v0, dtype=int)
    ijk1 = np.array(v1, dtype=int)

    # 3. Calculate grid size [X, Y, Z]
    size = ijk1 - ijk0 + 1

    # 4. Spin up an empty numpy array and dump the VDB data straight into it
    dense_array = np.zeros(size, dtype=np.float32)
    grid.copyToArray(dense_array, ijk=ijk0)

    # 5. Mitsuba expects a contiguous tensor of shape [Z, Y, X, Channels]
    # We transpose the OpenVDB [X, Y, Z] layout, add a channel dimension, and lock it in memory
    mi_array = np.ascontiguousarray(dense_array.transpose(2, 1, 0)[..., np.newaxis])

    # 6. Instantiate the Mitsuba VolumeGrid object right here in memory
    mi_cloud_grid = mi.VolumeGrid(mi.TensorXf(mi_array))
    # ==========================================

    # --- TRANSFORMS ---
    # mi.ScalarTransform4f chaining applies operations sequentially.
    cam_transform = mi.ScalarTransform4f.look_at(
        origin=[648.064, -82.473, -63.856],
        target=[6.021, 100.043, -43.679],
        up=[0.273, 0.962, -0.009]
    )

    disk_transform = mi.ScalarTransform4f.scale([20000.0, 20000.0, 20000.0]) \
        .rotate([1, 0, 0], -90.0) \
        .translate([0.0, -1000.0, 0.0])

    cube_transform = mi.ScalarTransform4f.scale([206.544, 140.4, 254.592]) \
        .translate([-9.984, 73.008, -42.64])

    # --- SCENE DICT ---
    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 8,
        },

        "sensor": {
            "type": "perspective",
            "fov_axis": "x",
            "fov": 54.43,
            "near_clip": 1.0,
            "far_clip": 1000000.0,
            "to_world": cam_transform,
            "sampler": {
                "type": "independent",
                "sample_count": config.get('batch_size', 1024)
            },
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 333),
                "height": config.get('res_h', 180),
                "pixel_format": "rgb",
            }
        },

        # --- EMITTERS ---
        "env_constant": {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": [0.03, 0.07, 0.23]
            }
        },
        "sun_directional": {
            "type": "directional",
            "irradiance": {
                "type": "rgb",
                "value": [2.6, 2.5, 2.3]
            },
            "direction": [-0.5826, -0.7660, -0.2717]
        },

        # --- MEDIUM ---
        "cloud_medium": {
            "type": "heterogeneous",
            "sigma_t": {
                "type": "gridvolume",
                "grid": mi_cloud_grid,  # <--- Passing the in-memory array object here
                "scale": 4.0
            },
            "albedo": {
                "type": "rgb",
                "value": [1.0, 1.0, 1.0]
            },
            "phase": {
                "type": "hg",
                "g": 0.877
            }
        },

        # --- SHAPES ---
        "ground_disk": {
            "type": "disk",
            "to_world": disk_transform,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.2, 0.2, 0.2]
                }
            }
        },

        "cloud_boundary": {
            "type": "cube",
            "to_world": cube_transform,
            "interior": {
                "type": "ref",
                "id": "cloud_medium"
            }
        }
    }

    return scene_dict
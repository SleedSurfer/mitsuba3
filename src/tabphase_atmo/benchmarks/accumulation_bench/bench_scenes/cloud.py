import mitsuba as mi
import numpy as np

import mitsuba as mi

from nimbuscore import create_atmospheric_phase


def get_scene(config,phase):
    # Swap this to wherever your 300MB champion lives
    cloud_path = "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/volumes/wdas_cloud_half.vol"
    cloud_phase = create_atmospheric_phase(phase,up_vector=(0.0, 1.0, 0.0),force_regen=True,threshold=1.0)
    # ==========================================
    # ☁️ CLOUD MASTER CONTROLS (Disney Values)
    # ==========================================
    # The XML translation coordinates
    cloud_pos = [-9.984, 73.008, -42.64]

    # The XML scaled a [-1, 1] cube by these values.
    # Since a standard cube is 2 units wide (-1 to 1), the total real-world
    # width/height/depth in meters is double the scale parameter.
    cloud_scale = [206.544 * 2.0, 140.4 * 2.0, 254.592 * 2.0]

    # ==========================================
    # 🏗️ THE TRANSFORM STACK (Reads Right-to-Left)
    # ==========================================
    # 1. CENTER: Move the [0, 1] grid so its center [0.5, 0.5, 0.5] sits at the origin
    center_tx = mi.ScalarTransform4f.translate([-0.5, -0.5, -0.5])

    # 2. SCALE: Blow it up to the massive real-world size in meters
    scale_tx = mi.ScalarTransform4f.scale(cloud_scale)

    # 3. POSITION: Move it to the exact XML location
    move_tx = mi.ScalarTransform4f.translate(cloud_pos)

    # 🔗 MULTIPLY: Applies Center -> Scale -> Move
    grid_transform = move_tx @ scale_tx @ center_tx

    # 📦 THE INVISIBLE BOX
    # Shrinks the [-1,1] cube to [0,1], then perfectly applies the grid's location
    box_transform = grid_transform @ mi.ScalarTransform4f.translate([0.5, 0.5, 0.5]) @ mi.ScalarTransform4f.scale(
        [0.5, 0.5, 0.5])

    # --- CAMERA & SUN ---
    cam_transform = mi.ScalarTransform4f.look_at(
        origin=[648.064, -82.473, -63.856],
        target=[6.021, 100.043, -43.679],
        up=[0.273, 0.962, -0.009]
    )

    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": -1,  # Let it bounce, the cloud is thick
            "rr_depth": 10  # Start killing weak rays early to save your CPU
        },

        "sensor": {
            "type": "perspective",
            "fov_axis": "x",
            "fov": 54.43,
            "to_world": cam_transform,
            "sampler": {
                "type": "independent",
                "sample_count": config.get('batch_size', 256)
            },
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 640),
                "height": config.get('res_h', 360),
                "pixel_format": "rgb",
                "rfilter": {"type": "box"},
            }
        },

        # --- EMITTERS ---
        "env_constant": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.03, 0.07, 0.23]}
        },
        "sun_directional": {
            "type": "directional",
            "irradiance": {"type": "rgb", "value": [2.6, 2.5, 2.3]},
            "direction": [-0.5826, -0.7660, -0.2717]
        },

        # --- THE CLOUD SHAPE ---
        "cloud_boundary": {
            "type": "cube",
            "to_world": box_transform,  # The fixed box bounds
            "bsdf": {"type": "null"},  # <--- THE MAGIC FIX: Make the box invisible
            "interior": {
                "type": "heterogeneous",
                "scale": 4.0,  # The master density multiplier from the XML
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": cloud_path,
                    "to_world": grid_transform  # The fixed grid mapping
                },
                "albedo": {
                    "type": "rgb",
                    "value": [1.0, 1.0, 1.0]
                },
                "phase": cloud_phase

            }
        }
    }

    return scene_dict
import mitsuba as mi
from ..utils import get_asset_path, get_phase_plugin, generate_cloud_grid


def get_scene(config):
    scene_name = "Spectre"
    mesh_dir = get_asset_path(scene_name, "meshes")

    light_mod = 40.0

    # --- PHASE FUNCTION ---
    # Spectre/Glory requires specific droplet sizes.
    # We stick to the 3.95um you had active, but 6.0um usually makes better rainbows.
    phase_dict = get_phase_plugin(
        radius_mean=3.95,
        radius_std=0.03,
        note="spectre_mist"
    )

    # --- CLOUD VOLUME ---
    grid_transform = mi.ScalarTransform4f.translate([-5, -5, -5]).scale(10.0)

    cloud_medium = {
        "type": "heterogeneous",
        "albedo": 0.98,
        "phase": phase_dict,
        "sigma_t": {
            "type": "gridvolume",
            "grid": generate_cloud_grid(),
            "to_world": grid_transform,
            "filter_type": "trilinear"
        }
    }

    # --- TRANSFORMS ---
    cam_transform = mi.ScalarTransform4f() \
        .rotate([1, 0, 0], 2.5044780654876655e-06) \
        .rotate([0, 1, 0], -5.008956130975331e-06) \
        .rotate([0, 0, 1], -5.008956130975331e-06) \
        .translate([-0.151983, 0.596070, -1.730589])

    rect_matrix_flat = [
        0.000000, -0.019629, -0.000000, -0.151983,
        -0.019629, 0.000000, -0.000000, 0.596070,
        0.000000, 0.000000, -0.019629, -22.928349,
        0.000000, 0.000000, 0.000000, 1.000000
    ]
    rect_transform = mi.ScalarTransform4f([rect_matrix_flat[i:i + 4] for i in range(0, 16, 4)]).scale(0.1)

    # --- SCENE DICT ---
    return {
        "type": "scene",
        "sub_global_mist": cloud_medium,
        "integrator": {
            "type": "volpathmis",
            "max_depth": -1,
        },
        "sensor": {
            "type": "perspective",
            "fov_axis": "x",
            "fov": 39.597755,
            "near_clip": 0.1,
            "far_clip": 1000.0,
            "medium": {"type": "ref", "id": "sub_global_mist"},
            "to_world": cam_transform,
            "sampler": {"type": "ldsampler", "sample_count": config['batch_size']},
            "film": {
                "type": "hdrfilm",
                "width": config['res_w'],
                "height": config['res_h'],
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            }
        },

        "default-bsdf": {"type": "twosided", "bsdf": {"type": "diffuse"}},

        # Light Source
        "elm__3": {
            "type": "rectangle",
            "flip_normals": True,
            "to_world": rect_transform,
            "emitter": {
                "type": "area",
                "radiance": {"type": "spectrum", "value": 1416666.0 * light_mod}
            },
            "bsdf": {"type": "null"}
        },

        # Meshes
        "elm__2": {
            "type": "ply",
            "filename": str(mesh_dir / "sbunny.ply"),
            "face_normals": True,
            "bsdf": {"type": "ref", "id": "default-bsdf"}
        },
        "elm__4": {
            "type": "ply",
            "filename": str(mesh_dir / "Plane.ply"),
            "face_normals": True,
            "bsdf": {"type": "null"},
        }
    }
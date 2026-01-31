import mitsuba as mi
from ..utils import (
    get_asset_path, get_phase_plugin, get_exact_blender_camera,
    make_disk_transform, generate_cloud_grid
)


def get_scene(config):
    scene_name = "RoadLampsXML"
    mesh_dir = get_asset_path(scene_name, "meshes")
    tex_dir = get_asset_path(scene_name, "textures")

    light_mod = 0.6

    # --- PHASE & ATMOSPHERE ---
    # Specific params from your code
    phase_dict = get_phase_plugin(
        radius_mean=3.95, radius_std=0.03, note="glory_395_003"
    )

    grid_transform = mi.ScalarTransform4f.translate([-100, -100, -100]).scale(200.0)
    global_fog = {
        "type": "heterogeneous",
        "albedo": 0.75,
        "sigma_t": {
            "type": "gridvolume",
            "grid": generate_cloud_grid(),
            "to_world": grid_transform,
            "filter_type": "trilinear"
        },
        "phase": phase_dict
    }

    # --- SPECTRUM (Dirty Green) ---
    mercury_data = [
        (365.0, 0.1), (404.7, 2.0), (435.8, 3.0),  # Crushed Blue
        (546.1, 28.0),  # Supercharged Green
        (577.0, 6.0), (579.1, 6.0), (600.0, 0.5), (700.0, 0.0)
    ]
    mercury_list = [(w, v * 500.0 * light_mod) for w, v in mercury_data]

    # --- SCENE GRAPH ---
    cam_transform = get_exact_blender_camera(
        loc=[-19.767, 11.002, 44.227],
        rot_deg=[98.737, 0.68787, 57.42]
    )

    # Matrices from your dump
    matrices = {
        "elm__12": [0.057929, 0.0, 0.0, -88.164650, 0.0, -0.004041, 0.169409, 54.223118, 0.0, -0.057787, -0.011846,
                    -20.315653, 0.0, 0.0, 0.0, 1.0],
        "elm__14": [0.057929, 0.0, 0.0, -75.365608, 0.0, -0.004041, 0.169409, 54.223118, 0.0, -0.057787, -0.011846,
                    -20.315653, 0.0, 0.0, 0.0, 1.0],
        "elm__15": [0.057929, 0.0, 0.0, -62.618069, 0.0, -0.004041, 0.169409, 54.223118, 0.0, -0.057787, -0.011846,
                    -20.315653, 0.0, 0.0, 0.0, 1.0],
        "elm__16": [0.057929, 0.0, 0.0, -49.816391, 0.0, -0.004041, 0.169409, 54.223118, 0.0, -0.057787, -0.011846,
                    -20.315653, 0.0, 0.0, 0.0, 1.0],
        "elm__17": [0.057929, 0.0, 0.0, -37.054989, 0.0, -0.004041, 0.169409, 54.223118, 0.0, -0.057787, -0.011846,
                    -20.315653, 0.0, 0.0, 0.0, 1.0],
    }

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "volpathmis", 'max_depth': -1},
        "my_global_fog": global_fog,
        "sensor": {
            "type": "perspective",
            "fov": 65.47,
            "to_world": cam_transform,
            "medium": {"type": "ref", "id": "my_global_fog"},
            "sampler": {"type": "stratified", "sample_count": config['batch_size']},
            "film": {
                "type": "hdrfilm",
                "width": config['res_w'], "height": config['res_h'],
                "pixel_format": "rgb", "rfilter": {"type": "gaussian", 'stddev': 0.25}
            }
        },
        # Materials
        "mat-road": {"type": "twosided", "bsdf": {"type": "principled", "base_color": {"type": "bitmap",
                                                                                       "filename": str(
                                                                                           tex_dir / "Road006_1K-PNG_Color.001.png")},
                                                  "roughness": {"type": "bitmap", "filename": str(
                                                      tex_dir / "Road006_1K-PNG_Roughness.001.png")}, "specular": 0.5}},
        "mat-Sidewalk": {"type": "twosided",
                         "bsdf": {"type": "principled", "base_color": {"type": "rgb", "value": [0.119, 0.119, 0.119]},
                                  "roughness": 0.77, "specular": 0.5}},
        "mat-Lamp_metal": {"type": "twosided",
                           "bsdf": {"type": "principled", "base_color": {"type": "rgb", "value": [0.446, 0.446, 0.446]},
                                    "metallic": 0.88, "roughness": 0.18}},
        "default-bsdf": {"type": "twosided", "bsdf": {"type": "diffuse"}},

        # Meshes
        "elm__2": {"type": "ply", "filename": str(mesh_dir / "Plane.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "mat-road"}},
        "elm__4": {"type": "ply", "filename": str(mesh_dir / "Cube.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "mat-Sidewalk"}},
        "elm__5": {"type": "ply", "filename": str(mesh_dir / "Cube_001.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "mat-Sidewalk"}},
        "elm__7": {"type": "ply", "filename": str(mesh_dir / "Plane_001.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__10": {"type": "ply", "filename": str(mesh_dir / "Lamp_001-Lamp_metal.ply"), "face_normals": True,
                    "bsdf": {"type": "ref", "id": "mat-Lamp_metal"}},
        "elm__13": {"type": "ply", "filename": str(mesh_dir / "LampCage2.ply"), "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
    }

    # Inject Lights
    for name, mat in matrices.items():
        scene_dict[name] = {
            "type": "disk",
            "flip_normals": True,
            "to_world": make_disk_transform(mat),
            "emitter": {"type": "area", "radiance": {"type": "irregular", "value": mercury_list}},
            "bsdf": {"type": "null"}
        }

    return scene_dict
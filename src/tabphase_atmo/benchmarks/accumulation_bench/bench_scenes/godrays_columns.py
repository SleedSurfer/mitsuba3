import mitsuba as mi
from ..utils import get_asset_path, get_phase_plugin


def get_scene(config):
    # --- ASSETS ---
    # Note: original code pointed to 'tests/scenes/meshes'
    # We assume you moved these to your standard assets folder under 'godrays' or similar
    # Adjust "godrays_test" to whatever folder name you put the meshes in.
    mesh_dir = get_asset_path("godrays_test", "meshes")

    # --- PHASE & ATMOSPHERE ---
    # Using a slightly generic mist for godrays
    phase_dict = get_phase_plugin(
        radius_mean=3.95,
        radius_std=0.03,
        note="godray_mist"
    )

    mist_def = {
        "type": "homogeneous",
        "sigma_t": 0.05,
        "albedo": 0.75,
        "phase": phase_dict
    }

    # --- TRANSFORMS ---
    cam_transform = mi.ScalarTransform4f.translate([18.739735, 3.279374, -1.910980]) \
        .rotate([0, 0, 1], 7.778137) \
        .rotate([0, 1, 0], -75.056272) \
        .rotate([1, 0, 0], -4.334142)

    spot_matrix_flat = [
        1.312597, -0.207895, 1.475958, -8.786666,
        0.310694, 1.961646, 0.000000, 2.718755,
        -1.457787, 0.230891, 1.328959, -6.061685,
        0.000000, 0.000000, 0.000000, 1.000000
    ]
    spot_transform = mi.ScalarTransform4f([spot_matrix_flat[i:i + 4] for i in range(0, 16, 4)])

    # --- SCENE DICT ---
    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "volpathmis",
            "max_depth": 3,
        },
        "my_global_mist": mist_def,

        "sensor": {
            "type": "perspective",
            "fov_axis": "x",
            "fov": 39.597755,
            "near_clip": 0.1,
            "far_clip": 100.0,
            "to_world": cam_transform,
            "medium": {"type": "ref", "id": "my_global_mist"},
            "sampler": {"type": "ldsampler", "sample_count": config['batch_size']},
            "film": {
                "type": "hdrfilm",
                "width": config['res_w'],
                "height": config['res_h'],
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            }
        },
        # --- EMITTERS ---
        "elm__7": {
            "type": "spot",
            "intensity": {
                "type": "rgb",
                "value": [39788.73, 31203.95, 20646.97]
            },
            "cutoff_angle": 15.159695,
            "beam_width": 0.3,
            "to_world": spot_transform
        },

        # --- SHAPES ---
        "default-bsdf": {"type": "twosided", "bsdf": {"type": "diffuse"}},

        # Geometry
        "elm__3": {"type": "ply", "filename": str(mesh_dir / "Cylinder.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__4": {"type": "ply", "filename": str(mesh_dir / "Cylinder_001.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__5": {"type": "ply", "filename": str(mesh_dir / "Cylinder_002.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__6": {"type": "ply", "filename": str(mesh_dir / "Plane.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__9": {"type": "ply", "filename": str(mesh_dir / "Cylinder_003.ply"), "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__10": {"type": "ply", "filename": str(mesh_dir / "Cylinder_004.ply"), "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__11": {"type": "ply", "filename": str(mesh_dir / "Cylinder_005.ply"), "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__12": {"type": "ply", "filename": str(mesh_dir / "Cylinder_006.ply"), "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__13": {"type": "ply", "filename": str(mesh_dir / "Plane_001.ply"), "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__14": {"type": "ply", "filename": str(mesh_dir / "Plane_002.ply"), "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
    }
    return scene_dict
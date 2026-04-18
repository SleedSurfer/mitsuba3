import mitsuba as mi

MESH_DIR = "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/meshes"

def get_scene(config, phase=None, sun_elevation_deg=20.0):
    return {
        "type": "scene",

        "integrator": {
            "type": "volpath",  # Swapped from volpath since standard cbox doesn't use volume scattering
            "max_depth": -1
        },

        "sensor": {
            "type": "perspective",
            "fov_axis": "smaller",
            "near_clip": 0.001,
            "far_clip": 100.0,
            "focus_distance": 1000.0,
            "fov": 39.3077,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 4),
                target=(0, 0, 0),
                up=(0, 1, 0)
            ),
            "sampler": {
                "type": "independent",
                "sample_count": config.get('batch_size', 128)
            },
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 256),
                "height": config.get('res_h', 256),
                "pixel_format": "rgb",
                "component_format": "float32",
                "rfilter": {"type": "tent"}
            }
        },

        # --- Top-Level BSDF Declarations ---
        "gray": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.85, 0.85, 0.85]}},
        "white": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.885809, 0.698859, 0.666422]}},
        "green": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.105421, 0.37798, 0.076425]}},
        "red": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.570068, 0.0430135, 0.0443706]}},
        "glass": {"type": "dielectric"},
        "mirror": {"type": "conductor"},

        # --- Emitters ---
        "light": {
            "type": "obj",
            "filename": f"{MESH_DIR}/cbox_luminaire.obj",
            "to_world": mi.ScalarTransform4f.translate([0, -0.01, 0]),
            "bsdf": {"type": "ref", "id": "white"},
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [18.387, 13.9873, 6.75357]}
            }
        },

        # --- Room Geometry ---
        "floor": {
            "type": "obj",
            "filename": f"{MESH_DIR}/cbox_floor.obj",
            "bsdf": {"type": "ref", "id": "white"}
        },
        "ceiling": {
            "type": "obj",
            "filename": f"{MESH_DIR}/cbox_ceiling.obj",
            "bsdf": {"type": "ref", "id": "white"}
        },
        "back": {
            "type": "obj",
            "filename": f"{MESH_DIR}/cbox_back.obj",
            "bsdf": {"type": "ref", "id": "white"}
        },
        "greenwall": {
            "type": "obj",
            "filename": f"{MESH_DIR}/cbox_greenwall.obj",
            "bsdf": {"type": "ref", "id": "green"}
        },
        "redwall": {
            "type": "obj",
            "filename": f"{MESH_DIR}/cbox_redwall.obj",
            "bsdf": {"type": "ref", "id": "red"}
        },

        # --- Primitives ---
        "mirrorsphere": {
            "type": "sphere",
            # Chained transformations: applies scale to unit sphere, then positions it
            "to_world": mi.ScalarTransform4f.translate([-0.3, -0.5, 0.2]).scale([0.5, 0.5, 0.5]),
            "bsdf": {"type": "ref", "id": "mirror"}
        },
        "glasssphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate([0.5, -0.75, -0.2]).scale([0.25, 0.25, 0.25]),
            "bsdf": {"type": "ref", "id": "glass"}
        }
    }
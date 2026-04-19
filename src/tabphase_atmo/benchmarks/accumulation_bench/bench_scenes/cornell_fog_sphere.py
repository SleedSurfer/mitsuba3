import mitsuba as mi

from src import create_atmospheric_phase

MESH_DIR = "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/meshes"


def get_scene(config, phase=None, sun_elevation_deg=20.0):
    phase = create_atmospheric_phase(phase, (0, 1, 0))
    return {
        "type": "scene",

        "integrator": {
            "type": "volpath",
            "max_depth": -1  # Still a thermal hazard, but you do you
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

        "frontlight": {
            "type": "spot",
            # Pushed back slightly to Z=1.5 so the tight cone has room to travel
            # Still aiming exactly at the top half of the sphere
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[-0.3, -0.25, 1.5],
                target=[-0.3, -0.5, 0.2],
                up=[0, 1, 0]
            ),
            # Spotlights use radiant intensity (W/sr) instead of radiance.
            # Because we are choking the beam angle so hard, it needs massive juice.
            "intensity": {"type": "rgb", "value": [8.0, 8.0, 8.0]},
            "cutoff_angle": 22.0,  # The hard boundary of the beam
            "beam_width": 7.0  # The inner core where the light starts fading out
        },

        # --- The Silver Lining / Rim Emitter ---
        "rimlight": {
            "type": "spot",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[-0.3, -0.5, -0.7],  # EXACTLY behind the sphere center
                target=[-0.3, -0.5, 0.2],   # Aiming exactly forward at the center
                up=[0, 1, 0]
            ),
            "intensity": {"type": "rgb", "value": [15.0, 15.0, 23.0]},
            "cutoff_angle": 45.0,
            "beam_width": 30.0
        },
        # "diskfrontlight": {
        #     "type": "disk",
        #     # Z=0.9 puts it in front of the sphere (which extends to Z=0.7).
        #     # Y=-0.25 puts it perfectly slightly above the equator.
        #     # Aimed directly down into the sphere's core at Z=0.2.
        #     "to_world": mi.ScalarTransform4f.look_at(
        #         origin=[-0.3, -0.25, 0.9],
        #         target=[-0.3, -0.5, 0.2],
        #         up=[0, 1, 0]
        #     ).scale([0.025, 0.025, 0.025]),
        #     "bsdf": {"type": "null"},  # Makes the back of the disk invisible to the camera
        #     "emitter": {
        #         "type": "area",
        #         "radiance": {"type": "rgb", "value": [300.0, 300.0, 300.0]}
        #     }
        # },

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
        "fogsphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate([-0.3, -0.5, 0.2]).scale([0.5, 0.5, 0.5]),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 16.0,
                "albedo": 0.9,
                "phase": phase#{"type": "hg", "g": 0.5}
            }
        },
        "glasssphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate([0.5, -0.75, -0.2]).scale([0.25, 0.25, 0.25]),
            "bsdf": {"type": "ref", "id": "glass"}
        }
    }
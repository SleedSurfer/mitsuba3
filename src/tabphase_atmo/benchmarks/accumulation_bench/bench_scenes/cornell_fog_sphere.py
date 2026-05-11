import mitsuba as mi

from nimbuscore.core.config  import Particle, DropletComposition, DropletShape, BackendType
from nimbuscore import create_atmospheric_phase

MESH_DIR = "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/meshes"

phase_mist_cfg = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 8.0, 0.006)],
                            num_angles=8192,
                            num_wavelengths=33,
                            num_phi_bins=360,
                            backend=BackendType.MIEPYTHON)


def get_scene(config, phase=None, sun_elevation_deg=20.0):
    phase_mist = create_atmospheric_phase(phase, (0, 1, 0),force_regen=True,threshold=1.0)
   # phase_rain = create_atmospheric_phase(phase, (0, 1, 0),force_regen=False)

    base_sigma_t = 30.0
    base_albedo = 0.9

    return {
        "type": "scene",

        "integrator": {
            "type": "volpath",
            "max_depth": -1,
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
                #"rfilter": {"type": "mitchell"}
            }
        },

        # --- Top-Level BSDF Declarations ---
        "gray": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.85, 0.85, 0.85]}},
        "white": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.885809, 0.698859, 0.666422]}},
        "green": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.105421, 0.37798, 0.076425]}},
        "red": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.570068, 0.0430135, 0.0443706]}},
        "glass": {"type": "dielectric"},

        #--- Emitters ---
        "light": {
            "type": "obj",
            "filename": f"{MESH_DIR}/cbox_luminaire.obj",
            "to_world": mi.ScalarTransform4f.translate([0, -0.01, 0]),
            "bsdf": {"type": "ref", "id": "white"},
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [x*0.3 for x in [18.387, 13.9873, 6.75357]]}#[18.387, 13.9873, 6.75357]}
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
            "intensity": {"type": "rgb", "value": [x*3 for x in [3.0, 3.0, 3.0]]},
            "cutoff_angle": 20.0,  # The hard boundary of the beam
            "beam_width": 17.0  # The inner core where the light starts fading out
        },

        # --- The Silver Lining / Rim Emitter ---
        "rimlight": {
            "type": "spot",
            "to_world": mi.ScalarTransform4f.look_at(
                # Calculated origin to be perfectly collinear with Camera and Sphere
                origin=[-0.473, -0.789, -0.9],
                target=[-0.3, -0.5, 0.2],    # Sphere Center
                up=[0, 1, 0]
            ),
            "intensity": {"type": "rgb", "value": [x*5 for x in [7.0, 7.0, 9.3]]}, # More juice for the thick fog
            "cutoff_angle": 45, # Tighter beam to focus energy on the rim
            "beam_width": 25
        },

        # # --- The Sniper Arc Light (Optically Correct) ---
        # # Replace the previous micro-light with this one.
        # "arclight": {
        #     "type": "spot",
        #     # Camera is at [0, 0, 4]. We put the light at Z=6.0 (behind you),
        #     # and offset it to the bottom left [-1.0, -1.0].
        #     # This creates the proper directional rays to hit the top-right slab
        #     # and bounce back into the camera lens at that crucial ~138° scattering angle.
        #     "to_world": mi.ScalarTransform4f.look_at(
        #         origin=[-1.0, -1.0, 6.0],
        #         target=[0.6, 0.7, 0.1],  # Dead center of the rainslab
        #         up=[0, 1, 0]
        #     ),
        #     # You'll need high intensity again because the light is traveling 6 meters.
        #     "intensity": {"type": "rgb", "value": [x * 10 for x in [8.0, 6.0, 9.0]]},
        #     # We choke the cutoff angle hard. A 3-degree beam is basically a laser pointer.
        #     # It will strictly illuminate the slab and nothing else, saving your contrast.
        #     "cutoff_angle": 5.0,
        #     "beam_width": 2.5
        # },
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

        # # --- The Dedicated Rain Slab ---
        # "rainslab": {
        #     "type": "cube",
        #     # Scale it into a thin 0.1-meter deep sheet spanning 0.6 meters wide/high
        #     # Translate it to the top-right corner, slightly pushed back
        #     "to_world": mi.ScalarTransform4f.translate([0.0, 0.0, 0.0]).scale([3.0, 3.0, 2.0]),#"to_world": mi.ScalarTransform4f.translate([0.6, 0.7, 0.1]).scale([0.3, 0.3, 0.05]),
        #     "bsdf": {"type": "null"},
        #     "interior": {
        #         "type": "homogeneous",
        #         "sigma_t": 0.1,
        #         "albedo": 0.95,
        #         "phase": phase_rain
        #     }
        # },



        #--- Primitives ---
        "fogsphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate([-0.3, -0.5, 0.2]).scale([0.5, 0.5, 0.5]),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": base_sigma_t,
                "albedo": base_albedo,
                "phase": phase_mist#{"type": "hg", "g": 0.95}
            }
        },
        "glasssphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate([0.5, -0.75, -0.2]).scale([0.25, 0.25, 0.25]),
            "bsdf": {"type": "ref", "id": "glass"}
        }
    }
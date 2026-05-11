import mitsuba as mi

from benchmarks.accumulation_bench.utils import generate_cloud_grid, assemble_volume_dict, generate_cumulus_grid
from nimbuscore.core.config  import Particle, DropletComposition, DropletShape, BackendType
from nimbuscore import create_atmospheric_phase

MESH_DIR = "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/meshes"

phase_mist_cfg = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 8.0, 0.003)],
                            num_angles=8192,
                            num_wavelengths=6,
                            num_phi_bins=360,
                            backend=BackendType.MIEPYTHON)


def get_scene(config, phase=None, sun_elevation_deg=20.0):
    phase_mist = create_atmospheric_phase(phase, (0, 1, 0), force_regen=False)
    #phase_rain = create_atmospheric_phase(phase, (0, 1, 0),forward_peak_limit=1.0, force_regen=False)

    base_sigma_t = 10.0
    base_albedo = 0.99

    # 1. Generate the cloud volume grid (crank res up to 128 for good fluff)
    cloud_grid = generate_cumulus_grid(res=128, seed=137, density_multiplier=20.0)

    # 2. Assemble the heterogeneous volume dict
    # We pass bounds=(0.5, 0.5, 0.5) to match the scale of the old fogsphere
    cloud_volume = assemble_volume_dict(
        density_grid=cloud_grid,
        phase_dict=phase_mist,#{'type':'hg','g':0.877},#phase_mist,
        bounds=(1.0, 1.0, 1.0),
        albedo=base_albedo,
        density_scale=base_sigma_t
    )

    translation = mi.ScalarTransform4f.translate([-0.3, -0.37, 0.2])

    # Move the outer shell
    cloud_volume["to_world"] = translation @ cloud_volume["to_world"]

    # Move the actual cloud data to match the shell
    cloud_volume["interior"]["sigma_t"]["to_world"] = translation @ cloud_volume["interior"]["sigma_t"]["to_world"]

    return {
        "type": "scene",

        "integrator": {
            "type": "volpath",
            "max_depth": -1,
            "rr_depth": 10,
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
                "radiance": {"type": "rgb", "value": [x*1.5 for x in [18.387, 13.9873, 6.75357]]}
            }
        },

        "frontlight": {
            "type": "spot",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[-0.3, -0.25, 1.5],
                target=[-0.3, -0.5, 0.2],
                up=[0, 1, 0]
            ),
            "intensity": {"type": "rgb", "value": [x * 2 for x in [3.0, 3.0, 3.0]]},
            "cutoff_angle": 40.0,
            "beam_width": 17.0
        },

        "rimlight": {
            "type": "spot",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[-0.473, -0.73, -0.7],
                target=[-0.3, -0.67, 0.2],
                up=[0, 1, 0]
            ),
            "intensity": {"type": "rgb", "value": [x *0.3 for x in [7.0, 7.0, 9.3]]},
            "cutoff_angle": 90,
            "beam_width": 60
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

        "actual_cloud": cloud_volume,

        "glasssphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate([0.5, -0.75, -0.2]).scale([0.25, 0.25, 0.25]),
            "bsdf": {"type": "ref", "id": "glass"}
        }
    }
import math
import mitsuba as mi

from ..utils import (
    get_asset_path,
    create_sunsky_emitter,
    generate_cumulus_grid,
    assemble_volume_dict,
    create_sun_aligned_camera,
    visualize_grid_seeds
)
from nimbuscore.core.gen_manager import create_atmospheric_phase
from nimbuscore.core.config import *


def create_anti_sun_camera(sun_elevation_deg, distance=5.0):
    """
    Puts the camera in front of the origin, looking AT the origin,
    with the sun shining from behind the camera. (Rainbow viewing mode!)
    """
    sun_elev_rad = math.radians(sun_elevation_deg)
    sun_dir_y = math.cos(sun_elev_rad)
    sun_dir_z = math.sin(sun_elev_rad)

    cam_origin = [
        0.0,
        distance * sun_dir_y,  # Positive distance instead of negative!
        distance * sun_dir_z
    ]

    return mi.ScalarTransform4f.look_at(
        origin=cam_origin,
        target=[0.0, 0.0, 0.0],
        up=(0.0, 0.0, 1.0)
    )

rain_phase_cfg = Particle.droplet(num_angles=8192,
                                  num_wavelengths=30,
                                  num_phi_bins=360,
                                  backend=BackendType.DRJIT,
                                  composition=[DropletComposition(DropletShape.SPHERE, 1.0, 400.0, 0.003)])


def get_scene(config, cloud_phase, sun_elevation_deg=130.0):
    # 1. Bake BOTH phase functions
    print("Baking Cloud Phase...")
    cloud_phase_dict = create_atmospheric_phase(cloud_phase, up_vector=(0.0, 0.0, 1.0), force_regen=False)

    print("Baking Rain Phase...")
    rain_phase_dict = create_atmospheric_phase(rain_phase_cfg, up_vector=(0.0, 0.0, 1.0), force_regen=False)

    # ==========================================
    # ☁️ CLOUD MASTER CONTROLS
    # ==========================================
    cloud_pos = [-20.0, 50.0, 60.0]  # Where it sits in the sky
    cloud_scale = [150.0, 150.0, 100.0]  # Make it wider/taller here

    # Rotation (in degrees).
    # Z rotates it like a record (changes which side faces the camera)
    # Y/X tilt it (be careful, tilting too much might clip it into your rain slab!)
    rot_z_deg = 80.0
    rot_y_deg = -110.0

    # ==========================================
    # 🏗️ THE TRANSFORM STACK (Reads Right-to-Left)
    # ==========================================

    # 1. CENTER: Move the [0, 1] grid so its center [0.5, 0.5, 0.5] sits at the origin [0, 0, 0]
    center_tx = mi.ScalarTransform4f.translate([-0.5, -0.5, -0.5])

    # 2. ROTATE: Now that it's centered, spin it in place
    spin_z_tx = mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=rot_z_deg)
    spin_y_tx = mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=rot_y_deg)

    # 3. SCALE: Blow it up to real-world meters
    scale_tx = mi.ScalarTransform4f.scale(cloud_scale)

    # 4. POSITION: Lift it into the sky
    move_tx = mi.ScalarTransform4f.translate(cloud_pos)

    # 🔗 MULTIPLY: Right-most matrix applies FIRST.
    # Order: Center -> Spin Z -> Spin Y -> Scale -> Move
    grid_transform = move_tx @ scale_tx @ spin_y_tx @ spin_z_tx @ center_tx

    # 📦 THE INVISIBLE BOX (DO NOT TOUCH)
    # This automatically shrinks the [-1,1] cube to [0,1] and then inherits all your spins/scales
    box_transform = grid_transform @ mi.ScalarTransform4f.translate([0.5, 0.5, 0.5]) @ mi.ScalarTransform4f.scale(
        [0.5, 0.5, 0.5])


    cloud_volume_dict = {
        "type": "cube",
        "to_world": box_transform,  # <--- USE THE FIXED BOX TRANSFORM HERE
        "bsdf": {"type": "null"},
        "interior": {
            "type": "heterogeneous",
            "sigma_t": {
                "type": "gridvolume",
                "filename": "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/volumes/cloud_08_variant_0000.vol",
                "to_world": grid_transform  # <--- USE THE GRID TRANSFORM HERE
            },
            "scale": 5.0,
            "albedo": 0.999,
            "phase": {'type':'hg','g':0.877}#cloud_phase_dict
        }
    }

    # 2. Bake the physical cumulus cloud COOL SEEDS : 4,11


    # raw_grid = generate_cumulus_grid(
    #     res=256,
    #     seed=11,
    #     radius=0.25,
    #     density_multiplier=1.0,
    #     width_stretch=0.4,
    #     height_stretch=0.6
    # )


    # # 3. Assemble the volume
    # cloud_volume_dict = assemble_volume_dict(
    #     density_grid=raw_grid,
    #     phase_dict=cloud_phase_dict,
    #     bounds=(100.0, 100.0, 60.0),  # Give the cloud some chunky physical volume
    #     albedo=0.995,  # Keep the 0.99 hack for Russian Roulette
    #     density_scale=20.0  # Your master optical depth knob
    # )

    # ✅ THE FIX: Lift BOTH the bounding shell and the voxel grid data
    #lift_transform = mi.ScalarTransform4f.translate([0.0, 0.0, 60.0])

    # 1. Move the outer shell
    #cloud_volume_dict["to_world"] = lift_transform @ cloud_volume_dict["to_world"]

    # 2. Move the actual volumetric grid data inside it
    #cloud_volume_dict["interior"]["sigma_t"]["to_world"] = lift_transform @ cloud_volume_dict["interior"]["sigma_t"]["to_world"]

    # 4. Assemble the scene payload
    return {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": -1,
            "rr_depth": 50
        },
        "sensor": {
            "type": "perspective",
            "fov": 75.0,
            "to_world": create_sun_aligned_camera(50, distance=100.0)@mi.ScalarTransform4f.translate([0.0, 0.0, 30.0]),
            "sampler": {"type": "independent", "sample_count": config.get('batch_size', 1024)},
            "film": {
                "type": "hdrfilm",
                "width": config.get('res_w', 1024),
                "height": config.get('res_h', 1024),
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            }
        },

        # The open sky environment
        "sky": create_sunsky_emitter(
            sun_elevation_deg=158, #148
            use_direct_vector=True,
            turbidity=2.5,
            sun_scale=1.0
        ),

        "cloud_volume": cloud_volume_dict,

        # "rain_slab": {
        #     "type": "cube",
        #     # Scale Z by 10 (20m tall bounding box) and lift by 10.
        #     # Sits perfectly from Z=0 to Z=20. Safe distance from the cloud!
        #     "to_world": mi.ScalarTransform4f.translate([6.0, -30.0, -85.0]) @ mi.ScalarTransform4f.scale(
        #         [28.0, 10.0, 70.0]),
        #     "bsdf": {"type": "null"},
        #     "interior": {
        #         "type": "homogeneous",
        #         "sigma_t": 0.01,
        #         "albedo": 0.99,
        #         "phase": rain_phase_dict,
        #     },
        # }
    }
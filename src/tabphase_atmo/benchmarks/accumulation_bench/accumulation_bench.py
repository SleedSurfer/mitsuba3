import mitsuba as mi
import sys
import importlib
import time
from pathlib import Path
import drjit as dr
import numpy as np
from PIL import Image
import pyoidn
from bench_scenes import street_lamps,godrays_columns,pure_rainbow,brocken_spectre


# 1. INIT & VARIANTS
mi.set_variant("llvm_spectral")

# ==========================================
#               THE CONFIG
#      (Twist these knobs, Space Cowboy)
# ==========================================
TARGET_SPP = 4096
BATCH_SIZE = 64#256#
CLAMP_VAL = 100.0  # DEPRECATED
RES_W = 640#1920#3840#
RES_H = 480#1080#2160#
DOWNSCALE_TARGET = (1920, 1080)

# Paths
HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "bench_renders"
WRAPPER_PATH = Path("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/wrapper.py").resolve()

# ==========================================
#           ATMOSPHERE SETUP
# ==========================================
# Keeps your atmospheric logic reusable across scenes
if not WRAPPER_PATH.exists():
    print(f"Warning: Wrapper not found at {WRAPPER_PATH}, using fallback HG phase.")
    PHASE_PLUGIN_DICT = {"type": "hg", "g": 0.8}
else:
    PACKAGE_ROOT = str(WRAPPER_PATH.parent.parent)
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    mod = importlib.import_module("python.atmospheric.wrapper")
    create_atmospheric_phase = getattr(mod, "create_atmospheric_phase")

    # You can parameterize this later if you want different mists per scene
    #PHASE_PLUGIN_DICT = create_atmospheric_phase(
    #    radius_mean_um=2.0, radius_std_um=0.5, num_angles=2048,
    #    num_wavelengths=64, note="mist_bench", force_regen=False
    #)MIST

    # PHASE_PLUGIN_DICT = create_atmospheric_phase(
    #     radius_mean_um=16.0,
    #     radius_std_um=2.6,
    #     num_angles=4096,  # Increased resolution for sharp rings
    #     num_wavelengths=32,
    #     note="glory",
    #     force_regen=False  # Force it to bake the new table
    # )

    # PHASE_PLUGIN_DICT = create_atmospheric_phase( # THE MONEY SHOT
    #     radius_mean_um=6.0,
    #     radius_std_um=1.0,
    #     num_angles=4096,
    #     num_wavelengths=64,
    #     note="glory",
    #     force_regen=False
    # )

    PHASE_PLUGIN_DICT = create_atmospheric_phase(  # Fog for lamps
        radius_mean_um=3.95,
        radius_std_um=0.03,
        num_angles=4096,
        num_wavelengths=64,
        note="glory_395_003",
        force_regen=False
    )

    # PHASE_PLUGIN_DICT = create_atmospheric_phase(
    #     radius_mean_um=100.0, radius_std_um=10.0, num_angles=2048,
    #     num_wavelengths=64, note="rainbow_proper", force_regen=False
    # )

def get_exact_blender_camera(loc, rot_deg):
    """
    Reconstructs the Mitsuba camera matrix directly from Blender values.
    Bypasses the ambiguous 'Matrix -> Euler -> XML' chain that breaks the plugin.
    """
    # 1. CONSTRUCT BLENDER ROTATION MATRIX (XYZ Euler)
    rx, ry, rz = np.radians(rot_deg)

    mat_x = np.array([[1, 0, 0, 0], [0, np.cos(rx), -np.sin(rx), 0], [0, np.sin(rx), np.cos(rx), 0], [0, 0, 0, 1]])
    mat_y = np.array([[np.cos(ry), 0, np.sin(ry), 0], [0, 1, 0, 0], [-np.sin(ry), 0, np.cos(ry), 0], [0, 0, 0, 1]])
    mat_z = np.array([[np.cos(rz), -np.sin(rz), 0, 0], [np.sin(rz), np.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Blender Order: Z @ Y @ X
    rot_mat = mat_z @ mat_y @ mat_x

    # 2. ADD TRANSLATION
    blender_matrix = rot_mat.copy()
    blender_matrix[:3, 3] = loc

    # 3. APPLY PLUGIN FIX (Local Flip 180 Y)
    # Matches: init_rot = Matrix.Rotation(np.pi, 4, 'Y')
    init_rot = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    fixed_matrix = blender_matrix @ init_rot

    # 4. COORDINATE CONVERSION (Z-up -> Y-up)
    basis_change = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    return mi.ScalarTransform4f(basis_change @ fixed_matrix)


def generate_cloud_grid(
    res: int = 32,
    seed: int = 1337,
    contrast_power: float = 2.5,
    fine_noise_amp: float = 0.2,
    base_grid_scale: float = 0.3,
    density_scale: float = 1.0,
    base_density: float = 0.05,
    clamp_max: float | None = None
):
    """
    Generates a cloud grid with adjustable density controls.

    To crank density, call with e.g.:
      generate_cloud_grid(density_scale=3.0, base_density=0.08)

    Parameters you can tweak:
    - contrast_power: lower -> more midtones, higher -> punchier holes
    - fine_noise_amp: larger -> more grain/detail
    - base_grid_scale: multiplies the combined noise before scaling
    - density_scale: global multiplier to 'crank' density
    - base_density: additive floor to avoid pure zeros
    - clamp_max: optional upper clamp for density
    """
    rng = np.random.default_rng(seed)

    # Raw noise
    raw_noise = rng.random((res, res, res))

    # Contrast curve (magic sauce)
    contrast_noise = np.power(raw_noise, contrast_power)

    # Fine detail
    fine_noise = rng.random((res, res, res)) * fine_noise_amp

    combined = contrast_noise * 0.8 + fine_noise * 0.2

    # Scale + global density multiplier + base offset
    density = combined * base_grid_scale * density_scale + base_density

    if clamp_max is not None:
        density = np.minimum(density, clamp_max)

    return mi.VolumeGrid(mi.TensorXf(density))


# ==========================================
#           THE RENDER BENCH
#      (Don't touch this logic often)
# ==========================================
def run_render_bench(scene_dict, run_name):
    """
    Takes a scene dictionary and a name.
    Renders, clamps, accumulates, saves 4K, saves 1080p.
    """
    print(f"--- STARTING BENCH: {run_name} ---")

    # Setup Output Directories for this specific run
    run_dir = OUT_ROOT / run_name
    exr_dir = run_dir / "exr"
    exr_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scene...")
    scene = mi.load_dict(scene_dict)

    total_passes = TARGET_SPP // BATCH_SIZE

    print(f"Settings: {TARGET_SPP} SPP ({total_passes} passes x {BATCH_SIZE}). Clamping at {CLAMP_VAL} (not really).")

    # Initialize Buffer
    film_size = scene.sensors()[0].film().crop_size()
    acc_buffer = dr.zeros(mi.TensorXf, shape=(film_size[1], film_size[0], 3))

    start_time = time.time()

    for i in range(total_passes):
        # Render batch
        img = mi.render(scene, seed=i, spp=BATCH_SIZE)

        # Accumulate
        acc_buffer += img

        # Stats & Banter
        elapsed = time.time() - start_time
        avg_per_pass = elapsed / (i + 1)
        remaining_passes = total_passes - (i + 1)
        if (i + 1) % 5 == 0:  # Reduce console spam, only print every 5th pass
            print(f"[{run_name}] Pass {i + 1}/{total_passes}. ETA: {remaining_passes * avg_per_pass / 60:.2f} mins")

        # Previews
        pass_num = i + 1
        save_early = pass_num in [1,2,3, 5, 10, 20, 50]
        save_periodic = pass_num > 50 and pass_num % 10 == 0

        if save_early or save_periodic:
            temp_img = acc_buffer / pass_num
            mi.util.write_bitmap(str(exr_dir / f"preview_{pass_num}_{BATCH_SIZE*pass_num}spp.exr"), temp_img)

    # 1. Final Average to get your 32-bit float array
    final_image = acc_buffer / total_passes
    final_np = np.array(final_image)  # Should be (H, W, 3) float32

    print("Vaporizing fireflies with pyoidn...")

    # 2. Create the Device (CPU)
    device = pyoidn.create_device()

    # 3. Create and execute the filter
    # For OIDN 2.0+, the filter "RT" (Ray Tracing) is the standard
    output_np = pyoidn.denoise(device, final_np, hdr=True)

    # Convert the clean numpy array back to a Mitsuba Tensor
    final_image_clean = mi.TensorXf(output_np)

    # 1. Save Full Res (4K)
    print("Saving Full Res (4K) Master...")
    mi.util.write_bitmap(str(exr_dir / "render_final_4k.exr"), final_image_clean)

    # 2. Downscale Logic
    print("Downscaling to 1080p...")
    final_np = np.array(final_image)
    # PIL Resize logic
    channels = [Image.fromarray(final_np[:, :, c], mode='F').resize(DOWNSCALE_TARGET, Image.LANCZOS) for c in range(3)]
    downscaled_np = np.stack([np.array(ch) for ch in channels], axis=-1)
    final_image_downscaled = mi.TensorXf(downscaled_np)

    mi.util.write_bitmap(str(exr_dir / "render_final_1080p.exr"), final_image_downscaled)
    print(f"--- FINISHED: {run_name} ---\n")


# ==========================================
#           SCENE GENERATORS
#      (Define your experiments here)
# ==========================================

def get_godray_scene_01():

    mesh_dir = Path(f"/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/tests/scenes/meshes")
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

    # --- MIST SETUP ---
    mist_def = {
        "type": "homogeneous",
        "sigma_t": 0.05,
        "albedo": 0.75,
        "phase": PHASE_PLUGIN_DICT
    }

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
            "sampler": {"type": "ldsampler", "sample_count": BATCH_SIZE},  # Default for safety
            "film": {
                "type": "hdrfilm",
                "width": RES_W,
                "height": RES_H,
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
        "elm__3": {"type": "ply", "filename": f"{mesh_dir}/Cylinder.ply", "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__4": {"type": "ply", "filename": f"{mesh_dir}/Cylinder_001.ply", "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__5": {"type": "ply", "filename": f"{mesh_dir}/Cylinder_002.ply", "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__6": {"type": "ply", "filename": f"{mesh_dir}/Plane.ply", "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__9": {"type": "ply", "filename": f"{mesh_dir}/Cylinder_003.ply", "face_normals": True,
                   "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__10": {"type": "ply", "filename": f"{mesh_dir}/Cylinder_004.ply", "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__11": {"type": "ply", "filename": f"{mesh_dir}/Cylinder_005.ply", "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__12": {"type": "ply", "filename": f"{mesh_dir}/Cylinder_006.ply", "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__13": {"type": "ply", "filename": f"{mesh_dir}/Plane_001.ply", "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},
        "elm__14": {"type": "ply", "filename": f"{mesh_dir}/Plane_002.ply", "face_normals": True,
                    "bsdf": {"type": "ref", "id": "default-bsdf"}},

        "default-bsdf": {"type": "twosided", "bsdf": {"type": "diffuse"}}
    }
    return scene_dict


def get_brocken_spectre_scene():
    # --- SCENE SPECIFIC PATHS ---
    scene_name = "Spectre"
    mesh_dir = Path(f"/home/speedlord/Desktop/Mitsuba3-scenes/{scene_name}/meshes")
    light_mod = 40.0  # Original was 1.0, toned down for visibility
    # --- TRANSFORMS ---
    # Updated Camera: Z axis moved significantly back (-1.73)
    cam_transform = mi.ScalarTransform4f() \
        .rotate([1, 0, 0], 2.5044780654876655e-06) \
        .rotate([0, 1, 0], -5.008956130975331e-06) \
        .rotate([0, 0, 1], -5.008956130975331e-06) \
        .translate([-0.151983, 0.596070, -1.730589])

    # Updated Rectangle Light Matrix (New Z translation: -22.928)
    rect_matrix_flat = [
        0.000000, -0.019629, -0.000000, -0.151983,
        -0.019629, 0.000000, -0.000000, 0.596070,
        0.000000, 0.000000, -0.019629, -22.928349,
        0.000000, 0.000000, 0.000000, 1.000000
    ]
    # Reshape to 4x4
    rect_transform = mi.ScalarTransform4f([rect_matrix_flat[i:i + 4] for i in range(0, 16, 4)]).scale(0.1)

    # --- CLOUD GRID SETUP ---
    # We scale the noise grid to cover the bunny area.
    # The grid is naturally 0..1, so we scale it up to size 10x10x10
    grid_transform = mi.ScalarTransform4f.translate([-5, -5, -5]).scale(10.0)

    cloud_medium = {
        "type": "heterogeneous",  # <--- Switched to Heterogeneous
        "albedo": 0.98,
        "phase": PHASE_PLUGIN_DICT,
        "sigma_t": {
            "type": "gridvolume",
            "grid": generate_cloud_grid(),
            "to_world": grid_transform,
            "filter_type": "trilinear"  # Cubic smoothing makes the 8x8 grid look like soft clouds
        }
    }

    # # --- CLOUD/MEDIUM SETUP ---
    # # Preserved exactly as requested
    # cloud_medium = {
    #     "type": "homogeneous",
    #     "sigma_t": 0.1,
    #     "albedo": 0.8,
    #     "phase": PHASE_PLUGIN_DICT
    # }

    # --- SCENE DICT ---
    return {
        "type": "scene",
        "sub_global_mist": cloud_medium,

        # Preserved Integrator Settings (User Override)
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
            "sampler": {"type": "ldsampler", "sample_count": BATCH_SIZE},
            "film": {
                "type": "hdrfilm",
                "width": RES_W,
                "height": RES_H,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            }
        },
        # --- MATERIALS ---
        "default-bsdf": {
            "type": "twosided",
            "bsdf": {"type": "diffuse"}
        },
        # --- EMITTERS ---
        # Updated Rectangle Light (New Radiance)
        "elm__3": {
            "type": "rectangle",
            "flip_normals": True,
            "to_world": rect_transform,
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "spectrum",
                    "value": 1416666.0 * light_mod,
                }
                # "radiance": {
                #     "type": "rgb",
                #     "value": [x*light_mod for x in [1400000.0, 1350000.0, 1500000.0]]
                # }
            },
            "bsdf": {"type": "null"}
        },

        # --- SHAPES ---
        "elm__2": {
            "type": "ply",
            "filename": str(mesh_dir / "sbunny.ply"),
            "face_normals": True,
            "bsdf": {"type": "ref", "id": "default-bsdf"}
        },
        # THE CLOUD PLANE (Logic preserved, ID updated to match XML)
        "elm__4": {
            "type": "ply",
            "filename": str(mesh_dir / "Plane.ply"),
            "face_normals": True,
            "bsdf": {"type": "null"},
            #"interior": cloud_medium
        }
    }

# ==========================================
#           SCENE: STREET LAMPS
# ==========================================
def get_street_fog_scene():
    # --- PATHS ---
    scene_name = "RoadLampsXML"
    scene_root = Path(f"/home/speedlord/Desktop/Mitsuba3-scenes/{scene_name}")
    mesh_dir = scene_root / "meshes"
    tex_dir = scene_root / "textures"
    max_depth = -1
    light_mod = 0.6

    # # --- Clear mercury vapor lamp spectrum data
    # mercury_data = [
    #     (365.0, 0.5),  # UV/Deep Violet (Spooky)
    #     (404.7, 4.0),  # Violet Line
    #     (435.8, 10.0),  # Blue G-line (The iconic "ghost" color)
    #     (546.1, 12.0),  # Green e-line (The main brightness)
    #     (577.0, 3.0),  # Yellow Doublet 1
    #     (579.1, 3.0),  # Yellow Doublet 2
    #     (600.0, 0.1),  # Drop off to zero red
    #     (700.0, 0.0)
    # ]

    # --- sickly green mercury vapor lamp
    mercury_data = [
        (365.0, 0.1),  # UV (Faint)
        (404.7, 2.0),  # Violet (Weak)
        (435.8, 3.0),  # CRUSHED BLUE (Was 10.0) -> This kills the "clean" cyan look
        (546.1, 28.0),  # SUPERCHARGED GREEN (Was 12.0) -> The Matrix vibe
        (577.0, 6.0),  # Boosted Yellow 1 -> Adds "grime"
        (579.1, 6.0),  # Boosted Yellow 2
        (600.0, 0.5),  # Red dropoff
        (700.0, 0.0)
    ]

    mercury_flat_list = []
    for w, v in mercury_data:
        mercury_flat_list.append((w,v * 500.0 * light_mod))

    # --- ATMOSPHERE (Attached to Camera) ---
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
        "phase": PHASE_PLUGIN_DICT
    }

    # global_fog = {
    #     "type": "homogeneous",
    #     "sigma_t": 0.04,
    #     "albedo": 0.7,
    #     "phase": {"type": "hg",
    #               "g": 0.85}
    # }

    vacuum_medium = {
        "type": "homogeneous",
        "sigma_t": 0.0,
        "albedo": 0.0
    }

    # --- CAMERA TRANSFORM ---
    cam_transform = get_exact_blender_camera(
        loc=[-19.767, 11.002, 44.227],
        rot_deg=[98.737, 0.68787, 57.42]
    )

    def make_disk_transform(flat_matrix):
        return mi.ScalarTransform4f([flat_matrix[i:i + 4] for i in range(0, 16, 4)])

    # The Lamp Disk Matrices
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

    # --- SCENE DICTIONARY ---
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "volpathmis", 'max_depth': max_depth, 'hide_emitters': False}, #

        "my_global_fog": global_fog,
        "my_vacuum": vacuum_medium,

        "sensor": {
            "type": "perspective",
            "fov": 65.470450,
            "near_clip": 0.5,
            "far_clip": 506.5,
            "to_world": cam_transform,
            "medium": {"type": "ref", "id": "my_global_fog"},
            "sampler": {"type": "stratified", "sample_count": BATCH_SIZE},
            "film": {
                "type": "hdrfilm",
                "width": RES_W,
                "height": RES_H,
                "pixel_format": "rgb",
                "rfilter": {"type": "gaussian", 'stddev': 0.25}
            }
        },

        # --- MATERIALS ---
        # NEW: The Vacuum Material (Invisible Glass)
        # This effectively creates a "hole" in the medium because it has no interior medium defined.
        "mat-vacuum": {
            "type": "dielectric",
            "int_ior": 1.0,
            "ext_ior": 1.0,
            "specular_reflectance": 0.0  # Ensure it's perfectly invisible, no reflections
        },

        "mat-road": {
            "type": "twosided",
            "bsdf": {
                "type": "principled",
                "base_color": {"type": "bitmap", "filename": str(tex_dir / "Road006_1K-PNG_Color.001.png")},
                "roughness": {"type": "bitmap", "filename": str(tex_dir / "Road006_1K-PNG_Roughness.001.png")},
                "specular": 0.5
            }
        },
        "mat-Sidewalk": {
            "type": "twosided",
            "bsdf": {
                "type": "principled",
                "base_color": {"type": "rgb", "value": [0.119727, 0.119727, 0.119727]},
                "roughness": 0.770801,
                "specular": 0.5
            }
        },
        "mat-Lamp_metal": {
            "type": "twosided",
            "bsdf": {
                "type": "principled",
                "base_color": {"type": "rgb", "value": [0.446357, 0.446357, 0.446357]},
                "metallic": 0.889764,
                "roughness": 0.184156,
                "specular": 0.444882
            }
        },
        "default-bsdf": {"type": "twosided", "bsdf": {"type": "diffuse"}},

        # ... (Your standard ply meshes stay here) ...
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

    # --- INJECT LIGHTS AND VACUUM BUBBLES ---
    for name, mat in matrices.items():
        # 1. The Light Disk (Unchanged)
        scene_dict[name] = {
            "type": "disk",
            "flip_normals": True,
            "to_world": make_disk_transform(mat),
            "emitter": {
                "type": "area",
                "radiance": {"type": "spectrum", "value": mercury_flat_list}
            },
            "bsdf": {"type": "null"}
        }
    return scene_dict

# ==========================================
#               MAIN QUEUE
# ==========================================
if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a city to burn.")


    # scene_1 = get_brocken_spectre_scene()
    # run_render_bench(scene_1,run_name="brocken_spectre_flipped_monowavelength_64batch")

    scene_2 = get_street_fog_scene()
    run_render_bench(scene_2,run_name="fog_streetlights_monowavelength")

    # scene_3 = get_godray_scene_01()
    # run_render_bench(scene_3, run_name="godrays_image_flipped")
    print("All jobs done. Go touch grass.")
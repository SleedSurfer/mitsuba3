import mitsuba as mi
import sys
from pathlib import Path as _Path
import importlib
from pathlib import Path as Path
import time
import drjit as dr  # <--- [NEW] Needed for the clamp math
import numpy as np
from PIL import Image

mi.set_variant("llvm_ad_spectral")

# --- PATHS ---
HERE = _Path(__file__).resolve().parent
OUT_DIR = HERE / "godrays_acummulated"
EXR_DIR = OUT_DIR / "exr"
EXR_DIR.mkdir(parents=True, exist_ok=True)
# --- WRAPPER SETUP ---
WRAPPER_ABS_PATH = _Path("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/wrapper.py").resolve()
if not WRAPPER_ABS_PATH.exists():
    PHASE_PLUGIN_DICT = {"type": "hg", "g": 0.8}
else:
    PACKAGE_ROOT = str(WRAPPER_ABS_PATH.parent.parent)
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    mod = importlib.import_module("python.atmospheric.wrapper")
    create_atmospheric_phase = getattr(mod, "create_atmospheric_phase")
    PHASE_PLUGIN_DICT = create_atmospheric_phase(
        radius_mean_um=2.0, radius_std_um=0.5, num_angles=2048,
        num_wavelengths=64, note="mist_test", force_regen=False
    )

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
    "albedo": 0.85,
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
        "sampler": {"type": "ldsampler", "sample_count": 64},  # Default for safety
        "film": {
            "type": "hdrfilm",
            "width": 3840,
            "height": 2160,
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
    "elm__3": {"type": "ply", "filename": "meshes/Cylinder.ply", "face_normals": True,
               "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__4": {"type": "ply", "filename": "meshes/Cylinder_001.ply", "face_normals": True,
               "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__5": {"type": "ply", "filename": "meshes/Cylinder_002.ply", "face_normals": True,
               "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__6": {"type": "ply", "filename": "meshes/Plane.ply", "face_normals": True,
               "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__9": {"type": "ply", "filename": "meshes/Cylinder_003.ply", "face_normals": True,
               "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__10": {"type": "ply", "filename": "meshes/Cylinder_004.ply", "face_normals": True,
                "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__11": {"type": "ply", "filename": "meshes/Cylinder_005.ply", "face_normals": True,
                "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__12": {"type": "ply", "filename": "meshes/Cylinder_006.ply", "face_normals": True,
                "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__13": {"type": "ply", "filename": "meshes/Plane_001.ply", "face_normals": True,
                "bsdf": {"type": "ref", "id": "default-bsdf"}},
    "elm__14": {"type": "ply", "filename": "meshes/Plane_002.ply", "face_normals": True,
                "bsdf": {"type": "ref", "id": "default-bsdf"}},

    "default-bsdf": {"type": "twosided", "bsdf": {"type": "diffuse"}}
}

# --- RENDER LOOP ---
print("Loading scene...")
scene = mi.load_dict(scene_dict)

# SETTINGS
target_spp = 4096
spp_per_pass = 64
total_passes = target_spp // spp_per_pass
clamp_value = 100.0  # <--- Caps max brightness to prevent fireflies from ruining the average

print(f"Starting batched render: {total_passes} passes of {spp_per_pass} SPP.")
print(f"Total SPP: {target_spp}. Clamping spikes at {clamp_value}.")

film_size = scene.sensors()[0].film().crop_size()
acc_buffer = dr.zeros(mi.TensorXf, shape=(film_size[1], film_size[0], 3))


start_time = time.time()

for i in range(total_passes):
    # Render
    img = mi.render(scene, seed=i, spp=spp_per_pass)

    # Accumulate
    acc_buffer += img

    # Stats
    elapsed = time.time() - start_time
    avg_per_pass = elapsed / (i + 1)
    remaining_passes = total_passes - (i + 1)
    print(f"Pass {i + 1}/{total_passes} complete. ETA: {remaining_passes * avg_per_pass / 60:.2f} mins")

    # Save previews: early passes more frequently, then every 50
    pass_num = i + 1
    save_early = pass_num in [1, 2, 5, 10, 20, 50]
    save_periodic = pass_num > 50 and pass_num % 50 == 0

    if save_early or save_periodic:
        temp_img = acc_buffer / pass_num
        mi.util.write_bitmap(str(EXR_DIR / f"preview_{pass_num}_spp.exr"), temp_img)
        print(f"  -> Saved preview at pass {pass_num}")

# Final Average
final_image = acc_buffer / total_passes

# Downscale to 1920x1080
final_np = np.array(final_image)
target_w, target_h = 1920, 1080
channels = [Image.fromarray(final_np[:,:,c], mode='F').resize((target_w, target_h), Image.LANCZOS) for c in range(3)]
downscaled_np = np.stack([np.array(ch) for ch in channels], axis=-1)
final_image_downscaled = mi.TensorXf(downscaled_np)

print("Saving Final Beauty Render...")
mi.util.write_bitmap(str(EXR_DIR / "render_beauty_final.exr"), final_image_downscaled)

print("Done. Clean and clamped.")
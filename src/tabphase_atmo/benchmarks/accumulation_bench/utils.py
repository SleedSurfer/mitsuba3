import mitsuba as mi
import numpy as np
from pathlib import Path

# --- PATH LOGIC ---
# Go up: utils -> core -> atmospheric -> src -> PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = PROJECT_ROOT / "assets"


def get_asset_path(scene_name, asset_type="meshes"):
    """
    Tries to find assets in project/assets.
    Falls back to Desktop if not found (legacy support).
    """
    # 1. Check Project Assets
    path = ASSETS_ROOT / scene_name / asset_type
    if path.exists():
        return path

    # 2. Check Legacy Desktop Path
    legacy = Path.home() / "Desktop" / "Mitsuba3-scenes" / scene_name / asset_type
    if legacy.exists():
        return legacy

    print(f"[Utils] Warning: Could not find assets for {scene_name}/{asset_type}")
    return path


# --- PHASE WRAPPER LOADER ---
def get_phase_plugin(radius_mean=3.95, radius_std=0.03,num_angles=4096,num_wavelengths=64, note="default"):
    """
    Dynamically loads the wrapper.py to generate the .bin table.
    Falls back to HG if wrapper is missing.
    """
    try:
        # Assumes src/atmospheric/phase/wrapper.py exists
        from src.core import create_atmospheric_phase
        return create_atmospheric_phase(
            radius_mean_um=radius_mean,
            radius_std_um=radius_std,
            num_angles=num_angles,
            num_wavelengths=num_wavelengths,
            note=note,
            force_regen=False
        )
    except ImportError:
        print("[Utils] Wrapper not found or import failed. Using HG fallback.")
        return {"type": "hg", "g": 0.8}


# --- MATH & NOISE ---
def get_exact_blender_camera(loc, rot_deg):
    rx, ry, rz = np.radians(rot_deg)
    mat_x = np.array([[1, 0, 0, 0], [0, np.cos(rx), -np.sin(rx), 0], [0, np.sin(rx), np.cos(rx), 0], [0, 0, 0, 1]])
    mat_y = np.array([[np.cos(ry), 0, np.sin(ry), 0], [0, 1, 0, 0], [-np.sin(ry), 0, np.cos(ry), 0], [0, 0, 0, 1]])
    mat_z = np.array([[np.cos(rz), -np.sin(rz), 0, 0], [np.sin(rz), np.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    rot_mat = mat_z @ mat_y @ mat_x
    blender_matrix = rot_mat.copy()
    blender_matrix[:3, 3] = loc

    # Coordinate fixes
    init_rot = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    basis_change = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    return mi.ScalarTransform4f(basis_change @ (blender_matrix @ init_rot))


def make_disk_transform(flat_matrix):
    return mi.ScalarTransform4f([flat_matrix[i:i + 4] for i in range(0, 16, 4)])


def generate_cloud_grid(res=32, seed=1337, contrast=2.5, fine_amp=0.2, scale=0.3, density=1.0, base=0.05, clamp=None):
    rng = np.random.default_rng(seed)
    raw_noise = rng.random((res, res, res))
    contrast_noise = np.power(raw_noise, contrast)
    fine_noise = rng.random((res, res, res)) * fine_amp
    combined = contrast_noise * 0.8 + fine_noise * 0.2

    d = combined * scale * density + base
    if clamp is not None:
        d = np.minimum(d, clamp)

    return mi.VolumeGrid(mi.TensorXf(d))
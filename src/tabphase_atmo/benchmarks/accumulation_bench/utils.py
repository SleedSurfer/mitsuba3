import mitsuba as mi
import numpy as np
from pathlib import Path
import scipy.ndimage as nd
import math
from datetime import datetime

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


def create_sunsky_emitter(
        sun_elevation_deg,
        use_direct_vector=True,
        is_afternoon=True,
        latitude=49.195,  # Defaulted to Brno
        longitude=16.606,
        timezone=2.0,  # CEST
        year=2026,
        month=4,
        day=14,
        turbidity=2.5,
        sun_scale=1.0,
        sky_scale=1.0
):
    """
    Creates a highly reusable Mitsuba sunsky dict.
    If use_direct_vector=True, bypasses time conversion to keep the sun perfectly
    aligned to the Y/Z plane (crucial for camera alignment in atmospheric renders).
    """

    if use_direct_vector:
        elev_rad = math.radians(max(0.001, sun_elevation_deg))
        sun_direction = [0.0, math.cos(elev_rad), math.sin(elev_rad)]

        return {
            "type": "sunsky",
            "sun_direction": sun_direction,
            "turbidity": turbidity,
            "sun_scale": sun_scale,
            "sky_scale": sky_scale,
            "sun_aperture": 0.1,
        }

    # --- GEOGRAPHICAL TIME SOLVER ---
    # 1. Get Day of Year
    date_obj = datetime(year, month, day)
    day_of_year = date_obj.timetuple().tm_yday

    # 2. Celestial Math
    lat_rad = math.radians(latitude)
    elev_rad = math.radians(sun_elevation_deg)

    # Approximate Solar Declination
    declination = math.radians(-23.44 * math.cos(math.radians((360.0 / 365.0) * (day_of_year + 10))))

    # Solve for Hour Angle
    numerator = math.sin(elev_rad) - (math.sin(lat_rad) * math.sin(declination))
    denominator = math.cos(lat_rad) * math.cos(declination)
    cos_h = numerator / denominator

    # Clamp to prevent math domain errors if you ask for an angle the sun literally cannot reach
    cos_h = max(-1.0, min(1.0, cos_h))

    h_rad = math.acos(cos_h)
    h_hours = math.degrees(h_rad) / 15.0

    # Calculate actual time (offsetting from solar noon ~12:00)
    decimal_time = 12.0 + h_hours if is_afternoon else 12.0 - h_hours

    hour = int(decimal_time)
    remainder = (decimal_time - hour) * 60.0
    minute = int(remainder)
    second = (remainder - minute) * 60.0

    return {
        "type": "sunsky",
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "turbidity": turbidity,
        "sun_scale": sun_scale,
        "sky_scale": sky_scale
    }

def generate_cloud_grid(res=64, seed=1337, radius=0.45, density_multiplier=50.0):
    rng = np.random.default_rng(seed)

    grid_1d = np.linspace(0, 1, res)
    X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

    dist_from_center = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)

    base_shape = np.clip((radius - dist_from_center) / radius, 0, 1)

    noise_res = res // 4  # Smaller grid = bigger cloud chunks
    raw_noise = rng.random((noise_res, noise_res, noise_res))

    smooth_noise = nd.zoom(raw_noise, res / noise_res, order=3)
    smooth_noise = np.clip(smooth_noise, 0, 1)
    cloud_density = base_shape - (smooth_noise * 0.6)

    cloud_density = np.clip(cloud_density, 0, 1)

    cloud_density *= density_multiplier

    return mi.VolumeGrid(mi.TensorXf(cloud_density.astype(np.float32)))


import numpy as np
import scipy.ndimage as nd
import mitsuba as mi


def generate_cirrus_grid(res=128, seed=420, density_multiplier=2.0, wind_stretch=6.0):
    rng = np.random.default_rng(seed)

    # 1. Base Grid (from -1 to 1 for easier center distance math)
    grid_1d = np.linspace(-1, 1, res)
    X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

    # 2. Soft Edge Mask
    # We stretch the Y-axis of the falloff too, so the boundary is an elongated oval
    # instead of a hard sphere, preventing the edges from looking like a Minecraft block.
    dist_from_center = np.sqrt(X ** 2 + (Y / wind_stretch) ** 2 + (Z * 3) ** 2)
    base_mask = np.clip(1.0 - dist_from_center, 0, 1)

    # 3. Anisotropic Fractal Noise
    # Instead of one uniform noise block, we layer two different resolutions.
    # Notice the Y-resolution is much smaller before scaling up—this creates the "streaks".

    # Octave 1: Big structural wind streaks
    res_o1 = [res // 4, max(2, res // int(4 * wind_stretch)), res // 4]
    noise_o1 = rng.random(res_o1)
    smooth_o1 = nd.zoom(noise_o1, [res / res_o1[0], res / res_o1[1], res / res_o1[2]], order=3)

    # Octave 2: Finer, broken up details
    res_o2 = [res // 2, max(4, res // int(2 * wind_stretch)), res // 2]
    noise_o2 = rng.random(res_o2)
    smooth_o2 = nd.zoom(noise_o2, [res / res_o2[0], res / res_o2[1], res / res_o2[2]], order=3)

    # Combine octaves
    final_noise = (smooth_o1 * 0.7) + (smooth_o2 * 0.3)

    # 4. Carve and Sculpt
    # Squaring the noise makes the peaks sharper and the valleys emptier (wispy look)
    cloud_density = base_mask * (final_noise ** 2)

    # Subtract a threshold to kill off the low-density fuzz, leaving only the streaks
    cloud_density = np.clip(cloud_density - 0.15, 0, 1)

    cloud_density *= density_multiplier

    # Mitsuba gets extremely pissed off if you feed it float64 arrays
    return mi.VolumeGrid(mi.TensorXf(cloud_density.astype(np.float32)))


def generate_cumulus_congestus_grid(res=128, seed=42, radius=0.45, width_stretch=1.3, height_stretch=0.7, density_multiplier=1.0):
    rng = np.random.default_rng(seed)

    # 1. Setup the 3D grid
    grid_1d = np.linspace(0, 1, res)
    X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

    center_x, center_z = 0.5, 0.5
    base_y = 0.20  # Dropped slightly to give the top more room to tower

    # 2. Build the Tapered Base Shape
    horiz_dist = np.sqrt(((X - center_x) / width_stretch) ** 2 + ((Z - center_z) / width_stretch) ** 2)
    vertical_taper = np.clip(1.0 - ((Y - base_y) / height_stretch), 0.0, 1.0)
    base_shape = np.clip((vertical_taper * radius - horiz_dist) / radius, 0.0, 1.0)

    # Hard flat bottom
    bottom_mask = np.clip((Y - base_y) * 20.0, 0.0, 1.0)
    base_shape *= bottom_mask

    # 3. The Cauliflower Generator
    def get_noise_octave(octave_res, amplitude):
        raw = rng.random((octave_res, octave_res, octave_res)) * 2.0 - 1.0
        smooth = nd.zoom(raw, res / octave_res, order=3)
        return smooth * amplitude

    # Fixed octave resolutions to ensure large, chunky billows regardless of grid res
    noise = np.abs(get_noise_octave(8, 0.6)) + \
            np.abs(get_noise_octave(16, 0.25)) + \
            np.abs(get_noise_octave(32, 0.1))

    # 4. Carve the billows into the base shape
    cloud_density = base_shape - (noise * 0.7)
    cloud_density = np.clip(cloud_density * 2.5, 0.0, 1.0)
    cloud_density *= density_multiplier

    return mi.VolumeGrid(mi.TensorXf(cloud_density.astype(np.float32)))


def generate_cumulus_grid(res=128, seed=42, radius=0.45, width_stretch=1.2, height_stretch=1.0, density_multiplier=1.0):
    rng = np.random.default_rng(seed)

    grid_1d = np.linspace(0, 1, res)
    X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

    center_x, center_z = 0.5, 0.5
    center_y = 0.35

    # ✅ THE FIX: Parameters restored
    dist = np.sqrt(((X - center_x) * width_stretch)**2 +
                   ((Y - center_y) * height_stretch)**2 +
                   ((Z - center_z) * width_stretch)**2)

    base_shape = np.clip((radius - dist) / radius, 0, 1)

    bottom_mask = np.clip((Y - 0.25) * 15.0, 0, 1)
    base_shape *= bottom_mask

    gradient = np.clip(1.0 - ((Y - 0.25) / 0.6), 0, 1)

    def get_noise_octave(octave_res, amplitude):
        raw = rng.random((octave_res, octave_res, octave_res))
        smooth = nd.zoom(raw, res / octave_res, order=3)
        return smooth * amplitude

    octave_low = get_noise_octave(12, 0.6)

    # Octave 2: Mid-level complexity/wisps (24 features).
    octave_mid = get_noise_octave(24, 0.3)

    # Octave 3: High-frequency surface fuzz (48 features).
    octave_high = get_noise_octave(48, 0.15)

    # Combine them (make sure to use np.abs() if you want the bubbly look)
    noise = np.abs(octave_low) + np.abs(octave_mid) + np.abs(octave_high)

    cloud_density = (base_shape * gradient) - (noise ** 2 * 0.6)
    cloud_density = np.clip(cloud_density - 0.05, 0, 1)
    cloud_density *= density_multiplier


    visualize_grid_seeds(cloud_density)
    return mi.VolumeGrid(mi.TensorXf(cloud_density.astype(np.float32)))

def assemble_volume_dict(density_grid, phase_dict, bounds=(50.0, 50.0, 2.0), albedo=0.99, density_scale=10.0):
    # The grid natively spans [0, 1].
    # To map it to the cube's [-bounds, bounds] space, we scale it by 2x the bounds,
    # and then translate it backwards by half that scale to center it.
    grid_transform = (
        mi.ScalarTransform4f.translate([-bounds[0], -bounds[1], -bounds[2]]) @
        mi.ScalarTransform4f.scale([bounds[0] * 2.0, bounds[1] * 2.0, bounds[2] * 2.0])
    )

    return {
        "type": "cube",
        "to_world": mi.ScalarTransform4f.scale(bounds),
        "bsdf": {"type": "null"},
        "interior": {
            "type": "heterogeneous",
            "sigma_t": {
                "type": "gridvolume",
                "grid": density_grid,
                "filter_type": "trilinear",
                "to_world": grid_transform,
                "wrap_mode": "repeat"
            },
            "scale": density_scale,
            "albedo": albedo,
            "phase":phase_dict##phase_dict,"
        },
    }


import matplotlib.pyplot as plt
import numpy as np


def visualize_grid_seeds(mi_grid):
    # VolumeGrid stores the data in the .data member, not a .tensor() method
    # We cast to np.array to pull it from the GPU/DrJit into CPU RAM
    data = np.array(mi_grid.data)

    # If it's a 4D tensor (e.g. channel-first or channel-last),
    # we need to squeeze it down to 3D for plotting
    if len(data.shape) == 4:
        # Usually (channels, z, y, x) or (z, y, x, channels)
        # We'll just take the first channel
        data = data[0] if data.shape[0] < data.shape[-1] else data[..., 0]

    res = data.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Voxel Seed Analysis (Res: {res})", fontsize=16)

    mid = res // 2

    # Using 'magma' or 'bone' helps see the density gradients better
    axes[0].imshow(data[mid, :, :], origin='lower', cmap='magma')
    axes[0].set_title("Side View (YZ)")

    axes[1].imshow(data[:, mid, :], origin='lower', cmap='magma')
    axes[1].set_title("Top-Down (XZ)")

    axes[2].imshow(data[:, :, mid], origin='lower', cmap='magma')
    axes[2].set_title("Front View (XY)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def create_sun_aligned_camera(sun_elevation_deg, distance=5.0):
    """
    Generates a Z-up camera transform perfectly aligned with the direct-vector sun.
    Places the camera beneath the origin, looking up the Y/Z slope.
    """
    sun_elev_rad = math.radians(sun_elevation_deg)

    sun_dir_y = math.cos(sun_elev_rad)
    sun_dir_z = math.sin(sun_elev_rad)

    cam_origin = [
        0.0,
        -distance * sun_dir_y,
        -distance * sun_dir_z
    ]

    # Returns the fully baked Mitsuba transform object
    return mi.ScalarTransform4f.look_at(
        origin=cam_origin,
        target=[0.0, 0.0, 0.0],
        up=(0.0, 0.0, 1.0)
    )
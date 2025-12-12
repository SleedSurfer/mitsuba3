import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import importlib

mi.set_variant("llvm_ad_spectral")

# ---------------------------------------------------------
# Phase Function Loader (Same as before)
# ---------------------------------------------------------
from pathlib import Path as _Path

WRAPPER_ABS_PATH = _Path("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/wrapper.py").resolve()

if not WRAPPER_ABS_PATH.exists():
    print(f"Warning: Wrapper file not found at {WRAPPER_ABS_PATH}")
else:
    PACKAGE_ROOT = str(WRAPPER_ABS_PATH.parent.parent)
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)

try:
    mod = importlib.import_module("python.atmospheric.wrapper")
    create_atmospheric_phase = getattr(mod, "create_atmospheric_phase")
    PHASE_PLUGIN_DICT = create_atmospheric_phase(
        radius_mean_um=150.0,
        radius_std_um=10.0,
        num_angles=1024,
        num_wavelengths=32,
        note="cornell_rainbow",
        force_regen=False
    )
except ImportError:
    print("Using standard HG phase fallback.")
    PHASE_PLUGIN_DICT = {"type": "hg", "g": 0.8}

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "atmo_phase_scan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RES_W, RES_H = 512, 512
SPP = 1024
FIXED_DEPTH = 4  # Keep at 4, but if rainbow is faint, try 1 later


def build_scene(phase, sun_vector, sigma_t=0.8):
    """
    Constructs the scene with a specific sun direction vector.
    """
    # Camera: Eye level (Y=2), Looking straight forward (-Z)
    camera_origin = (0, 2.0, 5.0)
    camera_target = (0, 2.0, 0.0)

    scene = {
        "type": "scene",
        "integrator": {
            "type": "volpathmis",
            "max_depth": FIXED_DEPTH,
            "rr_depth": 1000,
        },
        "sensor": {
            "type": "perspective",
            "fov": 55.0,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=camera_origin,
                target=camera_target,
                up=(0, 1, 0),
            ),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm",
                "width": RES_W, "height": RES_H,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"},
            },
        },
        # Ground: Dark neutral to avoid color contamination
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0, -0.01, 0)).rotate((1, 0, 0), -90).scale((500, 500, 1)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.2}}
        },
        # Marker Sphere
        "marker_sphere": {
            "type": "sphere",
            "center": (0.0, 0.5, 0.0), "radius": 0.2,
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": (0.8, 0.1, 0.1)}}
        },
        # The Interference Slab
        "air_box": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((0, 2.0, 0)).scale((4.0, 2.0, 0.5)),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": sigma_t,
                "albedo": 0.99,
                "phase": phase
            }
        },
        # Precise Sun Control
        "env": {
            "type": "sunsky",
            "sun_direction": sun_vector,  # Explicit vector
            "turbidity": 2.0,  # Low turbidity = clearer air = better rainbows
            "albedo": 0.2,  # Low ground albedo = higher contrast
            "sun_scale": 1.0,
            "sky_scale": 1.0,
            "sun_aperture": 0.5338  # Standard sun size
        }
    }
    return mi.load_dict(scene)


def render_and_save(scene, tag):
    img = mi.render(scene, spp=SPP, seed=0)
    dr.eval(img)
    # Save EXR
    exr_path = OUT_DIR / f"{tag}.exr"
    mi.util.write_bitmap(str(exr_path), img)
    return np.array(img, copy=False)


def plot_center_row(arr, tag, angle):
    row = arr[arr.shape[0] // 2, :, :3]
    x = np.arange(row.shape[0])

    plt.figure(figsize=(10, 4))
    plt.title(f"Scattering Intensity @ {angle}° — {tag}")
    plt.plot(x, row[:, 0], label="R", color='red', alpha=0.7)
    plt.plot(x, row[:, 1], label="G", color='green', alpha=0.7)
    plt.plot(x, row[:, 2], label="B", color='blue', alpha=0.7)
    plt.yscale("log")
    plt.ylim(1e-4, 100)  # Fix Y-axis to make comparing plots easier
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_plot.png")
    plt.close()  # Close to save memory


def get_sun_direction(angle_deg):
    """
    Calculates sun vector for a given angle relative to the view.
    Angle 0   = Sun BEHIND camera (Backscattering / Glory / Rainbow)
    Angle 90  = Sun to the RIGHT
    Angle 180 = Sun IN FRONT of camera (Forward scattering / Corona)
    """
    # Convert to radians
    # We rotate around Y axis.
    # Camera looks at -Z.
    # Angle 0 should be +Z (behind camera).
    # Angle 180 should be -Z (in front of camera).

    theta = np.deg2rad(angle_deg)

    # Simple circle in XZ plane, slightly elevated in Y to avoid perfect horizon issues
    elevation = np.deg2rad(15)  # Keep sun 15 degrees up so it hits the slab nicely

    x = np.sin(theta) * np.cos(elevation)
    y = np.sin(elevation)
    z = np.cos(theta) * np.cos(elevation)  # Z positive is behind camera

    return (x, y, z)


def main():
    # SCAN SETTINGS
    # We scan from 0 (Back) to 180 (Front) in 10-degree steps.
    # If you see a spike, you can narrow this down later.
    angles = range(0, 185, 5)

    print(f"--- STARTING PHASE FUNCTION SCAN ---")
    print(f"Scanning {len(angles)} angles from 0° (Back) to 180° (Front)...")

    for angle in angles:
        sun_vec = get_sun_direction(angle)
        tag = f"scan_angle_{angle:03d}"

        print(f"Rendering Angle {angle}°...")

        scene = build_scene(
            PHASE_PLUGIN_DICT,
            sun_vector=sun_vec,
            sigma_t=0.8  # Dense slab
        )

        arr = render_and_save(scene, tag)
        plot_center_row(arr, tag, angle)

    print("Scan complete. Check the 'atmo_phase_scan' folder.")


if __name__ == "__main__":
    main()
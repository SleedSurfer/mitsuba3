import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib

mi.set_variant("llvm_ad_spectral")
from pathlib import Path as _Path

WRAPPER_ABS_PATH = _Path("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/wrapper.py").resolve()
if not WRAPPER_ABS_PATH.exists():
    # Fallback for running without the specific wrapper
    PHASE_PLUGIN_DICT = {"type": "hg", "g": 0.8}
else:
    # Assuming the wrapper load logic works as in your snippet
    PACKAGE_ROOT = str(WRAPPER_ABS_PATH.parent.parent)
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    mod = importlib.import_module("python.atmospheric.wrapper")
    create_atmospheric_phase = getattr(mod, "create_atmospheric_phase")
    PHASE_PLUGIN_DICT = create_atmospheric_phase(
        radius_mean_um=100.0, radius_std_um=10.0, num_angles=2048,
        num_wavelengths=64, note="rainbow_proper", force_regen=False
    )

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "volpathmis_8192spp_rainbow_nosky_d2_wavelength64"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXR_DIR = OUT_DIR / "exr"
PNG_DIR = OUT_DIR / "png"
EXR_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

RES_W, RES_H = 1024, 512
SPP = 256
FOV = 100.0


def build_scene(phase, sun_dir, sigma_t=0.1, albedo=0.9, irradiance=50.0):
    scene = {
        "type": "scene",
        "integrator": {"type": "volpathmis", "max_depth": 4, "rr_depth": 2},
        "sensor": {
            "type": "perspective",
            "fov": float(FOV),
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0)
            ),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm", "width": RES_W, "height": RES_H,
                "pixel_format": "rgb", "rfilter": {"type": "box"},
            },
        },
        "sun": {
            "type": "directional",
            "direction": tuple(float(x) for x in sun_dir),
            "irradiance": {"type": "rgb", "value": float(irradiance)},
        },
        "boundary": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.scale(2.0),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": float(sigma_t),
                "albedo": float(albedo),
                "phase": phase
            },
        },
    }
    return mi.load_dict(scene)


def get_vertical_sun_vector(angle_deg):
    """
    Angle 0   = Sun Behind (Glory)
    Angle 180 = Sun In Front (The Sun itself)
    """
    rad = np.deg2rad(angle_deg)

    y = np.sin(rad)

    z = np.cos(rad)

    return (0.0, y, z)


def main():

    angles_to_scan = [
        10,  # Glory Region 0, 5,
        35, 40, 42, 45, 50,  # Rainbow Region (Expect Arch at ~42)
        180  # The Sun Itself
    ]

    print(f"--- SUN SCAN: {len(angles_to_scan)} Frames ---")
    print("If math holds: Angle 0 = Glory, Angle 42 = Rainbow Arch")

    for angle in angles_to_scan:
        sun_vec = get_vertical_sun_vector(angle)
        tag = f"{angle:03d}"
        print(f"Rendering Angle {tag}°...")

        scene = build_scene(PHASE_PLUGIN_DICT, sun_dir=sun_vec)
        img = mi.render(scene, spp=SPP, seed=0)
        dr.eval(img)

        mi.util.write_bitmap(str(EXR_DIR / f"scan_{tag}.exr"), img)

    print("Done. EXRs saved in:", EXR_DIR)


if __name__ == "__main__":
    main()
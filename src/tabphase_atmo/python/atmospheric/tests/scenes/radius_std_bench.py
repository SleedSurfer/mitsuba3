import mitsuba as mi
import drjit as dr
import numpy as np
from pathlib import Path
import sys
import importlib

mi.set_variant("llvm_ad_spectral")

# -----------------------------------------------------------------------------
# Phase plugin loader
# -----------------------------------------------------------------------------
from pathlib import Path as _Path
WRAPPER_ABS_PATH = _Path(
    "/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/wrapper.py"
).resolve()

if not WRAPPER_ABS_PATH.exists():
    raise RuntimeError("Atmospheric phase wrapper not found")

PACKAGE_ROOT = str(WRAPPER_ABS_PATH.parent.parent)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

mod = importlib.import_module("python.atmospheric.wrapper")
create_atmospheric_phase = getattr(mod, "create_atmospheric_phase")

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "radius_sweep_30std"
EXR_DIR = OUT_DIR / "exr"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXR_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Render config
# -----------------------------------------------------------------------------
RES_W, RES_H = 1024, 1024
SPP = 1024  # scanning -> keep cheaper; raise later
FOV = 120.0

RADIUS_MEAN_UM_SWEEP = [300.0] #[15.0, 30.0 , 50.0 , 70.0]
RADIUS_STD_SWEEP = [0.5, 2.0, 10.0, 15.0, 30.0]

ANGLES_TO_SCAN = [0,20,42]

# -----------------------------------------------------------------------------
# Sun direction logic (same as your working script)
# -----------------------------------------------------------------------------
def get_vertical_sun_vector(angle_deg):
    """
    Angle 0   = Sun Behind (Glory)
    Angle 180 = Sun In Front (Sun itself)
    """
    rad = np.deg2rad(angle_deg)
    y = np.sin(rad)
    z = np.cos(rad)
    return (0.0, -y, z)

# -----------------------------------------------------------------------------
# Scene construction
# -----------------------------------------------------------------------------
def build_scene(phase, sun_dir):
    scene = {
        "type": "scene",

        "integrator": {
            "type": "volpathmis",
            "max_depth": 2,
            "hide_emitters": True,
        },

        "sensor": {
            "type": "perspective",
            "fov": float(FOV),
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3),
                target=(0, 0, 0),
                up=(0, 1, 0),
            ),
            "sampler": {"type": "independent", "sample_count": SPP},
            "film": {
                "type": "hdrfilm",
                "width": RES_W,
                "height": RES_H,
                "pixel_format": "rgb",
                "component_format": "float32",
                "rfilter": {"type": "mitchell"},
            },
        },

        "sun": {
            "type": "directional",
            "direction": tuple(float(x) for x in sun_dir),
            "irradiance": {"type": "rgb", "value": 50.0},
        },

        "cloud_slab": {
            "type": "cube",
            "to_world": (
                mi.ScalarTransform4f.translate([0, 0, 0]) @
                mi.ScalarTransform4f.scale([10.5, 10.5, 1.5])
            ),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.4,
                "albedo": 0.88,
                "phase": phase,
            },
        },

        # "background": {
        #     "type": "rectangle",
        #     "to_world": (
        #         mi.ScalarTransform4f.translate([0, 0, -10.0]) @
        #         mi.ScalarTransform4f.scale([20.0, 20.0, 1.0])
        #     ),
        #     "bsdf": {
        #         "type": "diffuse",
        #         "reflectance": {"type": "rgb", "value": [0.55, 0.70, 1.00]}
        #     }
        # },

    }
    return mi.load_dict(scene)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    for radius in RADIUS_MEAN_UM_SWEEP:
        print(f"\n=== radius_um = {radius} µm ===")

        phase = create_atmospheric_phase(
            radius_mean_um=radius,
            radius_std_um=6.0,
            num_angles=8192,
            num_wavelengths=64,
            note=f"radius_{radius:.2f}",
            force_regen=False,
        )

        std_dir = EXR_DIR / f"radius_{radius:.2f}"
        std_dir.mkdir(parents=True, exist_ok=True)

        for angle in ANGLES_TO_SCAN:
            sun = get_vertical_sun_vector(angle)

            for tag, sdir in [("orig", sun)]: #("flip", sun_flip)
                fname = std_dir / f"scan_{angle:03d}_{tag}.exr"
                print(f"  Rendering angle {angle:03d} ({tag}) -> {fname.name}")

                scene = build_scene(phase, sdir)
                img = mi.render(scene, seed=0)
                dr.eval(img)
                mi.util.write_bitmap(str(fname), img)

    print("\nDone. EXRs saved in:", EXR_DIR)


if __name__ == "__main__":
    main()

import mitsuba as mi
import drjit as dr
import numpy as np
from pathlib import Path
from datetime import datetime
import importlib.util
import sys

mi.set_variant('llvm_ad_spectral')

# Load the wrapper module from an absolute filesystem path to avoid import-root issues.
# Strict absolute-path behavior: add the package root (the `python/` folder) to sys.path
# and import the package module so relative imports inside `wrapper.py` work.
from pathlib import Path as _Path
WRAPPER_ABS_PATH = _Path("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/wrapper.py").resolve()
if not WRAPPER_ABS_PATH.exists():
    raise ImportError(f"Wrapper file not found at absolute path: {WRAPPER_ABS_PATH}")

# The package root is the parent of the `atmospheric` folder (i.e. the `python/` dir)
PACKAGE_ROOT = str(WRAPPER_ABS_PATH.parent.parent)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

try:
    import importlib
    mod = importlib.import_module("python.atmospheric.wrapper")
    create_atmospheric_phase = getattr(mod, "create_atmospheric_phase")
except Exception as e:
    raise ImportError(f"Failed to import wrapper module from {WRAPPER_ABS_PATH}: {e}")

# --------------------------------------------------------------------
# User config
# --------------------------------------------------------------------
OUT_DIR = Path("atmo_ladder_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPP_LIST = [1, 4, 16, 64, 256]
RES = 256

PHASE_PLUGIN_DICT = create_atmospheric_phase(
        radius_mean_um=150.0,
        radius_std_um=10.0,
        num_angles=1024, num_wavelengths=32,
        note="cornell_rainbow", force_regen=False
    )


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def write_exr(path: Path, img):
    mi.util.write_bitmap(str(path), img)

def image_stats(img):
    arr = np.array(img, copy=False)
    # If rendered RGB: shape (H,W,3)
    finite = np.isfinite(arr)
    if not finite.all():
        nan_count = np.size(arr) - np.count_nonzero(finite)
    else:
        nan_count = 0
    # luminance-ish stats
    if arr.ndim == 3 and arr.shape[2] >= 3:
        y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    else:
        y = arr
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {
            "nan_count": nan_count, "mean": float("nan"), "max": float("nan"),
            "p99": float("nan"), "p999": float("nan")
        }
    return {
        "nan_count": int(nan_count),
        "mean": float(np.mean(stats_clip(y))),
        "max": float(np.max(y)),
        "p99": float(np.quantile(y, 0.99)),
        "p999": float(np.quantile(y, 0.999)),
    }

def stats_clip(y, clip_max=1e6):
    # protect quantiles from explosive outliers affecting float ops
    return np.clip(y, -clip_max, clip_max)

def T(flat_list):
    # Convenience helper like your snippet
    return mi.ScalarTransform4f(mi.ScalarMatrix4f(flat_list))


# --------------------------------------------------------------------
# Minimal test scenes (do not inherit any content from your Cornell)
# --------------------------------------------------------------------
def make_scene_test1_directional_single_scatter_only(phase):
    """
    Test 1: Homogeneous medium only + directional emitter + no surfaces that interact.
    The only geometry is a null boundary box to define an interior medium region.
    """
    scene = {
        "type": "scene",
        "integrator": {
            "type": "volpathmis",
            "max_depth": 2,   # single medium scatter + emitter eval
            "rr_depth": 100,  # disable RR for determinism here
        },
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3),
                target=(0, 0, 0),
                up=(0, 1, 0),
            ),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm",
                "width": RES, "height": RES,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            },
        },
        # Directional "sun"
        "sun": {
            "type": "directional",
            "direction": (0.2, -0.6, -1.0),
            "irradiance": {
                "type": "rgb",
                "value": 1.0
            }
        },
        # Define a medium region using a null boundary cube.
        "boundary": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.scale(2.0),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.5,
                "albedo": 0.99,
                "phase": phase
            }
        }
    }
    return mi.load_dict(scene)


def make_scene_test2_add_diffuse_plane(phase):
    """
    Test 2: Add a single diffuse plane behind the medium, still directional emitter.
    This checks whether surface interactions/occlusion start causing variance.
    """
    scene = {
        "type": "scene",
        "integrator": {
            "type": "volpathmis",
            "max_depth": 4,
            "rr_depth": 100,
        },
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3),
                target=(0, 0, 0),
                up=(0, 1, 0),
            ),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm",
                "width": RES, "height": RES,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            },
        },
        "sun": {
            "type": "directional",
            "direction": (0.2, -0.6, -1.0),
            "irradiance": {"type": "rgb", "value": 1.0}
        },
        "boundary": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.scale(2.0),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.5,
                "albedo": 0.99,
                "phase": phase
            }
        },
        "plane": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0, 0, -1.2)).scale((3, 3, 1)),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": 0.7}
            }
        }
    }
    return mi.load_dict(scene)


def make_scene_test3_area_emitter_no_glass(phase):
    """
    Test 3: Replace directional emitter with a finite area emitter (harder NEE),
    still no specular objects.
    """
    scene = {
        "type": "scene",
        "integrator": {
            "type": "volpathmis",
            "max_depth": 8,
            "rr_depth": 8,
        },
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3),
                target=(0, 0, 0),
                up=(0, 1, 0),
            ),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm",
                "width": RES, "height": RES,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            },
        },
        "boundary": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.scale(2.0),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.5,
                "albedo": 0.99,
                "phase": phase
            }
        },
        "light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0, 1.0, 0.5)).scale((0.3, 0.3, 1)),
            "bsdf": {"type": "null"},
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": 10.0}
            }
        },
        "plane": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0, 0, -1.2)).scale((3, 3, 1)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.7}}
        }
    }
    return mi.load_dict(scene)


def make_scene_test4_area_emitter_with_glass_sphere(phase):
    """
    Test 4: Same as Test 3, but add a single dielectric sphere (the suspected variance amplifier).
    """
    scene = {
        "type": "scene",
        "integrator": {
            "type": "volpathmis",
            "max_depth": 8,
            "rr_depth": 8,
        },
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3),
                target=(0, 0, 0),
                up=(0, 1, 0),
            ),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm",
                "width": RES, "height": RES,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"}
            },
        },
        "boundary": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.scale(2.0),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": 0.5,
                "albedo": 0.99,
                "phase": phase
            }
        },
        "light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0, 1.0, 0.5)).scale((0.3, 0.3, 1)),
            "bsdf": {"type": "null"},
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": 10.0}
            }
        },
        "plane": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0, 0, -1.2)).scale((3, 3, 1)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.7}}
        },
        "sphere": {
            "type": "sphere",
            "center": (0.2, -0.2, 0.0),
            "radius": 0.35,
            "bsdf": {
                "type": "dielectric",
                "int_ior": 1.5,
                "ext_ior": 1.0
            }
        },
    }
    return mi.load_dict(scene)


def render_scene(scene, name: str):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for spp in SPP_LIST:
        print(f"[{name}] rendering spp={spp} ...")
        img = mi.render(scene, spp=spp, seed=0)
        dr.eval(img)

        stats = image_stats(img)
        print(f"[{name}] spp={spp} stats: {stats}")

        out = OUT_DIR / f"{stamp}_{name}_spp{spp}.exr"
        write_exr(out, img)
        print(f"[{name}] wrote {out}")


def main(phase):
    scenes = [
        ("test1_directional_single_scatter_only", make_scene_test1_directional_single_scatter_only(phase)),
        ("test2_add_diffuse_plane",               make_scene_test2_add_diffuse_plane(phase)),
        ("test3_area_emitter_no_glass",           make_scene_test3_area_emitter_no_glass(phase)),
        ("test4_area_emitter_with_glass_sphere",  make_scene_test4_area_emitter_with_glass_sphere(phase)),
    ]

    for name, scene in scenes:
        print(f"\n=== Running {name} ===")
        render_scene(scene, name)


if __name__ == "__main__":
    # You can replace this with your create_atmospheric_phase(...) builder output.
    # Keep it as a dict or already-loaded plugin object. Both work as 'phase' value in load_dict.
    assert PHASE_PLUGIN_DICT is not None, "Set PHASE_PLUGIN_DICT at the top of this file."
    main(PHASE_PLUGIN_DICT)
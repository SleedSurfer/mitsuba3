import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

mi.set_variant("llvm_ad_spectral")

# ---------------------------------------------------------
# Phase instantiation: same convention as your ladder script
# ---------------------------------------------------------
from pathlib import Path as _Path
WRAPPER_ABS_PATH = _Path("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/atmospheric/wrapper.py").resolve()
if not WRAPPER_ABS_PATH.exists():
    raise ImportError(f"Wrapper file not found at absolute path: {WRAPPER_ABS_PATH}")

PACKAGE_ROOT = str(WRAPPER_ABS_PATH.parent.parent)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

try:
    import importlib
    mod = importlib.import_module("python.atmospheric.wrapper")
    create_atmospheric_phase = getattr(mod, "create_atmospheric_phase")
except Exception as e:
    raise ImportError(f"Failed to import wrapper module from {WRAPPER_ABS_PATH}: {e}")

PHASE_PLUGIN_DICT = create_atmospheric_phase(
    radius_mean_um=150.0,
    radius_std_um=10.0,
    num_angles=1024,
    num_wavelengths=32,
    note="cornell_rainbow",
    force_regen=False
)

# ---------------------------------------------------------
# Output + render config
# ---------------------------------------------------------
HERE = Path(__file__).resolve().parent  # assume same dir as ladder script
OUT_DIR = HERE / "atmo_rainbow_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RES_W = 512
RES_H = 256
SPP = 4096  # start high to see the curve; tune down later


def build_scene(phase, irradiance=2e3, sigma_t=0.15, albedo=0.99,
                max_depth=2, rr_depth=1000,
                sun_direction=(0.0, 0.2, -1.0),
                fov=70):
    """
    Rainbow diagnostic scene:
      - Directional light ("sun")
      - Homogeneous medium in a null-boundary box
      - Camera looks into the box; horizontal FOV causes view direction sweep
        -> sweeping scattering angle wrt sun direction.

    Keep geometry intentionally minimal (no surfaces / no glass).
    """
    scene = {
        "type": "scene",
        "integrator": {
            "type": "volpathmis",
            "max_depth": int(max_depth),
            "rr_depth": int(rr_depth),
        },
        "sensor": {
            "type": "perspective",
            "fov": float(fov),
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3),
                target=(0, 0, 0),
                up=(0, 1, 0),
            ),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm",
                "width": int(RES_W),
                "height": int(RES_H),
                "pixel_format": "rgb",
                "rfilter": {"type": "box"},
            },
        },
        "sun": {
            "type": "directional",
            "direction": tuple(float(x) for x in sun_direction),
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


def scattering_angle_per_pixel(scene, sun_direction):
    """
    Approximate scattering angle θ(x) for center-row pixels:
    - sample primary camera ray directions per pixel
    - compare view direction (~ -ray.d) with sun direction

    If sensor.sample_ray() signature doesn't match, return None.
    """
    sensor = scene.sensors()[0]
    film = sensor.film()
    w, h = film.size()

    y = (h - 1) * 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.full_like(xs, y + 0.5)

    pos = mi.Point2f(xs / w, ys / h)
    time = 0.0
    wav_sample = 0.5
    aperture = mi.Point2f(0.5, 0.5)

    try:
        ray, _ = sensor.sample_ray(time, wav_sample, pos, aperture)
        dirs = ray.d.numpy()
    except Exception:
        return None

    # Normalize and shape dirs into (N, 3) reliably across API variants
    dirs = np.asarray(dirs, dtype=np.float32)

    if dirs.ndim == 1:
        # single vector
        if dirs.size == 3:
            dirs = dirs.reshape(1, 3)
        else:
            return None
    elif dirs.ndim == 2:
        # If second axis is 3, we already have (N, 3)
        if dirs.shape[1] == 3:
            pass
        # If first axis is 3 and second isn't, assume (3, N) and transpose
        elif dirs.shape[0] == 3 and dirs.shape[1] != 3:
            dirs = dirs.T
        else:
            # Try to make a best-effort reshape if one dimension equals 3
            if dirs.shape[0] == 3:
                dirs = dirs.T
            elif dirs.shape[1] == 3:
                pass
            else:
                return None
    else:
        return None

    # Ensure final layout is (N, 3)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        return None

    sun_dir = np.array(sun_direction, dtype=np.float32)
    sun_norm = np.linalg.norm(sun_dir)
    if sun_norm == 0:
        return None
    sun_dir = sun_dir / sun_norm

    # view_dir: camera looks along -ray.d
    view_dir = -dirs

    # If view_dir somehow has shape (3, N), transpose to (N, 3)
    if view_dir.ndim == 2 and view_dir.shape[1] != 3 and view_dir.shape[0] == 3:
        view_dir = view_dir.T

    # Safe normalization: avoid divide-by-zero by replacing zero norms with 1.0
    norms = np.linalg.norm(view_dir, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    view_dir = view_dir / norms

    # Dot product per-row -> shape (N,)
    # Use dot for robust shape handling
    try:
        cos_theta = np.dot(view_dir, sun_dir)
    except ValueError:
        # As a fallback, try transposed layout
        cos_theta = np.dot(view_dir.T, sun_dir)
        # If this succeeds, ensure cos_theta is shaped (N,) by flattening
        cos_theta = np.asarray(cos_theta).ravel()

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))
    return theta


def run():
    # Dial these to get a visible signal (not black, not blown out)
    cfg = dict(
        irradiance=2e3,
        sigma_t=0.15,
        albedo=0.99,
        max_depth=2,
        rr_depth=1000,
        sun_direction=(0.0, 0.2, -1.0),
        fov=70
    )

    scene = build_scene(PHASE_PLUGIN_DICT, **cfg)

    img = mi.render(scene, spp=int(SPP), seed=0)
    dr.eval(img)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exr_path = OUT_DIR / (
        f"{stamp}_rainbow_diag_spp{SPP}"
        f"_st{cfg['sigma_t']}_alb{cfg['albedo']}_E{cfg['irradiance']}"
        f"_d{cfg['max_depth']}.exr"
    )
    mi.util.write_bitmap(str(exr_path), img)
    print(f"Wrote {exr_path}")

    arr = np.array(img, copy=False)
    row = arr[arr.shape[0] // 2, :, :3]

    theta = scattering_angle_per_pixel(scene, cfg["sun_direction"])
    if theta is None:
        x = np.arange(row.shape[0])
        xlabel = "pixel x"
    else:
        x = theta
        xlabel = "approx scattering angle θ (deg)"

    plt.figure(figsize=(12, 6))
    plt.title("Center-row RGB vs angle (rainbow diagnostic)")
    plt.plot(x, row[:, 0], label="R")
    plt.plot(x, row[:, 1], label="G")
    plt.plot(x, row[:, 2], label="B")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("radiance (log)")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()

    png_path = OUT_DIR / f"{stamp}_rainbow_diag_curve_spp{SPP}.png"
    plt.tight_layout()
    plt.savefig(png_path)
    print(f"Wrote {png_path}")
    plt.show()


if __name__ == "__main__":
    run()
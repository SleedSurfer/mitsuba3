import os
import shutil
import numpy as np

# Nimbus Imports moved here for caching
from nimbuscore.core.config import Particle, BackendType, HexagonalHabit, HexagonalComposition
from nimbuscore import create_atmospheric_phase

from scene import build_scene
import mitsuba as mi

mi.set_variant("llvm_spectral")
FPS = 12
OUTPUT_DIR = "render_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Master settings
CONFIG = {
    'target_spp': 20*32,
    'res_w': 1920,
    'res_h': 1080,
    'max_depth': 3
}


def get_frame_state(frame: int):
    """
    Returns: (sun_elev, cam_azim, turbulence, skip_render)
    """
    skip_render = False
    turbulence = 0.8
    mode = 'ice'
    radius = 0.0  # Only used if mode == 'mist'
    variance = 0.0  # Only used if mode == 'mist'

    # ---------------------------------------------------------
    # PHASE 1: Sunrise Sweep (Frames 0 - 47) | 2° to 24°
    # ---------------------------------------------------------
    if frame < 48:
        progress = frame / 47.0
        sun_elev = np.interp(progress, [0, 1], [2.0, 24.0])
        cam_azim = 180.0

    # ---------------------------------------------------------
    # PHASE 2: Turbulence Flex (Frames 48 - 95) | Frozen at 24°
    # ---------------------------------------------------------
    elif frame < 96:
        sun_elev = 24.0
        cam_azim = 180.0

        # 4 hard cuts, changing every 12 frames (1 second)
        turb_states = [0.5, 0.1, 2.0, 1.0]
        state_idx = (frame - 48) // 12
        turbulence = turb_states[state_idx]

        # Render the FIRST frame of each state, copy the rest
        if (frame - 48) % 12 != 0:
            skip_render = True

    # ---------------------------------------------------------
    # PHASE 3: Zenith Push (Frames 96 - 131) | 24° to 46°
    # ---------------------------------------------------------
    elif frame < 132:
        progress = (frame - 96) / 35.0
        sun_elev = np.interp(progress, [0, 1], [24.0, 46.0])
        cam_azim = 180.0

    # ---------------------------------------------------------
    # PHASE 4: Zenith Linger (Frames 132 - 143) | Frozen at 46°
    # ---------------------------------------------------------
    elif frame < 144:
        sun_elev = 46.0
        cam_azim = 180.0
        skip_render = True

    # ---------------------------------------------------------
    # PHASE 5: Detach & Drop (Frames 144 - 191) | Elev drops, Azim yaws
    # ---------------------------------------------------------
    elif frame < 192:
        progress = (frame - 144) / 47.0
        sun_elev = np.interp(progress, [0, 1], [46.0, 10.0])
        cam_azim = np.interp(progress, [0, 1], [180.0, 360.0])

    # ---------------------------------------------------------
    # PHASE 6: The Mist Transition & Rainbow Sweep (Frames 192 - 239)
    # ---------------------------------------------------------
    elif frame < 240:
        mode = 'mist'
        sun_elev = 15.0  # Kept low so the rainbow is actually in the sky
        cam_azim = 360.0

        # Sweep from 30 to 1000 over 48 frames (4 seconds)
        progress = (frame - 192) / 47.0
        raw_radius = np.interp(progress, [0, 1], [30.0, 1000.0])

        # SNIPER QUANTIZATION:
        # <200um: Highly volatile wave optics (15um steps)
        # 200-500um: Settling Airy patterns (50um steps)
        # >500um: Pure stable geometric optics (100um strides)
        if raw_radius < 200.0:
            step_size = 15.0
        elif raw_radius < 500.0:
            step_size = 50.0
        else:
            step_size = 100.0

        radius = round(raw_radius / step_size) * step_size

        # Ensure we don't snap below our starting size
        radius = max(30.0, radius)

        # Variance drops from 0.5 to 0.005 as the quantized radius grows from 30 to 200.
        # np.interp automatically clamps it at 0.005 for any radius > 200.
        variance = float(np.interp(radius, [30.0, 200.0], [0.5, 0.005]))

    # ---------------------------------------------------------
    # PHASE 7: The Whip Pan (Frames 240 - 263)
    # ---------------------------------------------------------
    elif frame < 264:
        mode = 'mist'
        sun_elev = 15.0
        variance = 0.005
        radius = 1000.0  # Hold the final radius from the sweep

        # Quick fucking turn: 2 seconds (24 frames) back to 180
        progress = (frame - 240) / 23.0
        # Easing out the camera movement makes it look less robotic
        eased_progress = progress * progress * (3 - 2 * progress)
        cam_azim = np.interp(eased_progress, [0, 1], [360.0, 180.0])

        # ---------------------------------------------------------
        # PHASE 8: The Corona Drops (Frames 264 - 280)
        # ---------------------------------------------------------
    else:
        mode = 'mist'
        sun_elev = 15.0
        cam_azim = 180.0
        variance = 0.005

        drop_steps = [1000, 100, 50, 30, 20, 12, 6]
        step_idx = (frame - 264) // 2

        if step_idx < len(drop_steps):
            radius = drop_steps[step_idx]
            # This is what you were missing! Skip the second frame of the step.
            if (frame - 264) % 2 != 0:
                skip_render = True
        else:
            radius = 6.0
            skip_render = True  # Freeze on the final 6um frame
    return sun_elev, cam_azim, turbulence, skip_render, mode, radius, variance


def run_bench():
    total_frames = 281  # Bumped so Phase 8 actually finishes
    START_FRAME = 0
    last_rendered_file = None

    # Trackers for the bake printouts
    last_bake_elev = None
    last_turb = None
    last_radius = None  # <--- Just track it here
    last_mode = None    # <--- Track the ice/mist switch
    current_pdict = None

    for frame in range(START_FRAME, total_frames):
        sun_elev, cam_azim, turb, skip, mode, radius, var = get_frame_state(frame)
        output_path = os.path.join(OUTPUT_DIR, f"frame_{frame:04d}.exr")

        # --- NEW: THE SMART RESUME CHECK ---
        # If the file exists AND is larger than 100 bytes (not a dead 0b file), skip it.
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            print(f"[{frame:03d}/{total_frames - 1}] ALREADY RENDERED. Skipping.")
            last_rendered_file = output_path  # Keep the pointer updated for copies!
            continue
        # -----------------------------------

        phase_bake_elev = round(sun_elev / 2.0) * 2.0

        # FALLBACK: If we need to skip, but the script just started (last_rendered_file is None)
        if skip and not last_rendered_file:
            prev_file = os.path.join(OUTPUT_DIR, f"frame_{frame - 1:04d}.exr")
            if os.path.exists(prev_file):
                last_rendered_file = prev_file
            else:
                print(f"⚠️ Missing {prev_file} to copy from! Forcing a render.")
                skip = False  # Force render to prevent a crash

        is_new_bake = (
                (phase_bake_elev != last_bake_elev) or
                (turb != last_turb) or
                (radius != last_radius) or
                (mode != last_mode)
        )

        if is_new_bake and not skip:
            print(
                f"\n🔥 [NEW PHASE BAKE] -> Mode: {mode.upper()} | Elev: {phase_bake_elev}° | Rad: {radius} | Turb/Var: {turb if mode == 'ice' else var}")

            if mode == 'ice':
                cfg = Particle.hexagonal(
                    num_angles=2048, num_phi_bins=1024, num_wavelengths=12,
                    backend=BackendType.DRJIT, sun_elevation_deg=phase_bake_elev,
                    air_turbulence_factor=turb,
                    composition=[
                        HexagonalComposition(HexagonalHabit.TUMBLING, 0.4, 1500, 800, 0.0),
                        HexagonalComposition(HexagonalHabit.PLATE, 0.3, 300, 1500, 0.0),
                        HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 0.2, 3000, 300, 0.0),
                        HexagonalComposition(HexagonalHabit.PARRY, 0.1, 3000, 300, 0.0)
                    ]
                )
            else:
                from nimbuscore.core.config import DropletComposition, DropletShape
                cfg = Particle.droplet(
                    composition=[DropletComposition(DropletShape.SPHERE, 1.0, radius, var)],
                    num_angles=8192, num_wavelengths=32, num_phi_bins=360, backend=BackendType.AUTO
                )

            current_pdict = create_atmospheric_phase(cfg, (0, 1, 0), force_regen=False, threshold=1.0)

            last_bake_elev = phase_bake_elev
            last_turb = turb
            last_radius = radius
            last_mode = mode

        print(f"[{frame:03d}/{total_frames - 1}] Sun: {sun_elev:05.2f}° | Azim: {cam_azim:06.2f}° | ", end="")

        # THE NEW COPY BLOCK
        if skip:
            shutil.copy(last_rendered_file, output_path)
            # CRITICAL: Update the pointer so subsequent skips copy the newest file!
            last_rendered_file = output_path
            print("COPIED")
            continue

        frame_params = {
            'sun_elevation': sun_elev,
            'phase_bake_elevation': phase_bake_elev,
            'camera_azimuth': cam_azim,
            'turbulence': turb,
            'max_depth': CONFIG['max_depth'],
            'spp': CONFIG['target_spp'],
            'res_w': CONFIG['res_w'],
            'res_h': CONFIG['res_h'],
            'phase_dict': current_pdict
        }

        scene_dict = build_scene(frame_params)
        scene = mi.load_dict(scene_dict)
        img = mi.render(scene, spp=frame_params['spp'])
        mi.util.write_bitmap(output_path, img)

        last_rendered_file = output_path
        print("RENDERED")


if __name__ == "__main__":
    run_bench()
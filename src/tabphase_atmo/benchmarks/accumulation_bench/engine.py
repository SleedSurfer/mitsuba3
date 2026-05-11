import mitsuba as mi
import drjit as dr
import time
import numpy as np
from PIL import Image
from pathlib import Path

# Saves to PROJECT_ROOT/bench_renders
HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parents[1] / "outputs"


def run_render_bench(scene_dict, run_name, config, start_from_preview=False):
    print(f"--- STARTING BENCH: {run_name} ---")
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scene...")
    scene = mi.load_dict(scene_dict)

    # Config unpacking
    target_spp = config.get('target_spp', 64)
    batch_size = config.get('batch_size', 16)
    film_size = scene.sensors()[0].film().crop_size()

    # --- CHECKPOINT RECOVERY LOGIC ---
    # Look for: preview_{pass}_{spp}spp.exr
    previews = list(run_dir.glob("preview_*_*spp.exr"))
    acc_buffer = dr.zeros(mi.TensorXf, shape=(film_size[1], film_size[0], 3))
    start_pass = 0
    total_accumulated_spp = 0

    if start_from_preview and previews:
        try:
            # Sort by SPP (the last number in the stem)
            latest_preview = max(previews, key=lambda p: int(p.stem.split('_')[-1].replace('spp', '')))
            last_spp_str = latest_preview.stem.split('_')[-1].replace('spp', '')
            total_accumulated_spp = int(last_spp_str)

            print(f"[+] Found checkpoint: {latest_preview.name}. Reconstructing state...")

            # Load and convert to float32 tensor (Raw/Linear)
            bmp = mi.Bitmap(str(latest_preview)).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
            checkpoint_tensor = mi.TensorXf(bmp)

            # Reconstruct the accumulation sum: Average * SPP
            acc_buffer = checkpoint_tensor * float(total_accumulated_spp)
            start_pass = total_accumulated_spp // batch_size

            dr.eval(acc_buffer)
            print(f"[+] Successfully resumed at Pass {start_pass} ({total_accumulated_spp} SPP).")
        except Exception as e:
            print(f"[!] Failed to parse previews: {e}. Starting fresh.")
    else:
        if previews and not start_from_preview:
            print(f"[i] Found {len(previews)} preview file(s) in {run_dir}, but preview loading is disabled. Starting fresh.")
        else:
            print("[*] No previews found. Starting fresh render.")

    # --- LOOP SETTINGS ---
    is_infinite = (target_spp == -1)
    total_passes = float('inf') if is_infinite else (target_spp // batch_size)

    pass_num = start_pass
    start_time = time.time()

    try:
        while pass_num < total_passes:
            # seed=pass_num ensures we continue the random sequence correctly
            img = mi.render(scene, seed=pass_num, spp=batch_size)

            # Ensure a numeric tensor type (Bitmap -> TensorXf or passthrough)
            try:
                img = mi.TensorXf(img)
            except Exception:
                # fall back to using the returned value as-is
                pass

            # mi.render returns an image averaged over `spp=batch_size` samples.
            # To accumulate energy correctly we must add the *sum* contributed by
            # this batch, not the averaged image. Multiply by batch_size.
            img_sum = img * float(batch_size)

            acc_buffer += img_sum
            dr.eval(acc_buffer)  # Keep VRAM updated

            # Debug prints for early passes to inspect dynamic range
            if pass_num <= 2:
                try:
                    img_np = np.array(img)
                    acc_np = np.array(acc_buffer)
                    print(f"[dbg] pass={pass_num} img min={img_np.min():.6g} max={img_np.max():.6g} mean={img_np.mean():.6g}")
                    print(f"[dbg] acc min={acc_np.min():.6g} max={acc_np.max():.6g} mean={acc_np.mean():.6g}")
                except Exception as e:
                    print(f"[dbg] could not print stats: {e}")

            pass_num += 1
            current_total_spp = pass_num * batch_size
            elapsed = time.time() - start_time

            if pass_num % 5 == 0:
                avg = elapsed / (pass_num - start_pass + 1e-5)
                if is_infinite:
                    print(
                        f"[{run_name}] Pass {pass_num} ({current_total_spp} SPP). Total session time: {elapsed / 60:.2f} mins")
                else:
                    eta = (total_passes - pass_num) * avg
                    print(
                        f"[{run_name}] Pass {pass_num}/{total_passes} ({current_total_spp} SPP). ETA: {eta / 60:.2f} mins")

            # Preview Logic (Saves the AVERAGED state for checkpointing)
            is_early = pass_num in [1, 2, 3, 5, 10, 20]
            is_periodic = pass_num > 20 and pass_num % 10 == 0

            if is_early or is_periodic:
                temp_avg = acc_buffer / float(current_total_spp)
                preview_name = f"preview_{pass_num}_{current_total_spp}spp.exr"
                mi.util.write_bitmap(str(run_dir / preview_name), temp_avg)

    except KeyboardInterrupt:
        print("\n[!] Render interrupted. Saving Master for safety...")

    # --- FINALIZATION ---
    final_spp = pass_num * batch_size
    if final_spp > 0:
        final_image = acc_buffer / float(final_spp)
        print(f"Saving Master EXR ({final_spp} total SPP)...")
        mi.util.write_bitmap(str(run_dir / "render_final_4k.exr"), final_image)

        if 'downscale_res' in config:
            print(f"Downscaling to {config['downscale_res']}...")
            final_np = np.array(final_image)
            tgt = config['downscale_res']
            # Using LANCZOS to maintain peak-frequency noise for the denoiser
            channels = [Image.fromarray(final_np[:, :, c], mode='F').resize(tgt, Image.Resampling.LANCZOS) for c in
                        range(3)]
            downscaled = np.stack([np.array(ch) for ch in channels], axis=-1)
            mi.util.write_bitmap(str(run_dir / "render_final_downscaled.exr"), mi.TensorXf(downscaled))

    print(f"--- FINISHED: {run_name} ---")

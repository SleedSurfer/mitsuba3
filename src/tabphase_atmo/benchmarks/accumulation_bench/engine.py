import mitsuba as mi
import drjit as dr
import time
import numpy as np
from PIL import Image
from pathlib import Path

# Saves to PROJECT_ROOT/bench_renders
HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parents[1] / "outputs"



def run_render_bench(scene_dict, run_name, config):
    print(f"--- STARTING BENCH: {run_name} ---")
    print(f"Output directory is {OUT_ROOT}/{run_name}")
    run_dir = OUT_ROOT / run_name
    exr_dir = run_dir
    exr_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scene...")
    scene = mi.load_dict(scene_dict)

    # Config unpacking
    target_spp = config.get('target_spp', 64)
    batch_size = config.get('batch_size', 16)
    total_passes = target_spp // batch_size

    print(f"Settings: {target_spp} SPP ({total_passes} batches x {batch_size}).")

    film_size = scene.sensors()[0].film().crop_size()
    acc_buffer = dr.zeros(mi.TensorXf, shape=(film_size[1], film_size[0], 3))

    start_time = time.time()

    for i in range(total_passes):
        img = mi.render(scene, seed=i, spp=batch_size)
        acc_buffer += img
        dr.eval(acc_buffer)

        elapsed = time.time() - start_time
        pass_num = i + 1

        if pass_num % 5 == 0:
            avg = elapsed / pass_num
            eta = (total_passes - pass_num) * avg
            print(f"[{run_name}] Pass {pass_num}/{total_passes}. ETA: {eta / 60:.2f} mins")

        # Preview Logic
        is_early = pass_num in [1, 2, 3, 5, 10, 20]
        is_periodic = pass_num > 50 and pass_num % 10 == 0

        if is_early or is_periodic:
            temp_img = acc_buffer / pass_num
            mi.util.write_bitmap(str(exr_dir / f"preview_{pass_num}_{batch_size * pass_num}spp.exr"), temp_img)

    # Final
    final_image = acc_buffer / total_passes
    print("Saving Master EXR...")
    mi.util.write_bitmap(str(exr_dir / "render_final_4k.exr"), final_image)

    # Downscale (Optional)
    if 'downscale_res' in config:
        print(f"Downscaling to {config['downscale_res']}...")
        final_np = np.array(final_image)
        tgt = config['downscale_res']
        # PIL Resize
        channels = [Image.fromarray(final_np[:, :, c], mode='F').resize(tgt, Image.LANCZOS) for c in range(3)]
        downscaled = np.stack([np.array(ch) for ch in channels], axis=-1)
        mi.util.write_bitmap(str(exr_dir / "render_final_downscaled.exr"), mi.TensorXf(downscaled))

    print(f"--- FINISHED: {run_name} ---")
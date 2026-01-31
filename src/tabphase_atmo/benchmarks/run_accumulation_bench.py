import sys
from pathlib import Path
import mitsuba as mi

# --- SETUP PATHS ---
# Add 'src' to path so we can import 'atmospheric'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# --- VARIANT ---
mi.set_variant("llvm_spectral")  # Change to 'cuda_spectral' on L4

from accumulation_bench.engine import run_render_bench
from accumulation_bench.bench_scenes import street_lamps, brocken_spectre, pure_rainbow,  cornell_exact
# --- CONFIG ---
CONFIG = {
    'target_spp': 16384*4,
    'batch_size': 256,
    'res_w': 256,
    'res_h': 256,
    'downscale_res': (1920, 1080)
}

if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a cpu to burn.")

    # 1. Get Scene
    #scene_dict = street_lamps.get_scene(CONFIG)
    scene_dict = cornell_exact.get_scene(CONFIG)
    # 2. Run
    run_render_bench(
        scene_dict,
        run_name="cornell_box_stresstest_SIMD_is_back",
        config=CONFIG
    )
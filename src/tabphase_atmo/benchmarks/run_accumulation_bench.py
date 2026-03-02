import mitsuba as mi


# --- VARIANT ---
mi.set_variant("llvm_spectral")

from accumulation_bench.engine import run_render_bench
from accumulation_bench.bench_scenes import street_lamps, brocken_spectre, pure_rainbow,  cornell_exact
from accumulation_bench.bench_scenes_win import pure_rainbow as rainbow
# --- CONFIG ---
CONFIG = {
    'target_spp': 16384,
    'batch_size': 256,
    'res_w': 1024,
    'res_h': 1024,
    'downscale_res': (1920, 1080),
    'show_previews': True,
}

if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a cpu to burn.")

    # scene_dict = rainbow.get_scene(CONFIG)
    # run_render_bench(
    #     scene_dict,
    #     run_name="cloud_test_capped_depth_16",
    #     config=CONFIG
    # )

    scene_dict = pure_rainbow.get_scene(CONFIG)
    run_render_bench(
        scene_dict,
        run_name="rain_test_newRT_waves_smallres",
        config=CONFIG,
    )


    # scene_dict = brocken_spectre.get_scene(CONFIG)
    # # 2. Run
    # run_render_bench(
    #     scene_dict,
    #     run_name="brocken_spectre",
    #     config=CONFIG
    # )
    # scene_dict = street_lamps.get_scene(CONFIG)
    # # 2. Run
    # run_render_bench(
    #     scene_dict,
    #     run_name="street_lamps_test",
    #     config=CONFIG
    # )

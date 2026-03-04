import mitsuba as mi


# --- VARIANT ---
mi.set_variant("llvm_spectral")

from accumulation_bench.engine import run_render_bench
from accumulation_bench.bench_scenes import street_lamps, brocken_spectre, pure_rainbow,  cornell_exact
from accumulation_bench.bench_scenes_win import pure_rainbow as rainbow
from src.core.config import MieConfig
# --- CONFIG ---
CONFIG = {
    'target_spp': 5120,
    'batch_size': 256,
    'res_w': 1024,
    'res_h': 1024,
    'downscale_res': (1920, 1080),
    'show_previews': False,
}

if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a cpu to burn.")

    # scene_dict = rainbow.get_scene(CONFIG)
    # run_render_bench(
    #     scene_dict,
    #     run_name="cloud_test_capped_depth_16",
    #     config=CONFIG
    # )

    cfg = MieConfig(radius_mean_um=600.0, variance=0.0, num_angles=8192, num_wavelengths=1,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG,cfg)
    run_render_bench(
        scene_dict,
        run_name="600_poly_00red",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=600.0, variance=0.1, num_angles=8192, num_wavelengths=1,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="600_poly_10red",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=600.0, variance=0.25, num_angles=8192, num_wavelengths=1,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="600_poly_25red",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=600.0, variance=0.6, num_angles=8192, num_wavelengths=1,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="600_poly_60red",
        config=CONFIG,
    )
   #----------------------------------------------------------------------------------------
   #----------------------------------------------------------------------------------------
   #----------------------------------------------------------------------------------------
   #----------------------------------------------------------------------------------------
   #----------------------------------------------------------------------------------------
    cfg = MieConfig(radius_mean_um=600.0, variance=0.0, num_angles=8192, num_wavelengths=32,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="600_poly_00",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=600.0, variance=0.1, num_angles=8192, num_wavelengths=32,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="600_poly_10",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=600.0, variance=0.25, num_angles=8192, num_wavelengths=32,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="600_poly_25",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=600.0, variance=0.6, num_angles=8192, num_wavelengths=32,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="600_poly_60",
        config=CONFIG,
    )
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------

    cfg = MieConfig(radius_mean_um=200.0, variance=0.0, num_angles=8192, num_wavelengths=32,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="200_poly_00",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=200.0, variance=0.1, num_angles=8192, num_wavelengths=32,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="200_poly_10",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=200.0, variance=0.25, num_angles=8192, num_wavelengths=32,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="200_poly_25",
        config=CONFIG,
    )

    cfg = MieConfig(radius_mean_um=200.0, variance=0.6, num_angles=8192, num_wavelengths=32,
                    brute_force_integration=True)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="200_poly_60",
        config=CONFIG,
    )
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    
    
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

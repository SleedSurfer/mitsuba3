import mitsuba as mi

from config import ParticleShape

# --- VARIANT ---
mi.set_variant("llvm_spectral")

from accumulation_bench.engine import run_render_bench
from accumulation_bench.bench_scenes import street_lamps, brocken_spectre, pure_rainbow,  cornell_exact, Ice
from accumulation_bench.bench_scenes_win import pure_rainbow as rainbow

from src.core.config import MieConfig, BackendType
# --- CONFIG ---
CONFIG = {
    'target_spp': 1280,
    'batch_size': 256,
    'res_w': 1024,
    'res_h': 1024,
    'downscale_res': (1920, 1080),
    'show_previews': True,
}

if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a cpu to burn.")

    # cfg = MieConfig(radius_mean_um=2000.0,
    #                 variance=0.0,
    #                 num_angles=2048,
    #                 num_phi_bins=720,
    #                 num_wavelengths=64,
    #                 note="hexagonal",
    #                 backend=BackendType.DRJIT,
    #                 shape=ParticleShape.HEXAGONAL)
    # scene_dict = Ice.get_scene(CONFIG, cfg)
    # run_render_bench(
    #     scene_dict,
    #     run_name="hexagonal_squarephase",
    #     config=CONFIG,
    # )



    # cfg = MieConfig(radius_mean_um=300.0,
    #                 variance=0.0,
    #                 num_angles=8192,
    #                 num_phi_bins=360,
    #                 num_wavelengths=32,
    #                 note="oblate",
    #                 backend=BackendType.DRJIT,
    #                 shape=ParticleShape.OBLATE)
    # scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    # run_render_bench(
    #     scene_dict,
    #     run_name="oblate",
    #     config=CONFIG,
    # )

    cfg = MieConfig(radius_mean_um=100.0,
                    variance=0.0,
                    num_angles=8192,
                    num_phi_bins=360,
                    num_wavelengths=32,
                    note="ball",
                    backend=BackendType.DRJIT,
                    shape=ParticleShape.SPHERE)
    scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    run_render_bench(
        scene_dict,
        run_name="ball_32wl_100um",
        config=CONFIG,
    )


    # scene_dict = rainbow.get_scene(CONFIG)
    # run_render_bench(
    #     scene_dict,
    #     run_name="cloud_test_capped_depth_16",
    #     config=CONFIG
    # )

    # cfg = MieConfig(radius_mean_um=120.0,
    #                    variance=0.0,
    #                    num_angles=4192,
    #                    num_wavelengths=32,
    #                    brute_force_integration=True,
    #                    note="backend_visual_differences_hybrid",
    #                    backend=BackendType.HYBRID)
    # scene_dict = pure_rainbow.get_scene(CONFIG,cfg)
    # run_render_bench(
    #     scene_dict,
    #     run_name="backend_visual_differences_hybrid",
    #     config=CONFIG,
    # )

    # cfg = MieConfig(radius_mean_um=120.0,
    #                 variance=0.0,
    #                 num_angles=4192,
    #                 num_wavelengths=32,
    #                 brute_force_integration=True,
    #                 note="backend_visual_differences_mie",
    #                 backend=BackendType.MIEPYTHON)
    # scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    # run_render_bench(
    #     scene_dict,
    #     run_name="backend_visual_differences_mie",
    #     config=CONFIG,
    # )

    # cfg = MieConfig(radius_mean_um=120.0,
    #                 variance=0.0,
    #                 num_angles=8192,
    #                 num_wavelengths=32,
    #                 brute_force_integration=True,
    #                 note="backend_visual_differences_rt",
    #                 backend=BackendType.DRJIT)
    # scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    # run_render_bench(
    #     scene_dict,
    #     run_name="backend_visual_differences_rt_g2048",
    #     config=CONFIG,
    # )

    # cfg = MieConfig(radius_mean_um=500.0, variance=0.0, num_angles=8192, num_wavelengths=64,
    #                 brute_force_integration=True)
    # scene_dict = pure_rainbow.get_scene(CONFIG,cfg)
    # run_render_bench(
    #     scene_dict,
    #     run_name="300_poly_00_32wl",
    #     config=CONFIG,
    # )

   #  cfg = MieConfig(radius_mean_um=600.0, variance=0.1, num_angles=8192, num_wavelengths=1,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="600_poly_10red",
   #      config=CONFIG,
   #  )
   #
   #  cfg = MieConfig(radius_mean_um=600.0, variance=0.25, num_angles=8192, num_wavelengths=1,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="600_poly_25red",
   #      config=CONFIG,
   #  )
   #
   #  cfg = MieConfig(radius_mean_um=600.0, variance=0.6, num_angles=8192, num_wavelengths=1,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="600_poly_60red",
   #      config=CONFIG,
   #  )
   # #----------------------------------------------------------------------------------------
   # #----------------------------------------------------------------------------------------
   # #----------------------------------------------------------------------------------------
   # #----------------------------------------------------------------------------------------
   # #----------------------------------------------------------------------------------------
   #  cfg = MieConfig(radius_mean_um=600.0, variance=0.0, num_angles=8192, num_wavelengths=32,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="600_poly_00",
   #      config=CONFIG,
   #  )
   #
   #  cfg = MieConfig(radius_mean_um=600.0, variance=0.1, num_angles=8192, num_wavelengths=32,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="600_poly_10",
   #      config=CONFIG,
   #  )
   #
   #  cfg = MieConfig(radius_mean_um=600.0, variance=0.25, num_angles=8192, num_wavelengths=32,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="600_poly_25",
   #      config=CONFIG,
   #  )
   #
   #  cfg = MieConfig(radius_mean_um=600.0, variance=0.6, num_angles=8192, num_wavelengths=32,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="600_poly_60",
   #      config=CONFIG,
   #  )
   #  # ----------------------------------------------------------------------------------------
   #  # ----------------------------------------------------------------------------------------
   #  # ----------------------------------------------------------------------------------------
   #  # ----------------------------------------------------------------------------------------
   #  # ----------------------------------------------------------------------------------------
   #
   #  cfg = MieConfig(radius_mean_um=200.0, variance=0.0, num_angles=8192, num_wavelengths=32,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="200_poly_00",
   #      config=CONFIG,
   #  )
   #
   #  cfg = MieConfig(radius_mean_um=200.0, variance=0.1, num_angles=8192, num_wavelengths=32,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="200_poly_10",
   #      config=CONFIG,
   #  )
   #
   #  cfg = MieConfig(radius_mean_um=200.0, variance=0.25, num_angles=8192, num_wavelengths=32,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="200_poly_25",
   #      config=CONFIG,
   #  )
   #
   #  cfg = MieConfig(radius_mean_um=200.0, variance=0.6, num_angles=8192, num_wavelengths=32,
   #                  brute_force_integration=True)
   #  scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
   #  run_render_bench(
   #      scene_dict,
   #      run_name="200_poly_60",
   #      config=CONFIG,
   #  )
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

import mitsuba as mi

from benchmarks.accumulation_bench.bench_scenes import cornell_fog_cloud, sunsky_cloud, cloud
from config import HexagonalHabit, HexagonalComposition, DropletComposition
from src.core.config import Particle, BackendType, DropletShape

# --- VARIANT ---
mi.set_variant("llvm_spectral")

from accumulation_bench.engine import run_render_bench
from accumulation_bench.bench_scenes import Ice_sunsky, pure_rainbow,mitsuba3cornell,cornell_fog_sphere
import accumulation_bench.bench_scenes.cornell_exact as fogbox

RES_M = 16 #BEAUTY PASS

# --- CONFIG ---
CONFIG = {
    'target_spp': -1,
    'batch_size': 16,
    # 'res_w':512,
    # 'res_h':512,
    # 'res_w':1024,
    # 'res_h':1024,
    'res_w':64*RES_M,
    'res_h':35*RES_M,
    # 'res_w':8,
    # 'res_h':8,

    'downscale_res': (1024,1024),
    'show_previews': False,
    'denoise_enabled': False,
}

if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a cpu to burn.")




    # mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 8.0, 0.006)],
    #                         num_angles=8192,
    #                         num_wavelengths=33,
    #                         num_phi_bins=360,
    #                         backend=BackendType.MIEPYTHON)
    # mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 1500.0, 0.0)],
    #                         num_angles=8192,
    #                         num_wavelengths=6,
    #                         num_phi_bins=360,
    #                         backend=BackendType.DRJIT)
    # rain = Particle.droplet(composition=[DropletComposition(DropletShape.OBLATE, 0.2, 400.0, 0.02),
    #                                      DropletComposition(DropletShape.SPHERE, 0.8, 400.0, 0.02)],
    #                         num_angles=8192,
    #                         num_wavelengths=36,
    #                         num_phi_bins=2048,
    #                         backend=BackendType.DRJIT)
    #rain = ""

    # SUN_ELEVATION = 22.0
    # TURBULENCE = 1
    # cfg = Particle.hexagonal(
    #     num_angles=8192,
    #     num_phi_bins=1024,
    #     num_wavelengths=32,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=SUN_ELEVATION,
    #     air_turbulence_factor=TURBULENCE,
    #     composition=[
    #         HexagonalComposition(HexagonalHabit.TUMBLING, 0.2, 1500, 800, 0.05),
    #         HexagonalComposition(HexagonalHabit.PLATE, 0.3,  800, 1500, 0.05),
    #         HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 0.5, 1500, 800, 0.05),
    #     ]
    # )

    cfg = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 8.0, 0.3)],
                                      num_angles=8192,
                                      num_wavelengths=33,
                                      num_phi_bins=360,
                                      backend=BackendType.MIEPYTHON)

    run_name = "disney_cloud"
    dictionary = cloud.get_scene(CONFIG,cfg)
    run_render_bench(dictionary, run_name=run_name, config=CONFIG)

    # run_name="sunsky_cloud_single_HG"
    # dictionary = sunsky_cloud.get_scene(CONFIG, cfg)
    # run_render_bench(dictionary, run_name=run_name, config=CONFIG)
    # run_name = "grih"
    # dictionary = cornell_fog_sphere.get_scene(CONFIG, cfg)
    # run_render_bench(dictionary, run_name=run_name, config=CONFIG)

    # cfg = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 300.0, 0.0)],
    #                        num_angles=8192,
    #                        num_wavelengths=6,
    #                        num_phi_bins=360,
    #                        backend=BackendType.MIEPYTHON)
    # run_name = "rainbow test"
    # dictionary = pure_rainbow.get_scene(CONFIG, cfg)
    # run_render_bench(dictionary, run_name=run_name, config=CONFIG)



    # mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 8.0, 0.1)],
    #                         num_angles=8192,
    #                         num_wavelengths=33,
    #                         num_phi_bins=360,
    #                         backend=BackendType.MIEPYTHON)
    # # mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 1500.0, 0.0)],
    # #                         num_angles=8192,
    # #                         num_wavelengths=6,
    # #                         num_phi_bins=360,
    # #                         backend=BackendType.DRJIT)
    #
    # run_name = f"blowout_cap1/8um/0var"
    # scene_dict = cornell_fog_sphere.get_scene(CONFIG, mist)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)



    # mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0,8.0, 0.0)],
    #                         num_angles=4096,
    #                         num_wavelengths=17,
    #                         num_phi_bins=360,
    #                         backend=BackendType.MIEPYTHON)
    # run_name = f"mietest/8um/0var"
    # scene_dict = pure_rainbow.get_scene(CONFIG, mist)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)
    #
    #
    # rain = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 300.0, 0.0)],
    #                         num_angles=8192*2,
    #                         num_wavelengths=32,
    #                         num_phi_bins=360,
    #                         backend=BackendType.MIEPYTHON)
    #
    # run_name = f"mietest/300um/0var"
    # scene_dict = pure_rainbow.get_scene(CONFIG, rain)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)
    #
    # rain = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 400.0, 0.00)],
    #                         num_angles=8192 * 2,
    #                         num_wavelengths=32,
    #                         num_phi_bins=360,
    #                         backend=BackendType.MIEPYTHON)
    #
    # run_name = f"Jittest_d32/300um/50var"
    # scene_dict = pure_rainbow.get_scene(CONFIG, rain)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)

    # rain = Particle.droplet(composition=[DropletComposition(DropletShape.OBLATE, 0.5, 400.0, 0.02),DropletComposition(DropletShape.SPHERE, 0.5, 400.0, 0.02),],
    #                         num_angles=8192,
    #                         num_wavelengths=36,
    #                         num_phi_bins=2048,
    #                         backend=BackendType.DRJIT)
    #
    # run_name = f"Oblate_test"
    # scene_dict = pure_rainbow.get_scene(CONFIG, rain)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)

    # # # ===============================
    # # #      JIT RT TESTS
    # # # ===============================
    # #
    # cfg = Particle.droplet(
    #     num_angles=8192,
    #     num_phi_bins=360,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     composition=[DropletComposition(DropletShape.SPHERE, 1.0,500.0, 0.04)],
    # )
    # run_name = f"rainbow_rt/size_500/test"
    # scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    #
    # print(f"\n>>> STARTING BATCH")
    # print(f">>> Target Path: {run_name}")
    #
    # run_render_bench(
    #     scene_dict,
    #     run_name=run_name,
    #     config=CONFIG,
    # )
    #
    #



    # SUN_ELEVATIONS = [40.0,50.0,70.0,120.0]#5.0,10.0,22.0,
    # TURBULENCES = [1,3,7]
    #
    #
    # for elevation in SUN_ELEVATIONS:
    #     for turbulence in TURBULENCES:
    #         cfg = Particle.hexagonal(
    #             num_angles=4096,
    #             num_phi_bins=2048,
    #             num_wavelengths=24,
    #             backend=BackendType.DRJIT,
    #             sun_elevation_deg=elevation,
    #             air_turbulence_factor=turbulence,
    #             composition=[
    #                 HexagonalComposition(HexagonalHabit.TUMBLING, 0.4, 1500, 800, 0.0),
    #                 HexagonalComposition(HexagonalHabit.PLATE, 0.3, 800, 1500, 0.0),
    #                 HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 0.2, 1500, 800, 0.0),
    #                 HexagonalComposition(HexagonalHabit.PARRY, 0.1, 1500, 800, 0.0)
    #             ]
    #         )
    #         run_name = f"Halo_Beautyshot/turb_{turbulence}/elev_{elevation}"
    #         scene_dict = Ice_sunsky.get_scene(CONFIG, cfg, sun_elevation_deg=elevation)
    #
    #         print(f"\n>>> STARTING BATCH: HALO Turbulence={turbulence}, elevation={elevation}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(scene_dict, run_name=run_name, config=CONFIG)




    # SUN_ELEVATION = 22.0
    # TURBULENCE = 1
    # cfg = Particle.hexagonal(
    #     num_angles=8192,
    #     num_phi_bins=360,
    #     num_wavelengths=32,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=SUN_ELEVATION,
    #     air_turbulence_factor=TURBULENCE,
    #     composition=[
    #         HexagonalComposition(HexagonalHabit.TUMBLING, 1.0, 1500, 800, 0.0)
    #     ]
    # )
    #
    # run_name = f"Halo_pure/turb_{TURBULENCE}/elev_{SUN_ELEVATION}"
    # scene_dict = Ice_sunsky.get_scene(CONFIG, cfg, sun_elevation_deg=SUN_ELEVATION)
    #
    # print(f"\n>>> STARTING BATCH: HALO Turbulence={TURBULENCE}, elevation={SUN_ELEVATION}")
    # print(f">>> Target Path: {run_name}")
    #
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)

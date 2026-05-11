import mitsuba as mi

from benchmarks.accumulation_bench.bench_scenes import cornell_fog_cloud, sunsky_cloud, cloud, sunsky_rainbow, \
    mars_halo, sunsky_corona, sunsky_glory, white_furnace, Polar_region
from config import HexagonalHabit, HexagonalComposition, DropletComposition, CuboctahedralComposition, \
    CuboctahedralHabit
from nimbuscore.core.config import Particle, BackendType, DropletShape

# --- VARIANT ---
mi.set_variant("llvm_ad_spectral")

from accumulation_bench.engine import run_render_bench
from accumulation_bench.bench_scenes import Ice_sunsky, pure_rainbow,mitsuba3cornell,cornell_fog_sphere
import accumulation_bench.bench_scenes.cornell_exact as fogbox

RES_M = 2 #BEAUTY PASS

# --- CONFIG ---
CONFIG = {
    'target_spp': 640+320,
    'batch_size': 64,
    # 'res_w':512,
    # 'res_h':512,
    'res_w':1920,
    'res_h':1080,
    # 'res_w': 1600*2,
    # 'res_h': 1600*2,
    # 'res_w':64*RES_M,
    # 'res_h':35*RES_M,
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

    SUN_ELEVATION = 8.0
    TURBULENCE = 0.4
    cfg = Particle.hexagonal(
        num_angles=2048,
        num_phi_bins=1024,
        num_wavelengths=12,
        sun_elevation_deg=SUN_ELEVATION,
        air_turbulence_factor=TURBULENCE,
        composition=[
            HexagonalComposition(HexagonalHabit.TUMBLING, 0.5, 1500, 800, 0.05),
            HexagonalComposition(HexagonalHabit.PLATE, 0.3,  800, 2000, 0.05),
            HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 0.1, 3000, 800, 0.005),
            HexagonalComposition(HexagonalHabit.PARRY, 0.1, 3000, 800, 0.005),

        ]
    )

    run_name = f"halo_bshot"
    dictionary = Polar_region.get_scene(CONFIG, cfg)
    run_render_bench(dictionary, run_name=run_name, config=CONFIG,start_from_preview=False)




    # # RUNNING PASS 60
    # cfg = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 8.0, 0.0)],
    #                                   num_angles=8192,
    #                                   num_wavelengths=33,
    #                                   num_phi_bins=360,
    #                                   backend=BackendType.MIEPYTHON)

    # run_name = "disney_cloud"
    # dictionary = cloud.get_scene(CONFIG,cfg)
    # run_render_bench(dictionary, run_name=run_name, config=CONFIG,start_from_preview=True)

    # run_name="sunsky_cloud_single_HG"
    # dictionary = sunsky_cloud.get_scene(CONFIG, cfg)
    # run_render_bench(dictionary, run_name=run_name, config=CONFIG)
    # run_name = "grih"
    # dictionary = cornell_fog_sphere.get_scene(CONFIG, cfg)
    # run_render_bench(dictionary, run_name=run_name, config=CONFIG)

    # cfg = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 300.0, 0.0)],
    #                        num_angles=4096,
    #                        num_wavelengths=12,
    #                        num_phi_bins=360,
    #                        backend=BackendType.DRJIT)
    # cfg = Particle.droplet(
    #     composition=[DropletComposition(DropletShape.SPHERE, 1.0, 375, 0.00)],
    #     num_angles=8192,
    #     num_wavelengths=32,
    #     num_phi_bins=360,
    #     backend=BackendType.DRJIT
    # )
    # run_name = "shifting_interference/RT"
    # dictionary = sunsky_rainbow.get_scene(CONFIG, cfg)
    # run_render_bench(dictionary, run_name=run_name, config=CONFIG)
    #
    # cfg = Particle.droplet(
    #     composition=[DropletComposition(DropletShape.SPHERE, 1.0, 375, 0.00)],
    #     num_angles=8192,
    #     num_wavelengths=64,
    #     num_phi_bins=360,
    #     backend=BackendType.MIEPYTHON
    # )
    # run_name = "shifting_interference/Mie"
    # dictionary = sunsky_rainbow.get_scene(CONFIG, cfg)
    # run_render_bench(dictionary, run_name=run_name, config=CONFIG)

    # CONFIG = {
    #     'target_spp': 4096,
    #     'batch_size': 64,
    #     'res_w': 512//4,
    #     'res_h': 512//4,
    #     # 'res_w':1024,
    #     # 'res_h':1024,
    #     # 'res_w':64*RES_M,
    #     # 'res_h':35*RES_M,
    #     # 'res_w':8,
    #     # 'res_h':8,
    #
    #     'downscale_res': (1024, 1024),
    #     'show_previews': False,
    #     'denoise_enabled': False,
    # }
    #
    #
    # mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 8.0, 0.05)],
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
    # run_name = f"regression"
    # scene_dict = cornell_fog_sphere.get_scene(CONFIG, mist)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG, start_from_preview=False)



    # cfg = ""
    # CONFIG = {
    #     'target_spp': 8192,
    #     'batch_size': 16,
    #     'res_w': 1920,
    #     'res_h': 1024,
    #     # 'res_w':1024,
    #     # 'res_h':1024,
    #     # 'res_w':64*RES_M,
    #     # 'res_h':35*RES_M,
    #     # 'res_w':8,
    #     # 'res_h':8,
    #
    #     'downscale_res': (1024, 1024),
    #     'show_previews': False,
    #     'denoise_enabled': False,
    # }
    #
    # crystal = Particle.cuboctahedral(num_angles=4096,
    #                                  num_phi_bins=2048,
    #                                  num_wavelengths=24,
    #                                  sun_elevation_deg=15,
    #                                  air_turbulence_factor=0.4,
    #                                  backend=BackendType.DRJIT,
    #                                  composition=[
    #                                      CuboctahedralComposition(CuboctahedralHabit.TUMBLING, 0.7, 800.0, 0.005),
    #                                      CuboctahedralComposition(CuboctahedralHabit.PLATE, 0.3, 800.0, 0.005)])
    #
    # run_name = f'mars_banner'
    # scene_dict = mars_halo.get_scene(CONFIG,crystal)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG,start_from_preview=False)

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
    #
    # CONFIG = {
    #     'target_spp': 2048,
    #     'batch_size': 64,
    #     # 'res_w':512,
    #     # 'res_h':512,
    #     # 'res_w':1024,
    #     # 'res_h':1024,
    #     'res_w': 1600*2,
    #     'res_h': 1600*2,
    #     # 'res_w':8,
    #     # 'res_h':8,
    #
    #     'downscale_res': (1024, 1024),
    #     'show_previews': False,
    #     'denoise_enabled': False,
    # }
    #
    # for size in [1000.0]:
    #     cfg = Particle.droplet(num_angles=8192,
    #                               num_wavelengths=32,
    #                               num_phi_bins=360,
    #                               backend=BackendType.MIEPYTHON,
    #                               composition=[DropletComposition(DropletShape.SPHERE, 1.0, size, 0.0)])
    #     run_name = f"rainbow_mie_beautyshot/size_{size}"
    #     scene_dict = sunsky_rainbow.get_scene(CONFIG, cfg)
    #     print(f"\n>>> STARTING BATCH: {run_name}")
    #     print(f">>> Target Path: {run_name}")
    #     run_render_bench(scene_dict, run_name=run_name, config=CONFIG,start_from_preview=False)


    # for size in [30.0,70.0,140.0,300.0]:
    #     cfg = Particle.droplet(num_angles=8192,
    #                               num_wavelengths=32,
    #                               num_phi_bins=360,
    #                               backend=BackendType.MIEPYTHON,
    #                               composition=[DropletComposition(DropletShape.SPHERE, 1.0, size, 0.0)])
    #     run_name = f"rainbow_mie_beautyshot/size_{size}"
    #     scene_dict = sunsky_rainbow.get_scene(CONFIG, cfg)
    #     print(f"\n>>> STARTING BATCH: rainbow_mie_beautyshot/size_{size}")
    #     print(f">>> Target Path: {run_name}")
    #     run_render_bench(scene_dict, run_name=run_name, config=CONFIG,start_from_preview=False)
    #
    # for wlcount in [3,6,11]: #
    #     cfg = Particle.droplet(num_angles=8192,
    #                               num_wavelengths=wlcount,
    #                               num_phi_bins=360,
    #                               backend=BackendType.MIEPYTHON,
    #                               composition=[DropletComposition(DropletShape.SPHERE, 1.0, 300, 0.0)])
    #     run_name = f"rainbow_mie_beautyshot/h_wavelengths/nonH{wlcount}"
    #     scene_dict = sunsky_rainbow.get_scene(CONFIG, cfg)
    #     print(f"\n>>> STARTING BATCH: rainbow_mie_beautyshot/h_wavelengths/nonH{wlcount}")
    #     print(f">>> Target Path: {run_name}")
    #     run_render_bench(scene_dict, run_name=run_name, config=CONFIG,start_from_preview=False)
    #========= GROUND TRUTH ============#
    # cfg = Particle.droplet(num_angles=8192,
    #                        num_wavelengths=6,
    #                        num_phi_bins=360,
    #                        backend=BackendType.MIEPYTHON,
    #                        composition=[DropletComposition(DropletShape.SPHERE, 1.0, 300, 0.1)])
    # isbrute = True
    # run_name = f"rainbow_mie_beautyshot/variancebrutevspost/brute"
    # scene_dict = sunsky_rainbow.get_scene(CONFIG, cfg, isbrute=isbrute)
    # print(f"\n>>> STARTING BATCH: {run_name}")
    # print(f">>> Target Path: {run_name}")
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG, start_from_preview=False)

    # isbrute = False
    # for variance in [0.03,0.028,0.024]: #
    #     cfg = Particle.droplet(num_angles=8192,
    #                            num_wavelengths=6,
    #                            num_phi_bins=360,
    #                            backend=BackendType.MIEPYTHON,
    #                            composition=[DropletComposition(DropletShape.SPHERE, 1.0, 300, variance)])
    #     run_name = f"rainbow_mie_beautyshot/variancebrutevspost/post/{variance}"
    #     scene_dict = sunsky_rainbow.get_scene(CONFIG, cfg,isbrute)
    #     print(f"\n>>> STARTING BATCH: {run_name}")
    #     print(f">>> Target Path: {run_name}")
    #     run_render_bench(scene_dict, run_name=run_name, config=CONFIG,start_from_preview=False)





    # CONFIG = {
    #     'target_spp': 2048,
    #     'batch_size': 64,
    #     # 'res_w':512,
    #     # 'res_h':512,
    #     # 'res_w':1024,
    #     # 'res_h':1024,
    #     'res_w': 1024,
    #     'res_h': 1024,
    #     # 'res_w':8,
    #     # 'res_h':8,
    #
    #     'downscale_res': (1024, 1024),
    #     'show_previews': False,
    #     'denoise_enabled': False,
    # }
    #
    # cfg = Particle.droplet(num_angles=8192,
    #                        num_wavelengths=32,
    #                        num_phi_bins=360,
    #                        backend=BackendType.MIEPYTHON,
    #                        composition=[DropletComposition(DropletShape.SPHERE, 1.0, 12.0, 0.0)])
    # run_name = f"white_furnace"
    # scene_dict = white_furnace.get_scene(CONFIG, cfg)
    # print(f"\n>>> STARTING BATCH: {run_name}")
    # print(f">>> Target Path: {run_name}")
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG, start_from_preview=False)

    # for size in [6.0,12.0,18.0,25.0]:
    #     cfg = Particle.droplet(num_angles=8192,
    #                            num_wavelengths=32,
    #                            num_phi_bins=360,
    #                            backend=BackendType.MIEPYTHON,
    #                            composition=[DropletComposition(DropletShape.SPHERE, 1.0, size, 0.0)])
    #     run_name = f"glory_beautyshot/size_{size}"
    #     scene_dict = sunsky_glory.get_scene(CONFIG, cfg)
    #     print(f"\n>>> STARTING BATCH: {run_name}")
    #     print(f">>> Target Path: {run_name}")
    #     run_render_bench(scene_dict, run_name=run_name, config=CONFIG, start_from_preview=False)

    # for size in [6.0,12.0,18.0,25.0]:
    #     cfg = Particle.droplet(num_angles=8192,
    #                            num_wavelengths=32,
    #                            num_phi_bins=360,
    #                            backend=BackendType.MIEPYTHON,
    #                            composition=[DropletComposition(DropletShape.SPHERE, 1.0, size, 0.0)])
    #     run_name = f"corona_beautyshot/size_{size}"
    #     scene_dict = sunsky_corona.get_scene(CONFIG, cfg)
    #     print(f"\n>>> STARTING BATCH: {run_name}")
    #     print(f">>> Target Path: {run_name}")
    #     run_render_bench(scene_dict, run_name=run_name, config=CONFIG, start_from_preview=False)

    # CONFIG = {
    #     'target_spp': 640,
    #     'batch_size': 32,
    #     # 'res_w':512,
    #     # 'res_h':512,
    #     'res_w': 1024,
    #     'res_h': 1024,
    #     # 'res_w':64*RES_M,
    #     # 'res_h':35*RES_M,
    #     # 'res_w':8,
    #     # 'res_h':8,
    #
    #     'downscale_res': (1024, 1024),
    #     'show_previews': False,
    #     'denoise_enabled': False,
    # }
    #
    #
    # SUN_ELEVATION = 25.0
    # for SUN_ELEVATION in [25.0,45.0]:
    #     cfg = Particle.cuboctahedral(num_angles=1024,
    #                                 num_phi_bins=2048,
    #                                 num_wavelengths=12,
    #                                 backend=BackendType.DRJIT,
    #                                 sun_elevation_deg=SUN_ELEVATION,
    #                                 air_turbulence_factor=0.6,
    #                                 composition=[
    #                                     CuboctahedralComposition(CuboctahedralHabit.TUMBLING,0.6,800.0,0.005),
    #                                     CuboctahedralComposition(CuboctahedralHabit.PLATE,0.4,800.0,0.005)
    #                                 ])
    # SUN_ELEVATION =35.0
    # TURBULENCE = 0.7
    # cfg = Particle.hexagonal(
    #     num_angles=1024,
    #     num_phi_bins=2048,
    #     num_wavelengths=18,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=SUN_ELEVATION,
    #     air_turbulence_factor=TURBULENCE,
    #     composition=[
    #         HexagonalComposition(HexagonalHabit.TUMBLING, 0.4, 1500, 800, 0.0),
    #         HexagonalComposition(HexagonalHabit.PLATE, 0.3, 100, 500, 0.0),
    #         HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 0.2, 3000, 300, 0.00),
    #         HexagonalComposition(HexagonalHabit.PARRY, 0.1, 1000, 100, 0.00),
    #
    #     ]
    # )
    #)
        # run_name = f"halo_mars_elevation_{SUN_ELEVATION}"
        # scene_dict = Ice_sunsky.get_scene(CONFIG, cfg, sun_elevation_deg=SUN_ELEVATION)
        # run_render_bench(scene_dict, run_name=run_name, config=CONFIG,start_from_preview=False)



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

    # CONFIG = {
    #     'target_spp': 2048*2,
    #     'batch_size': 64,
    #     # 'res_w':512,
    #     # 'res_h':512,
    #     # 'res_w':1024,
    #     # 'res_h':1024,
    #     'res_w': 1024,
    #     'res_h': 1024,
    #     # 'res_w':8,
    #     # 'res_h':8,
    #
    #     'downscale_res': (1024, 1024),
    #     'show_previews': False,
    #     'denoise_enabled': False,
    # }

    phase_params = Particle.droplet(num_angles=4096,
                                    num_phi_bins=360,
                                    num_wavelengths=16,
                                    backend=BackendType.DRJIT,
                                    composition=[DropletComposition(DropletShape.SPHERE, 1.0, 300.0, 0.03)]
                                                 #DropletComposition(DropletShape.OBLATE, 0.5, 300.0, 0.01)]
    )

    run_name = f"rainbow_twinned_bshot/size300"
    dict = sunsky_rainbow.get_scene(CONFIG, phase_params)
    run_render_bench(dict, run_name=run_name, config=CONFIG,start_from_preview=False)





import mitsuba as mi

from config import HexagonalHabit, HexagonalComposition, DropletComposition
from src.core.config import Particle, BackendType, DropletShape

# --- VARIANT ---
mi.set_variant("llvm_spectral")

from accumulation_bench.engine import run_render_bench
from accumulation_bench.bench_scenes import Ice_sunsky, pure_rainbow,mitsuba3cornell,cornell_fog_sphere

# --- CONFIG ---
CONFIG = {
    'target_spp': 512*8,
    'batch_size': 64,
    'res_w': 1024,
    'res_h': 1024,
    'downscale_res': (512, 512),
    'show_previews': False,
    'denoise_enabled': False,
}

if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a cpu to burn.")

    # mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 50.0, 0.0)],
    #                         num_angles=4096,
    #                         num_wavelengths=6,
    #                         num_phi_bins=360,
    #                         backend=BackendType.DRJIT)
    #
    # run_name = f"mitsuba3cornell/RTSphereLight_emitter_facing_wall/beam"
    # scene_dict = cornell_fog_sphere.get_scene(CONFIG,mist)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)

    mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 10.0, 0.0)],
                            num_angles=4096,
                            num_wavelengths=6,
                            num_phi_bins=360,
                            backend=BackendType.MIEPYTHON)

    run_name = f"mitsuba3cornell/MieSphereLight_emitter_facing_wall_final/10um"
    scene_dict = cornell_fog_sphere.get_scene(CONFIG, mist)
    run_render_bench(scene_dict, run_name=run_name, config=CONFIG)


    # #CBOX default:
    # run_name = f"mitsuba3cornell/HGSphere09/"
    # scene_dict = cornell_fog_sphere.get_scene(CONFIG)
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)


    # #CBOX default:
    # run_name = f"mitsuba3cornell/default/"
    # scene_dict = mitsuba3cornell.get_scene(CONFIG)
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
    # run_name = f"rainbow_rt/size_500/"
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


    # SUN_ELEVATION = 22.0
    # TURBULENCE = 1
    # cfg = Particle.hexagonal(
    #     num_angles=4096,
    #     num_phi_bins=1080,
    #     num_wavelengths=6,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=SUN_ELEVATION,
    #     air_turbulence_factor=TURBULENCE,
    #     composition=[
    #         HexagonalComposition(HexagonalHabit.TUMBLING, 0.3, 1500, 800, 0.0),
    #         HexagonalComposition(HexagonalHabit.PLATE, 0.3, 800, 1500, 0.0),
    #         HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 0.2, 1500, 800, 0.05),
    #         HexagonalComposition(HexagonalHabit.PARRY, 0.2, 1500, 800, 0.05)
    #     ]
    # )

    # run_name = f"Halo_Mixed/turb_{TURBULENCE}/elev_{SUN_ELEVATION}"
    # scene_dict = Ice_sunsky.get_scene(CONFIG, cfg, sun_elevation_deg=SUN_ELEVATION)
    #
    # print(f"\n>>> STARTING BATCH: HALO Turbulence={TURBULENCE}, elevation={SUN_ELEVATION}")
    # print(f">>> Target Path: {run_name}")
    #
    # run_render_bench(scene_dict, run_name=run_name, config=CONFIG)

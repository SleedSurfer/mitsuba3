import mitsuba as mi

from config import HexagonalHabit, HexagonalComposition, DropletComposition
from src.core.config import Particle, BackendType, DropletShape

# --- VARIANT ---
mi.set_variant("llvm_spectral")

from accumulation_bench.engine import run_render_bench
from accumulation_bench.bench_scenes import street_lamps, brocken_spectre, pure_rainbow,  cornell_exact, Ice, Ice_sunsky
from accumulation_bench.bench_scenes_win import pure_rainbow as rainbow

# --- CONFIG ---
CONFIG = {
    'target_spp': 512*2,
    'batch_size': 512,
    'res_w': 1024,
    'res_h': 1024,
    'downscale_res': (1920, 1080),
    'show_previews': True,
    'denoise_enabled': False,
}

if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a cpu to burn.")

    # cfg = Particle.hexagonal(
    #     num_angles=8192,
    #     num_phi_bins=360,
    #     num_wavelengths=6,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=0.0,
    #     air_turbulence_factor=1.0,
    #     composition=[HexagonalComposition(HexagonalHabit.PARRY, 1.0, 1500, 800, 0.0)]
    # )
    # for phi in [360,720,1440,2880,3600]:
    #     cfg.num_phi_bins = phi
    #     scene_dict = Ice_sunsky.get_scene(CONFIG, cfg)
    #     run_name = f"Halo_Tumbling_phitest/phi_{phi}"
    #     run_render_bench(
    #         scene_dict,
    #         run_name=run_name,
    #         config=CONFIG,
    #     )


    # #===============================
    # #      BASELINE RENDERS MIE
    # #===============================
    #
    # cfg = Particle.droplet(
    #     num_angles=8192,
    #     num_phi_bins=360,
    #     num_wavelengths=33,
    #     backend=BackendType.MIEPYTHON,
    #     composition=[DropletComposition(DropletShape.SPHERE, 1.0, 70.0, 0.0)],
    # )
    # # I'll let you fill these in as planned
    # sizes = [400.0] #[50.0, 70.0, 80.0, 100.0, 400.0]
    # variances = [0.1, 0.15, 0.3]#[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.3]
    #
    # for r in sizes:
    #     for v in variances:
    #         cfg.composition[0] = cfg.composition[0]._replace(radius_mean_um=r, variance=v)
    #         r_str = str(r).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"rainbow_mie/size_{r_str}/var_{v_str}"
    #         scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    #
    #         print(f"\n>>> STARTING BATCH: Radius={r}um, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(
    #             scene_dict,
    #             run_name=run_name,
    #             config=CONFIG,
    #         )


    # # ===============================
    # #      JIT RT TESTS
    # # ===============================
    #
    # cfg = Particle.droplet(
    #     num_angles=8192,
    #     num_phi_bins=360,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     composition=[DropletComposition(DropletShape.SPHERE, 1.0, 70.0, 0.0)],
    # )
    # sizes = [800, 900, 1100, 1800, 2500]#[70.0, 80.0, 100.0, 400.0, 500, 600, 800, 900, 1100, 1800, 2500] #[50.0, 70.0, 80.0, 100.0, 400.0, 500, 600, 800, 900, 1100, 1800, 2500]
    # variances = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.3]
    #
    # for r in sizes:
    #     for v in variances:
    #         cfg.composition[0] = cfg.composition[0]._replace(radius_mean_um=r, variance=v)
    #         r_str = str(r).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"rainbow_rt/size_{r_str}/var_{v_str}"
    #         scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    #
    #         print(f"\n>>> STARTING BATCH: Radius={r}um, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(
    #             scene_dict,
    #             run_name=run_name,
    #             config=CONFIG,
    #         )
    #
    # # ===============================
    # #      JIT RT TESTS oblate
    # # ===============================
    #
    # cfg = Particle.droplet(
    #     num_angles=8192,
    #     num_phi_bins=1440,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     composition=[DropletComposition(DropletShape.OBLATE, 1.0, 70.0, 0.0)],
    # )
    # sizes = [50.0, 70.0, 80.0, 100.0, 400.0, 500, 600, 800, 900, 1100, 1800, 2500]
    # variances = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.3]
    #
    # for r in sizes:
    #     for v in variances:
    #         cfg.composition[0] = cfg.composition[0]._replace(radius_mean_um=r, variance=v)
    #         r_str = str(r).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"rainbow_rt_oblate/size_{r_str}/var_{v_str}"
    #         scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    #
    #         print(f"\n>>> STARTING BATCH: Radius={r}um, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(
    #             scene_dict,
    #             run_name=run_name,
    #             config=CONFIG,
    #         )
    #
    #
    # # ===============================
    # #      JIT RT TESTS twinned
    # # ===============================
    #
    # cfg = Particle.droplet(
    #     num_angles=8192,
    #     num_phi_bins=1440,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     composition=[DropletComposition(DropletShape.OBLATE, 0.5, 70.0, 0.0), DropletComposition(DropletShape.SPHERE, 0.5, 70.0, 0.0)],
    # )
    # sizes = [50.0, 70.0, 80.0, 100.0, 400.0, 500, 600, 800, 900, 1100, 1800, 2500]
    # variances = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.3]
    #
    # for r in sizes:
    #     for v in variances:
    #         cfg.composition[0] = cfg.composition[0]._replace(radius_mean_um=r, variance=v)
    #         cfg.composition[1] = cfg.composition[1]._replace(radius_mean_um=r, variance=v)
    #         r_str = str(r).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"rainbow_twinned/size_{r_str}/var_{v_str}"
    #         scene_dict = pure_rainbow.get_scene(CONFIG, cfg)
    #
    #         print(f"\n>>> STARTING BATCH: Radius={r}um, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(
    #             scene_dict,
    #             run_name=run_name,
    #             config=CONFIG,
    #         )

    # ===============================
    #        ICE HALO BENCH
    # ===============================

    # cfg = Particle.hexagonal(
    #     num_angles=8192,
    #     num_phi_bins=360,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=0.0,
    #     air_turbulence_factor=1.0,
    #     composition=[HexagonalComposition(HexagonalHabit.TUMBLING, 1.0, 1500, 800, 0.0)]
    # )
    #
    # turbulences = [0.0, 0.5, 1.0, 1.05, 1.1, 1.5, 2.0, 3.0]
    #
    # # Variance: Keep these values much smaller for ice!
    # # Ice halos are delicate. 0.1 will completely nuke the halo into fog.
    # variances = [0.0, 0.001, 0.002, 0.01, 0.05, 0.1]
    #
    # for t in turbulences:
    #     cfg.air_turbulence_factor = t
    #     for v in variances:
    #         cfg.composition[0] = cfg.composition[0]._replace(variance=v)
    #
    #         t_str = str(t).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"Halo_Tumbling/turb_{t_str}/var_{v_str}"
    #         scene_dict = Ice_sunsky.get_scene(CONFIG, cfg)
    #
    #         print(f"\n>>> STARTING BATCH: Turbulence={t}, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(
    #             scene_dict,
    #             run_name=run_name,
    #             config=CONFIG,
    #         )
    #
    #
    # # ===============================
    # #        ICE Parhelic BENCH
    # # ===============================
    #
    # cfg = Particle.hexagonal(
    #     num_angles=8192,
    #     num_phi_bins=360,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=0.0,
    #     air_turbulence_factor=1.0,
    #     composition=[HexagonalComposition(HexagonalHabit.PLATE, 1.0, 800, 1500, 0.0)]
    # )
    #
    # turbulences = [0.0, 0.5, 1.0, 1.05, 1.1, 1.5, 2.0, 3.0]
    #
    # # Variance: Keep these values much smaller for ice!
    # # Ice halos are delicate. 0.1 will completely nuke the halo into fog.
    # variances = [0.0, 0.001, 0.002, 0.01, 0.05, 0.1]
    #
    # for t in turbulences:
    #     cfg.air_turbulence_factor = t
    #     for v in variances:
    #         cfg.composition[0] = cfg.composition[0]._replace(variance=v)
    #
    #         t_str = str(t).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"Halo_Parhelia/turb_{t_str}/var_{v_str}"
    #         scene_dict = Ice_sunsky.get_scene(CONFIG, cfg)
    #
    #         print(f"\n>>> STARTING BATCH: Turbulence={t}, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(
    #             scene_dict,
    #             run_name=run_name,
    #             config=CONFIG,
    #         )

    # # ===============================
    # #        ICE Column BENCH
    # # ===============================
    #
    # cfg = Particle.hexagonal(
    #     num_angles=2048,
    #     num_phi_bins=360,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=0.0,
    #     air_turbulence_factor=1.0,
    #     composition=[HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 1.0, 1500, 800, 0.0)]
    # )
    #
    # turbulences = [0.0, 0.5, 1.0, 1.05, 1.1, 1.5, 2.0, 3.0]
    #
    # # Variance: Keep these values much smaller for ice!
    # # Ice halos are delicate. 0.1 will completely nuke the halo into fog.
    # variances = [0.0, 0.001, 0.002, 0.01, 0.05, 0.1]
    #
    # for t in turbulences:
    #     cfg.air_turbulence_factor = t
    #     for v in variances:
    #         cfg.composition[0] = cfg.composition[0]._replace(variance=v)
    #
    #         t_str = str(t).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"Halo_Columns/turb_{t_str}/var_{v_str}"
    #         scene_dict = Ice_sunsky.get_scene(CONFIG, cfg)
    #
    #         print(f"\n>>> STARTING BATCH: Turbulence={t}, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(
    #             scene_dict,
    #             run_name=run_name,
    #             config=CONFIG,
    #         )
    #
    # # ===============================
    # #        ICE Column BENCH
    # # ===============================
    #
    # cfg = Particle.hexagonal(
    #     num_angles=2048,
    #     num_phi_bins=360,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=0.0,
    #     air_turbulence_factor=1.0,
    #     composition=[HexagonalComposition(HexagonalHabit.PARRY, 1.0, 1500, 800, 0.0)]
    # )
    #
    # turbulences = [0.0, 0.5, 1.0, 1.05, 1.1, 1.5, 2.0, 3.0]
    #
    # # Variance: Keep these values much smaller for ice!
    # # Ice halos are delicate. 0.1 will completely nuke the halo into fog.
    # variances = [0.0, 0.001, 0.002, 0.01, 0.05, 0.1]
    #
    # for t in turbulences:
    #     cfg.air_turbulence_factor = t
    #     for v in variances:
    #         cfg.composition[0] = cfg.composition[0]._replace(variance=v)
    #
    #         t_str = str(t).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"Halo_Parry/turb_{t_str}/var_{v_str}"
    #         scene_dict = Ice_sunsky.get_scene(CONFIG, cfg)
    #
    #         print(f"\n>>> STARTING BATCH: Turbulence={t}, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(
    #             scene_dict,
    #             run_name=run_name,
    #             config=CONFIG,
    #         )

    # ===============================
    #        ICE COMPOSITES
    # ===============================


    # # --- MIX 1: The "Standard" Display ---
    # # 60% Tumbling (22° Halo) + 40% Plates (Sun Dogs & CZA)
    # cfg_standard = Particle.hexagonal(
    #     num_angles=8192,
    #     num_phi_bins=360,
    #     num_wavelengths=33,
    #     backend=BackendType.DRJIT,
    #     sun_elevation_deg=0.0,
    #     air_turbulence_factor=1.0,
    #     composition=[
    #         HexagonalComposition(HexagonalHabit.TUMBLING, 0.6, 1500, 800, 0.0),
    #         HexagonalComposition(HexagonalHabit.PLATE, 0.4, 800, 1500, 0.0)
    #     ]
    # )
    #
    # for t in turbulences:
    #     cfg_standard.air_turbulence_factor = t
    #     for v in variances:
    #         # Update variance for EVERY habit in the composite simultaneously
    #         cfg_standard.composition = [
    #             comp._replace(variance=v) for comp in cfg_standard.composition
    #         ]
    #
    #         t_str = str(t).replace('.', '-')
    #         v_str = str(v).replace('.', '-')
    #
    #         run_name = f"Halo_Mix_Standard/turb_{t_str}/var_{v_str}"
    #         scene_dict = Ice_sunsky.get_scene(CONFIG, cfg_standard)
    #
    #         print(f"\n>>> STARTING BATCH: Standard Mix | Turbulence={t}, Variance={v}")
    #         print(f">>> Target Path: {run_name}")
    #
    #         run_render_bench(scene_dict, run_name=run_name, config=CONFIG)

    # --- MIX 2: The "God Mode" Album Cover ---
    # The kitchen sink. You saw this one earlier.

    turbulences = [0.0, 1.0, 2.0, 3.0, 4.0]
    sun_elevations = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    variances = [0.0, 0.001, 0.008, 0.012, 0.03, 0.06, 0.09, 0.12, 0.15, 0.3]
    cfg_god_mode = Particle.hexagonal(
        num_angles=4096,
        num_phi_bins=360,
        num_wavelengths=16,
        backend=BackendType.DRJIT,
        sun_elevation_deg=0.0,
        air_turbulence_factor=1.0,
        composition=[
            HexagonalComposition(HexagonalHabit.TUMBLING, 0.4, 1500, 800, 0.0),
            HexagonalComposition(HexagonalHabit.PLATE, 0.25, 800, 1500, 0.0),
            HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 0.2, 1500, 800, 0.0),
            HexagonalComposition(HexagonalHabit.PARRY, 0.15, 1500, 800, 0.0)
        ]
    )

    for t in turbulences:
        cfg_god_mode.air_turbulence_factor = t
        for e in sun_elevations:
            cfg_god_mode.sun_elevation_deg = e
            for v in variances:
                cfg_god_mode.composition = [
                    comp._replace(variance=v) for comp in cfg_god_mode.composition
                ]
                t_str = str(t).replace('.', '-')
                e_str = str(e).replace('.', '-')
                v_str = str(v).replace('.', '-')

                run_name = f"Halo_Mix_GodMode/turb_{t_str}/elev_{e_str}/var{v_str}"
                scene_dict = Ice_sunsky.get_scene(CONFIG, cfg_god_mode)

                print(f"\n>>> STARTING BATCH: God Mode Mix | Turbulence={t}, elevation={e}")
                print(f">>> Target Path: {run_name}")

                run_render_bench(scene_dict, run_name=run_name, config=CONFIG)

    # cfg = Particle.hexagonal(
    #     composition=[
    #         HexagonalComposition(HexagonalHabit.TUMBLING, 0.4, 1500, 800,0.0),
    #         HexagonalComposition(HexagonalHabit.PLATE, 0.2, 600, 1500,0.0),
    #         HexagonalComposition(HexagonalHabit.COLUMN_HORIZONTAL, 0.25, 1800, 700,0.0),
    #         HexagonalComposition(HexagonalHabit.PARRY, 0.15, 1800, 700,0.0)
    #     ],
    #     num_wavelengths=32,
    #     num_angles=2048,
    #     num_phi_bins=2048,  # Good, keep this high for the Parry arcs
    #     sun_elevation_deg=22.0,
    #     backend=BackendType.DRJIT,
    # )
    # #scene_dict = Ice.get_scene(CONFIG, cfg)
    # scene_dict = Ice_sunsky.get_scene(CONFIG, cfg)
    # run_render_bench(
    #     scene_dict,
    #     run_name="halo_combined_sky_t",
    #     config=CONFIG,
    # )

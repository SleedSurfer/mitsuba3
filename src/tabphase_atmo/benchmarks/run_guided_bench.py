import json
import os
import math
import time
from random import randint
from typing import List

import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import progressbar

# --- VARIANT ---
mi.set_variant('llvm_spectral')

# Your custom guiding imports
from practical_path_guiding.src.path_guiding_integrator import PathGuidingIntegrator
from practical_path_guiding.src.file_name_manager import FileNameManager
from practical_path_guiding.src.common import PerformanceData, printTitle, printBoldUnderLine

# Your bench imports
from accumulation_bench.bench_scenes import Ice_sunsky, pure_rainbow, mitsuba3cornell, cornell_fog_sphere
import accumulation_bench.bench_scenes.cornell_exact as fogbox
from src.core.config import Particle, BackendType, DropletShape,HexagonalHabit, HexagonalComposition, DropletComposition

# ==============================================================================
# GUIDING CONFIGURATION ZONE
# ==============================================================================
#TOTAL_BUDGET_SPP = 40000
TOTAL_BUDGET_SPP = 10000
#TOTAL_BUDGET_SPP = 64
TRAINING_SCHEDULE = [64, 128, 256, 512, 512, 1024, 1024]
# TRAINING_SCHEDULE = [64, 128, 256, 512]

GUIDING_START_ITERATION = 0
GUIDING_RAMP_ITERATIONS = 1
SDTREE_PDF_PRIOR = 0.1
TRAINING_RADIANCE_CLAMP = 25000000.0
TRAINING_LOG_COMPRESSION = False
FINAL_BATCH_SPP = 64
RECORD_PERFORMANCE = True

# Optional: Set to None if you don't have a GT for these dynamic scenes yet
GROUND_TRUTH_FILE = None

# --- BENCH CONFIG ---
CONFIG = {
    'target_spp': -1,
    'batch_size': 32,
    'res_w': 128,
    'res_h': 128,
    'downscale_res': (512, 512),
    'show_previews': False,
    'denoise_enabled': False,
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_cumm_targets(schedule: List[int], budget: int) -> List[int]:
    targets = []
    current = 0
    for s in schedule:
        current += s
        if current > budget: break
        targets.append(current)
    return targets


def get_evenly_spaced_targets(total: int, count: int) -> List[int]:
    if total <= 0 or count <= 0:
        return []
    step_targets = [math.ceil(total * (i / count)) for i in range(1, count + 1)]
    return sorted(set(min(total, max(1, t)) for t in step_targets))


# ==============================================================================
# THE GUIDED RUNNER
# ==============================================================================
def run_guided_bench(scene_dict: dict, run_name: str, config: dict):
    printTitle(f'Initializing Path Guiding Render: {run_name}')

    # 1. THE HIJACK: Rip out the vanilla integrator and inject the guided one
    print("Injecting path_guiding_integrator into scene dictionary...")
    scene_dict['integrator'] = {
        'type': 'path_guiding_integrator',
        'max_depth': 32,  # Tweak if your wave optics need more/less bounces
    }

    # Load the hijacked scene
    scene = mi.load_dict(scene_dict)
    sensors = scene.sensors()
    film_size = sensors[0].film().size()

    # 2. Setup File Management
    FileNameManager.setSceneName(run_name.replace('/', '_'))
    FileNameManager.createDebugFolder()

    # 3. Ground Truth Handling (Optional now)
    groundTruth = None
    if GROUND_TRUTH_FILE and os.path.exists(GROUND_TRUTH_FILE):
        raw_bmp = mi.Bitmap(GROUND_TRUTH_FILE)
        clean_bmp = raw_bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
        resized_bmp = clean_bmp.resample([film_size[0], film_size[1]])
        groundTruthImage = mi.TensorXf(resized_bmp)
        groundTruth = dr.unravel(mi.Color3f, groundTruthImage.array)
        print("Ground Truth loaded for MSE tracking.")
    else:
        print("No Ground Truth found/specified. Flying blind without MSE metrics.")

    # 4. Integrator Setup
    pathGuidingIntegrator: PathGuidingIntegrator = scene.integrator()
    bbox: mi.ScalarBoundingBox3f = scene.bbox()
    epsilon = 1e-4

    pathGuidingIntegrator.setup(
        numRays=(film_size[0] * film_size[1]),
        bbox_min=bbox.min - epsilon,
        bbox_max=bbox.max + epsilon,
        sdTreeMaxDepth=6,  # Locked at 5 so it doesn't overfit into a laser
        quadTreeMaxDepth=6,  # Locked at 5 so it doesn't overfit into a laser
        isStoreNEERadiance=True,
        bsdfSamplingFraction=0.5,
        guidingStartIteration=GUIDING_START_ITERATION,
        guidingRampIterations=GUIDING_RAMP_ITERATIONS,
        sdtreePdfPrior=SDTREE_PDF_PRIOR,
        trainingRadianceClamp=TRAINING_RADIANCE_CLAMP,
        trainingLogCompression=TRAINING_LOG_COMPRESSION
    )

    initial_seed = randint(0, 1000000)
    printBoldUnderLine('Initial seed:', initial_seed)

    # ---------------------------------------------------------
    # JSON Camera Data Extraction for Plotter Decoupling
    # ---------------------------------------------------------
    print("Extracting camera telemetry for TreePlotter...")
    try:
        # Get compiled matrix and resolution directly from the active sensor
        active_sensor = scene.sensors()[0]
        to_world_matrix = active_sensor.world_transform().matrix.numpy().tolist()
        res_x, res_y = active_sensor.film().size()

        # Safely recurse through the dictionary to find the FOV
        def find_fov(d):
            if isinstance(d, dict):
                if 'fov' in d: return d['fov']
                for v in d.values():
                    res = find_fov(v)
                    if res is not None: return res
            return None

        fov = find_fov(scene_dict)
        if fov is None:
            fov = 39.3077  # Safe Mitsuba default fallback
            print("Warning: FOV not explicitly found in scene_dict. Falling back to default.")

        camera_config = {
            "res_x": float(res_x),
            "res_y": float(res_y),
            "fov": float(fov),
            "to_world": to_world_matrix
        }

        # Step back one directory from the performance folder to hit the run's root
        run_root_dir = os.path.abspath(os.path.join(FileNameManager.PERFORMANCE_FOLDER_PATH, os.pardir))
        config_path = os.path.join(run_root_dir, 'camera_config.json')

        with open(config_path, 'w') as f:
            json.dump(camera_config, f, indent=4)
        print(f"Camera config successfully dumped to: {config_path}")

    except Exception as e:
        print(f"Failed to dump camera config: {e}")

    # 5. Performance Tracking Setup
    perf_variance_inIter = PerformanceData()
    perf_variance_gt_inIter = PerformanceData()
    perf_mse_gt_inIter = PerformanceData()
    perf_variance_endIter = PerformanceData()
    perf_variance_gt_endIter = PerformanceData()
    perf_mse_gt_endIter = PerformanceData()

    cumm_spp = 0
    cumm_time = 0.0
    iteration_count = 0
    final_image = None
    cumm_spp_targets = get_cumm_targets(TRAINING_SCHEDULE, TOTAL_BUDGET_SPP)

    # ==========================================================================
    # PHASE 1: SD-TREE TRAINING
    # ==========================================================================
    printTitle('PHASE 1: SD-Tree Training')

    for iter_spp in TRAINING_SCHEDULE:
        if cumm_spp + iter_spp >= TOTAL_BUDGET_SPP:
            print(f"Warning: Schedule exceeds budget. Halting training early at iter {iteration_count}.")
            break

        printTitle(f'Training Iteration {iteration_count}')
        print(f'Iteration SPP: {iter_spp} | Cumulative SPP: {cumm_spp} | Budget Left: {TOTAL_BUDGET_SPP - cumm_spp}')

        start_time = time.perf_counter()
        pathGuidingIntegrator.resetVarianceCounter()
        pathGuidingIntegrator.setIteration(iteration_count, isFinalIter=False)

        curr_iter_image = None
        spp_per_pass = 1
        iter_passes = iter_spp

        pbar = progressbar.ProgressBar(maxval=iter_passes,
                                       widgets=[progressbar.Bar('=', 'Render [', ']'), ' ', progressbar.Percentage()])
        pbar.start()

        for pass_i in range(iter_passes):
            image_pass = mi.render(scene=scene, spp=spp_per_pass, seed=initial_seed + cumm_spp + pass_i)
            weight = 1.0 / iter_passes
            if curr_iter_image is None:
                curr_iter_image = image_pass * weight
            else:
                curr_iter_image += image_pass * weight
            dr.eval(curr_iter_image)

            # Performance Tracking
            if RECORD_PERFORMANCE:
                current_pass_cumm = cumm_spp + pass_i + 1
                var_self = pathGuidingIntegrator.computeVariance(pass_i + 1)
                elapsed = (time.perf_counter() - start_time) + cumm_time
                perf_variance_inIter.append(elapsed, pass_i + 1, current_pass_cumm, iteration_count, var_self)

                if groundTruth is not None:
                    var_gt = pathGuidingIntegrator.computeVariance(pass_i + 1, groundTruth)
                    mse_gt = pathGuidingIntegrator.computeMSE(pass_i + 1, groundTruth)
                    perf_variance_gt_inIter.append(elapsed, pass_i + 1, current_pass_cumm, iteration_count, var_gt)
                    perf_mse_gt_inIter.append(elapsed, pass_i + 1, current_pass_cumm, iteration_count, mse_gt)

            pbar.update(pass_i + 1)
        pbar.finish()

        cumm_spp += iter_spp
        cumm_time += time.perf_counter() - start_time

        # End of Iteration Saves
        img_name = FileNameManager.generateImageFileName(iteration_count, iter_spp)
        mi.util.write_bitmap(f"{img_name}_cumm_spp-{cumm_spp}.exr", curr_iter_image)
        tree_name = FileNameManager.generateTreeDataFileName(iteration_count)
        pathGuidingIntegrator.saveSDTreeToFile(tree_name)

        pathGuidingIntegrator.refineAndPrepareSDTreeForNextIteration()
        iteration_count += 1
        final_image = curr_iter_image

    # ==========================================================================
    # PHASE 2: FINAL RENDER
    # ==========================================================================
    remaining_spp = TOTAL_BUDGET_SPP - cumm_spp
    if remaining_spp > 0:
        printTitle('PHASE 2: Final Guided Render')
        print(f'Remaining Budget: {remaining_spp} SPP using locked SD-Tree')

        start_time = time.perf_counter()
        pathGuidingIntegrator.resetVarianceCounter()
        pathGuidingIntegrator.setIteration(iteration_count, isFinalIter=True)

        final_iter_image = None
        num_passes = math.ceil(remaining_spp / FINAL_BATCH_SPP)
        spp_rendered = 0
        phase2_preview_targets = get_evenly_spaced_targets(remaining_spp, 4)
        phase2_preview_index = 0

        pbar = progressbar.ProgressBar(maxval=remaining_spp, widgets=[progressbar.Bar('=', 'Final Render [', ']'), ' ',
                                                                      progressbar.Percentage()])
        pbar.start()

        for pass_i in range(num_passes):
            current_batch_spp = min(FINAL_BATCH_SPP, remaining_spp - spp_rendered)
            image_pass = mi.render(scene=scene, spp=current_batch_spp, seed=initial_seed + cumm_spp + spp_rendered)

            weight = current_batch_spp / float(remaining_spp)
            if final_iter_image is None:
                final_iter_image = image_pass * weight
            else:
                final_iter_image += image_pass * weight

            dr.eval(final_iter_image)
            spp_rendered += current_batch_spp
            pbar.update(spp_rendered)

            # Previews
            while phase2_preview_index < len(phase2_preview_targets) and spp_rendered >= phase2_preview_targets[
                phase2_preview_index]:
                preview_spp = phase2_preview_targets[phase2_preview_index]
                preview_total_spp = cumm_spp + preview_spp
                current_unweighted = final_iter_image * (float(remaining_spp) / spp_rendered)
                preview_name = FileNameManager.generateImageFileName(iteration_count, preview_spp)
                mi.util.write_bitmap(
                    f"{preview_name}_preview-{phase2_preview_index + 1}_cumm_spp-{preview_total_spp}.exr",
                    current_unweighted)
                phase2_preview_index += 1

        pbar.finish()
        cumm_time += time.perf_counter() - start_time
        cumm_spp += remaining_spp
        final_image = final_iter_image

        img_name = FileNameManager.generateImageFileName(iteration_count, remaining_spp)
        mi.util.write_bitmap(f"{img_name}_FINAL_cumm_spp-{cumm_spp}.exr", final_image)

    # ==========================================================================
    # CSV EXPORTS
    # ==========================================================================
    printTitle('Render Complete. Saving Performance Logs.')
    if RECORD_PERFORMANCE:
        perf_variance_inIter.saveToFile(FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_inIter.csv')
        if groundTruth is not None:
            perf_variance_gt_inIter.saveToFile(FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_gt_inIter.csv')
            perf_mse_gt_inIter.saveToFile(FileNameManager.PERFORMANCE_FOLDER_PATH + 'mse_gt_inIter.csv')

    plt.axis('off')
    # Tone map for matplotlib preview
    plt.imshow(dr.unravel(mi.Color3f, final_image) ** (1.0 / 2.2))
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Wake the fuck up samurai, we have a cpu to burn.")

    mist = Particle.droplet(composition=[DropletComposition(DropletShape.SPHERE, 1.0, 50.0, 0.01)],
                            num_angles=4096,
                            num_wavelengths=17,
                            num_phi_bins=360,
                            backend=BackendType.MIEPYTHON)

    run_name = f"blowouts_guided"

    # Generate the scene dictionary from your bench code
    scene_dict = cornell_fog_sphere.get_scene(CONFIG, mist)

    # Pass it straight into the new guided bench wrapper
    run_guided_bench(scene_dict=scene_dict, run_name=run_name, config=CONFIG)
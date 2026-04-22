import math
import time
from random import randint
from typing import List

import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import progressbar

# Mitsuba Setup
mi.set_variant('llvm_spectral')

from practical_path_guiding.src.path_guiding_integrator import PathGuidingIntegrator
from practical_path_guiding.src.file_name_manager import FileNameManager
from practical_path_guiding.src.common import PerformanceData, printTitle, printBoldUnderLine

# ==============================================================================
# CONFIGURATION ZONE - Tweak your shit here, not deep in the spaghetti
# ==============================================================================
SCENE_FILE = 'scenes/volumetric-caustic/scene.xml'
GROUND_TRUTH_FILE = 'scenes/volumetric-caustic/TungstenRender.exr'
SCENE_NAME = 'volumetric-caustic'

TOTAL_BUDGET_SPP = 10000

# The absolute source of truth for your training phase.
# It will train for exactly these iterations, then use the remainder for the final render.
# Starts fat (32) to beat the unguided volumetric tax.
TRAINING_SCHEDULE = [64, 128, 256, 512, 1024, 1536, 2048]

FINAL_BATCH_SPP = 64  # How many SPP per pass during the final locked-tree render
RECORD_PERFORMANCE = True

if __name__ == '__main__':
    # ---------------------------------------------------------
    # 1. Initialization & Ground Truth Loading
    # ---------------------------------------------------------
    printTitle('Initializing Path Guiding Render')

    scene = mi.load_file(SCENE_FILE)
    sensors = scene.sensors()
    film_size = sensors[0].film().size()

    # Ground Truth Prep
    raw_bmp = mi.Bitmap(GROUND_TRUTH_FILE)
    clean_bmp = raw_bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
    resized_bmp = clean_bmp.resample([film_size[0], film_size[1]])
    groundTruthImage = mi.TensorXf(resized_bmp)
    groundTruth = dr.unravel(mi.Color3f, groundTruthImage.array)

    FileNameManager.setSceneName(SCENE_NAME)
    FileNameManager.createDebugFolder()

    # Integrator Setup
    pathGuidingIntegrator: PathGuidingIntegrator = scene.integrator()
    bbox: mi.ScalarBoundingBox3f = scene.bbox()
    epsilon = 1e-4

    pathGuidingIntegrator.setup(
        numRays=(film_size[0] * film_size[1]),
        bbox_min=bbox.min - epsilon,
        bbox_max=bbox.max + epsilon,
        sdTreeMaxDepth=32,
        quadTreeMaxDepth=32,
        isStoreNEERadiance=True,
        bsdfSamplingFraction=0.5
    )

    initial_seed = randint(0, 1000000)
    printBoldUnderLine('Initial seed:', initial_seed)

    # ---------------------------------------------------------
    # 2. Performance Tracking Setup
    # ---------------------------------------------------------
    perf_variance_inIter = PerformanceData()
    perf_variance_gt_inIter = PerformanceData()
    perf_mse_gt_inIter = PerformanceData()

    perf_variance_endIter = PerformanceData()
    perf_variance_gt_endIter = PerformanceData()
    perf_mse_gt_endIter = PerformanceData()
    perf_variance_estimated = PerformanceData()

    cumm_spp = 0
    cumm_time = 0.0
    iteration_count = 0
    final_image = None


    # Helper function to compute cumulative targets for logging
    def get_cumm_targets(schedule: List[int], budget: int) -> List[int]:
        targets = []
        current = 0
        for s in schedule:
            current += s
            if current > budget: break
            targets.append(current)
        return targets


    cumm_spp_targets = get_cumm_targets(TRAINING_SCHEDULE, TOTAL_BUDGET_SPP)

    # ==============================================================================
    # PHASE 1: TRAINING THE SD-TREE
    # ==============================================================================
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
        spp_per_pass = 1  # Must be 1 during training to record paths correctly
        iter_passes = iter_spp

        # Render Passes
        pbar = progressbar.ProgressBar(maxval=iter_passes,
                                       widgets=[progressbar.Bar('=', 'Render [', ']'), ' ', progressbar.Percentage()])
        pbar.start()

        for pass_i in range(iter_passes):
            image_pass = mi.render(scene=scene, spp=spp_per_pass, seed=initial_seed + cumm_spp + pass_i)

            # Accumulate average mathematically safely
            weight = 1.0 / iter_passes
            if curr_iter_image is None:
                curr_iter_image = image_pass * weight
            else:
                curr_iter_image += image_pass * weight

            dr.eval(curr_iter_image)

            # Intra-iteration Metrics
            if RECORD_PERFORMANCE:
                current_pass_cumm = cumm_spp + pass_i + 1
                var_self = pathGuidingIntegrator.computeVariance(pass_i + 1)
                var_gt = pathGuidingIntegrator.computeVariance(pass_i + 1, groundTruth)
                mse_gt = pathGuidingIntegrator.computeMSE(pass_i + 1, groundTruth)

                elapsed = (time.perf_counter() - start_time) + cumm_time
                perf_variance_inIter.append(elapsed, pass_i + 1, current_pass_cumm, iteration_count, var_self)
                perf_variance_gt_inIter.append(elapsed, pass_i + 1, current_pass_cumm, iteration_count, var_gt)
                perf_mse_gt_inIter.append(elapsed, pass_i + 1, current_pass_cumm, iteration_count, mse_gt)

            pbar.update(pass_i + 1)

        pbar.finish()
        cumm_spp += iter_spp
        cumm_time += time.perf_counter() - start_time

        # End of Iteration Metrics & Saving
        if RECORD_PERFORMANCE:
            var_self = pathGuidingIntegrator.computeVariance(iter_spp)
            var_gt = pathGuidingIntegrator.computeVariance(iter_spp, groundTruth)
            mse_gt = pathGuidingIntegrator.computeMSE(iter_spp, groundTruth)

            perf_variance_endIter.append(cumm_time, iter_spp, cumm_spp, iteration_count, var_self)
            perf_variance_gt_endIter.append(cumm_time, iter_spp, cumm_spp, iteration_count, var_gt)
            perf_mse_gt_endIter.append(cumm_time, iter_spp, cumm_spp, iteration_count, mse_gt)

            printBoldUnderLine('Variance:', var_self)
            printBoldUnderLine('MSE wrt GT:', mse_gt)

        # Output Data
        img_name = FileNameManager.generateImageFileName(iteration_count, iter_spp)
        mi.util.write_bitmap(f"{img_name}_cumm_spp-{cumm_spp}.png", curr_iter_image)
        mi.util.write_bitmap(f"{img_name}_cumm_spp-{cumm_spp}.exr", curr_iter_image)

        tree_name = FileNameManager.generateTreeDataFileName(iteration_count)
        pathGuidingIntegrator.saveSDTreeToFile(tree_name)
        obj_name = FileNameManager.generateOBJFileName(iteration_count)
        pathGuidingIntegrator.saveSDTreeOBJ(obj_name)

        # Refine the Tree! (No merge deletion allowed)
        pathGuidingIntegrator.refineAndPrepareSDTreeForNextIteration()

        iteration_count += 1
        final_image = curr_iter_image  # Keep latest just in case

    # ==============================================================================
    # PHASE 2: FINAL RENDER (EAT THE REMAINING BUDGET)
    # ==============================================================================
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

        pbar = progressbar.ProgressBar(maxval=remaining_spp, widgets=[progressbar.Bar('=', 'Final Render [', ']'), ' ',
                                                                      progressbar.Percentage()])
        pbar.start()

        for pass_i in range(num_passes):
            # Calculate batch size (handle the tail end of the budget correctly)
            current_batch_spp = min(FINAL_BATCH_SPP, remaining_spp - spp_rendered)

            image_pass = mi.render(scene=scene, spp=current_batch_spp, seed=initial_seed + cumm_spp + spp_rendered)

            # Accumulate heavily weighted by SPP fraction
            weight = current_batch_spp / float(remaining_spp)
            if final_iter_image is None:
                final_iter_image = image_pass * weight
            else:
                final_iter_image += image_pass * weight

            dr.eval(final_iter_image)
            spp_rendered += current_batch_spp

            pbar.update(spp_rendered)

            # Check if we crossed a cumulative milestone for saving intermediate EXRs
            current_total_spp = cumm_spp + spp_rendered
            if current_total_spp in cumm_spp_targets:
                temp_img_name = FileNameManager.generateImageFileName(iteration_count, spp_rendered)
                # Un-weight it temporarily just for the save
                current_unweighted = final_iter_image * (float(remaining_spp) / spp_rendered)
                mi.util.write_bitmap(f"{temp_img_name}_final_cumm_spp-{current_total_spp}.exr", current_unweighted)

        pbar.finish()
        cumm_time += time.perf_counter() - start_time
        cumm_spp += remaining_spp
        final_image = final_iter_image

        # Final Metrics
        if RECORD_PERFORMANCE:
            var_self = pathGuidingIntegrator.computeVariance(remaining_spp)
            mse_gt = pathGuidingIntegrator.computeMSE(remaining_spp, groundTruth)
            printBoldUnderLine('Final Variance:', var_self)
            printBoldUnderLine('Final MSE wrt GT:', mse_gt)

        # Final Save
        img_name = FileNameManager.generateImageFileName(iteration_count, remaining_spp)
        mi.util.write_bitmap(f"{img_name}_FINAL_cumm_spp-{cumm_spp}.png", final_image)
        mi.util.write_bitmap(f"{img_name}_FINAL_cumm_spp-{cumm_spp}.exr", final_image)

    # ==============================================================================
    # WRAP UP & CSV EXPORTS
    # ==============================================================================
    printTitle('Render Complete. Saving Performance Logs.')

    if RECORD_PERFORMANCE:
        perf_variance_inIter.saveToFile(FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_inIter.csv')
        perf_variance_gt_inIter.saveToFile(FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_groundTruth_inIter.csv')
        perf_mse_gt_inIter.saveToFile(FileNameManager.PERFORMANCE_FOLDER_PATH + 'mse_groundTruth_inIter.csv')

        perf_variance_endIter.saveToFile(FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_endIter.csv')
        perf_variance_gt_endIter.saveToFile(
            FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_groundTruth_endIter.csv')
        perf_mse_gt_endIter.saveToFile(FileNameManager.PERFORMANCE_FOLDER_PATH + 'mse_groundTruth_endIter.csv')

    # Display Result
    plt.axis('off')
    plt.imshow(final_image ** (1.0 / 2.2))
    plt.show()
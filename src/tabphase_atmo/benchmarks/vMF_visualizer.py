import numpy as np
import os
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import mitsuba as mi
import drjit as dr

from vMF.common import printTitle

mi.set_variant('llvm_spectral')

from vMF.Models.vMF_kdtree import vMF_KDTree
from practical_path_guiding.src.file_name_manager import FileNameManager
import json


class vMF_Plotter:
    def __init__(self, dataNumpy: np.array) -> None:
        """Loads the flat vMF arrays from the npz dictionary."""
        self.w0 = dataNumpy['vmf_w0']
        self.w1 = dataNumpy['vmf_w1']
        self.w2 = dataNumpy['vmf_w2']

        self.mu0_x = dataNumpy['vmf_mu0_x']
        self.mu0_y = dataNumpy['vmf_mu0_y']
        self.mu0_z = dataNumpy['vmf_mu0_z']

        self.mu1_x = dataNumpy['vmf_mu1_x']
        self.mu1_y = dataNumpy['vmf_mu1_y']
        self.mu1_z = dataNumpy['vmf_mu1_z']

        self.mu2_x = dataNumpy['vmf_mu2_x']
        self.mu2_y = dataNumpy['vmf_mu2_y']
        self.mu2_z = dataNumpy['vmf_mu2_z']

        self.k0 = dataNumpy['vmf_kappa0']
        self.k1 = dataNumpy['vmf_kappa1']
        self.k2 = dataNumpy['vmf_kappa2']

    def evaluate_pdf(self, leaf_idx: int, dir_x: np.ndarray, dir_y: np.ndarray, dir_z: np.ndarray) -> np.ndarray:
        """Evaluates the unnormalized PDF of the mixture across a grid of 3D directions (numpy vectors)."""

        def eval_lobe(w, mu_x, mu_y, mu_z, kappa):
            if kappa < 1e-4:
                return np.full_like(dir_x, w / (4.0 * np.pi))

            cos_t = (dir_x * mu_x) + (dir_y * mu_y) + (dir_z * mu_z)
            norm = kappa / (2.0 * np.pi * (1.0 - np.exp(-2.0 * kappa)))
            # cos_t - 1 is strictly <= 0, so this exp is numerically stable
            return w * norm * np.exp(kappa * (cos_t - 1.0))

        pdf0 = eval_lobe(self.w0[leaf_idx], self.mu0_x[leaf_idx], self.mu0_y[leaf_idx], self.mu0_z[leaf_idx],
                         self.k0[leaf_idx])
        pdf1 = eval_lobe(self.w1[leaf_idx], self.mu1_x[leaf_idx], self.mu1_y[leaf_idx], self.mu1_z[leaf_idx],
                         self.k1[leaf_idx])
        pdf2 = eval_lobe(self.w2[leaf_idx], self.mu2_x[leaf_idx], self.mu2_y[leaf_idx], self.mu2_z[leaf_idx],
                         self.k2[leaf_idx])

        return pdf0 + pdf1 + pdf2

    def plot_vmf_sphere(self, leaf_idx: int, title: str, ax: Axes3D) -> None:
        """Plots a 3D sphere colored by the PDF of the vMF mixture."""
        # Create a spherical mesh
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Evaluate PDF at every point on the sphere
        pdf_values = self.evaluate_pdf(leaf_idx, x, y, z)

        # Normalize colors to colormap
        pdf_max = np.max(pdf_values)
        if pdf_max > 0:
            colors = plt.cm.turbo(pdf_values / pdf_max)
        else:
            colors = plt.cm.turbo(np.zeros_like(pdf_values))

        # Plot the surface
        ax.plot_surface(x, y, z, facecolors=colors, rstride=2, cstride=2, antialiased=True, shade=False)

        # Draw axes for visual reference
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], color='r', alpha=0.5, linewidth=1)  # X
        ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='g', alpha=0.5, linewidth=1)  # Y
        ax.plot([0, 0], [0, 0], [-1.2, 1.2], color='b', alpha=0.5, linewidth=1)  # Z

        # Clean up the view
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        ax.set_title(title, pad=0, fontsize=10)


class KDTreePlotter:
    def __init__(self, fileName: str) -> None:
        treeDataNumpy = np.load(fileName)

        # THE FIX: Instantiate the new vMF tree and use its decomposed loader
        self.tree = vMF_KDTree()
        self.tree.loadFromFile(treeDataNumpy)

        # Point kdTreeNode to the loaded tree's node so your findLeafNode logic still works
        self.kdTreeNode = self.tree.kdTreeNode

        # Init the new vMF Plotter
        self.vmfPlotter = vMF_Plotter(treeDataNumpy)

    def findLeafNode(self, position: mi.Vector3f) -> Tuple[mi.UInt32, mi.Bool]:
        # (Your exact findLeafNode implementation from your code goes here)
        nodeIndex = dr.zeros(mi.UInt32, dr.width(position))
        rootNodeBBox = self.kdTreeNode.getBBox(0)
        isInsideRoot = rootNodeBBox.contains(position)
        active = mi.Bool(isInsideRoot)

        def loop_cond(active, nodeIndex):
            return active

        def loop_body(active, nodeIndex):
            isLeafNode = dr.gather(mi.Bool, self.kdTreeNode.isLeaf, nodeIndex, active)
            isNotLeafNode = ~isLeafNode
            active &= isNotLeafNode

            child_left_idx = dr.gather(mi.UInt32, self.kdTreeNode.child_left_index, nodeIndex, active)
            child_right_idx = dr.gather(mi.UInt32, self.kdTreeNode.child_right_index, nodeIndex, active)

            child_left_bbox = self.kdTreeNode.getBBox(child_left_idx)
            child_left_bbox_test = child_left_bbox.contains(position)
            nodeIndex[child_left_bbox_test & active] = child_left_idx

            child_right_bbox = self.kdTreeNode.getBBox(child_right_idx)
            child_right_bbox_test = child_right_bbox.contains(position)
            nodeIndex[child_right_bbox_test & active] = child_right_idx

            return (active, nodeIndex)

        active, nodeIndex = dr.while_loop(state=(active, nodeIndex), cond=loop_cond, body=loop_body)
        return nodeIndex, isInsideRoot

    def findHottestPositions(self, top_n: int = 5) -> List[List[float]]:
        # (Your exact findHottestPositions implementation from your code)
        leafNodeIndex_mi = dr.compress(self.kdTreeNode.isLeaf)
        leafNodeIndex_np = leafNodeIndex_mi.numpy()

        vertCounts_mi = dr.gather(mi.Float, self.kdTreeNode.vertCount, leafNodeIndex_mi)
        vertCounts_np = vertCounts_mi.numpy()

        top_indices_relative = np.argsort(vertCounts_np)[::-1][:top_n]
        top_leaf_indices = leafNodeIndex_np[top_indices_relative]

        hot_bboxes = self.kdTreeNode.getBBox(top_leaf_indices)

        centers_x = (hot_bboxes.min.x.numpy() + hot_bboxes.max.x.numpy()) / 2.0
        centers_y = (hot_bboxes.min.y.numpy() + hot_bboxes.max.y.numpy()) / 2.0
        centers_z = (hot_bboxes.min.z.numpy() + hot_bboxes.max.z.numpy()) / 2.0

        hot_positions = []
        for i in range(len(centers_x)):
            pos = [float(centers_x[i]), float(centers_y[i]), float(centers_z[i])]
            print(f"Discovered Hotspot {i + 1}: {pos} (Traffic: {vertCounts_np[top_indices_relative[i]]})")
            hot_positions.append(pos)

        return hot_positions

    def plotVMFAtPosition(self, position: List[float], title: str, ax: Axes3D) -> None:
        position_mi = mi.Vector3f(position)
        leafNodeIndex_mi, isValid_mi = self.findLeafNode(position_mi)
        leafNodeIndex = leafNodeIndex_mi.numpy()[0]
        isValid = isValid_mi.numpy()[0]

        if isValid:
            self.vmfPlotter.plot_vmf_sphere(leafNodeIndex, title, ax)
        else:
            ax.set_title("Position outside root BBox", fontsize=10)
            ax.axis('off')


class MultiIterationTreePlotter:
    def __init__(self, sceneName: str, numIteration: int) -> None:
        self.sceneName = sceneName
        self.numIteration = numIteration
        self.kdTreePlotters: List[KDTreePlotter] = []

        self.to_world, self.fov, self.res_x, self.res_y = self._parse_scene_camera()

        for i in range(numIteration):
            fileName = FileNameManager.generateTreeDataFileName(i, withNpzEnding=True)
            if os.path.exists(fileName):
                kdTreePlotter = KDTreePlotter(fileName)
                self.kdTreePlotters.append(kdTreePlotter)
            else:
                print(f"Warning: {fileName} not found. Skipping.")

    def plotHottestRegions(self, top_n: int = 3, saveFig: bool = False) -> None:
        printTitle(f"Auto-Discovering Top {top_n} Hottest Regions")
        if not self.kdTreePlotters:
            return
        final_kd_tree = self.kdTreePlotters[-1]
        hot_positions = final_kd_tree.findHottestPositions(top_n=top_n)

        for idx, pos in enumerate(hot_positions):
            print(f"\nGenerating plot for Hotspot {idx + 1} at {pos}...")
            self.plotTreeEvolutionAtPosition(pos, saveFig=saveFig, file_suffix=f"_hotspot_{idx + 1}")

    def plotTreeEvolutionAtPosition(self, position: List[float], saveFig: bool = False, file_suffix: str = "") -> None:
        numIter = len(self.kdTreePlotters)

        # THE FIX: We need mixed axes types. Top row is 2D images, bottom row is 3D scatter plots.
        fig = plt.figure(figsize=(numIter * 4, 8))
        fig.suptitle(
            f'{self.sceneName} Evolution\nTarget Hotspot: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]',
            fontsize=16, fontweight='bold')

        for index, kdTreePlotter in enumerate(self.kdTreePlotters):
            # Add 2D Image Subplot (Top Row)
            ax_render = fig.add_subplot(2, numIter, index + 1)
            ax_render.set_title(f'Render (Iter {index})')
            ax_render.axis('off')

            exr_format = f"{self.sceneName}_iter-{index}_*.exr"
            exr_search_pattern = os.path.join(FileNameManager.IMAGE_FOLDER_PATH, exr_format)
            exr_matching_files = glob.glob(exr_search_pattern)

            if exr_matching_files:
                exr_bmp = mi.Bitmap(exr_matching_files[0])
                converted_bmp = exr_bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
                img_array = np.array(converted_bmp)

                ax_render.imshow(img_array)
                px, py = self._project_world_to_pixel(position)

                if px is not None and py is not None:
                    ax_render.axvline(x=px, color='lime', linestyle='--', linewidth=1, alpha=0.6)
                    ax_render.axhline(y=py, color='lime', linestyle='--', linewidth=1, alpha=0.6)
                    target_radius = self.res_x * 0.03
                    target = patches.Circle((px, py), radius=target_radius, edgecolor='red', facecolor='none',
                                            linewidth=1.5)
                    ax_render.add_patch(target)
                else:
                    ax_render.text(0.5, 0.5, 'Hotspot Behind Camera', ha='center', va='center', color='red',
                                   transform=ax_render.transAxes)
            else:
                ax_render.text(0.5, 0.5, 'Image Not Found', ha='center', va='center')

            # Add 3D vMF Subplot (Bottom Row)
            ax_tree = fig.add_subplot(2, numIter, numIter + index + 1, projection='3d')
            tree_title = f'vMF Lobes (Iter {index})'
            kdTreePlotter.plotVMFAtPosition(position, tree_title, ax=ax_tree)

        fig.tight_layout()
        if saveFig:
            figFileName = os.path.join(FileNameManager.PLOT_FOLDER_PATH,
                                       f'{self.sceneName}{file_suffix}_evolution_pos_{position[0]:.2f}_{position[1]:.2f}_{position[2]:.2f}.png')
            plt.savefig(fname=figFileName, dpi=300, bbox_inches='tight')
            print(f"Saved timeline to {figFileName}")
            plt.close(fig)

    # (Keep your exact _parse_scene_camera and _project_world_to_pixel methods here)
    def _parse_scene_camera(self):
        run_root_dir = os.path.abspath(os.path.join(FileNameManager.PLOT_FOLDER_PATH, os.pardir))
        config_path = os.path.join(run_root_dir, 'camera_config.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"CRITICAL: camera_config.json not found at {config_path}")

        with open(config_path, 'r') as f:
            cam_data = json.load(f)

        to_world = np.array(cam_data['to_world']).reshape(4, 4)
        return to_world, float(cam_data['fov']), float(cam_data['res_x']), float(cam_data['res_y'])

    def _project_world_to_pixel(self, p_world: list[float]) -> tuple[float, float]:
        w2c = np.linalg.inv(self.to_world)
        p_w = np.array([p_world[0], p_world[1], p_world[2], 1.0])
        p_cam = w2c @ p_w

        if p_cam[2] <= 0:
            return None, None

        x_cam = p_cam[0] / p_cam[2]
        y_cam = p_cam[1] / p_cam[2]

        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        aspect = self.res_x / self.res_y

        x_ndc = x_cam / tan_half_fov
        y_ndc = y_cam / (tan_half_fov / aspect)

        pixel_x = (1.0 - x_ndc) * 0.5 * self.res_x
        pixel_y = (1.0 - y_ndc) * 0.5 * self.res_y

        return pixel_x, pixel_y


if __name__ == '__main__':
    sceneName = 'blowouts_guided'
    iteration = 5  # Set this to the number of iterations you actually ran

    # 1. Init the manager
    FileNameManager.setSceneName(sceneName)

    # 2. THE FIX: Force the file manager to anchor to the benchmarks folder
    # This assumes this visualizer script is located inside the 'benchmarks' folder.
    target_debug_dir = os.path.join(os.path.dirname(__file__), 'debug', sceneName)

    FileNameManager.DEBUG_FOLDER_PATH = target_debug_dir + '/'
    FileNameManager.IMAGE_FOLDER_PATH = os.path.join(target_debug_dir, 'image') + '/'
    FileNameManager.PLOT_FOLDER_PATH = os.path.join(target_debug_dir, 'plot') + '/'
    FileNameManager.PERFORMANCE_FOLDER_PATH = os.path.join(target_debug_dir, 'performance') + '/'
    FileNameManager.TREE_DATA_FOLDER_PATH = os.path.join(target_debug_dir, 'tree-data') + '/'

    # 3. Fire the plotter
    multiIterationTreePlotter = MultiIterationTreePlotter(sceneName, iteration)
    multiIterationTreePlotter.plotHottestRegions(top_n=3, saveFig=True)
from __future__ import annotations

import drjit as dr
import mitsuba as mi

mi.set_variant('llvm_spectral')
import json
import numpy as np
import os
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from typing import List, Tuple

from practical_path_guiding.src.common import *
from practical_path_guiding.src.quadtree import QuadTreeNode
from practical_path_guiding.src.kdtree import KDTreeNode
from practical_path_guiding.src.file_name_manager import FileNameManager


class QuadTreePlotter:
    def __init__(self, fileName: str) -> None:
        """
        QuadTree data represented as numpy array fashion.
        Mainly use for offline data processing.
        - fileName: file name of the QuadTree data to be loaded.
        """
        # Load from file
        treeDataNumpy = np.load(fileName)

        # Init the QuaddTreeNode and load data into
        self.quadTreeNode = QuadTreeNode()
        self.quadTreeNode.loadFromFile(treeDataNumpy)

    def getMaxDepth(self, rootIndex: int) -> int:
        """
        Get maximum depth of a QuadTree
        """
        leafNodeIndex = self.quadTreeNode.getAllLeafNodeIndex(mi.UInt32(rootIndex))
        leafNodeDepth = dr.gather(mi.UInt32, self.quadTreeNode.depth, leafNodeIndex)
        maxDepth = np.max(leafNodeDepth.numpy())

        return maxDepth

    def sampleIrradiance(self, rootIndex: mi.UInt32, position: mi.Vector2f) -> mi.Float:
        """
        Sample irradiance from a given postion. 'rootIndex' and 'position' MUST have
        the same size.
        - rootIndex: index of tree root (0, 1, 2, ...).
        - position: sampling position.
        """
        # Start searching at root node index
        nodeIndex = dr.gather(mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex)

        # Test if data is within the root node bbox
        rootNodeBBox = self.quadTreeNode.getBBox(nodeIndex)
        isInsideRoot = rootNodeBBox.contains(position)
        active = mi.Bool(isInsideRoot)

        # Traverse the tree to find the corresponding leaf node
        def loop_cond(active, nodeIndex):
            return active

        def loop_body(active, nodeIndex):
            # If at leaf node then stop, if not then continue
            isLeafNode = dr.gather(mi.Bool, self.quadTreeNode.isLeaf, nodeIndex, active)
            isNotLeafNode = ~isLeafNode
            active &= isNotLeafNode

            # Else, check which child node it belongs to and store the node index
            child_1_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_1_index, index=nodeIndex, active=active)
            child_2_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_2_index, index=nodeIndex, active=active)
            child_3_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_3_index, index=nodeIndex, active=active)
            child_4_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_4_index, index=nodeIndex, active=active)

            child_1_bbox = self.quadTreeNode.getBBox(child_1_idx)
            child_1_bbox_test = child_1_bbox.contains(position)
            nodeIndex[child_1_bbox_test & active] = child_1_idx

            child_2_bbox = self.quadTreeNode.getBBox(child_2_idx)
            child_2_bbox_test = child_2_bbox.contains(position)
            nodeIndex[child_2_bbox_test & active] = child_2_idx

            child_3_bbox = self.quadTreeNode.getBBox(child_3_idx)
            child_3_bbox_test = child_3_bbox.contains(position)
            nodeIndex[child_3_bbox_test & active] = child_3_idx

            child_4_bbox = self.quadTreeNode.getBBox(child_4_idx)
            child_4_bbox_test = child_4_bbox.contains(position)
            nodeIndex[child_4_bbox_test & active] = child_4_idx

            return (active, nodeIndex)

        active, nodeIndex = dr.while_loop(
            state=(active, nodeIndex),
            cond=loop_cond,
            body=loop_body
        )

        # Get irradiance from the corresponding leaf node index
        irradiance = dr.gather(mi.Float, self.quadTreeNode.irradiance, nodeIndex, isInsideRoot)

        # Compute area of the node
        nodeBBox = self.quadTreeNode.getBBox(nodeIndex)
        nodeBBoxExtent = (nodeBBox.max - nodeBBox.min)
        nodeArea = nodeBBoxExtent.x * nodeBBoxExtent.y

        # Normalized irradiance by the array size
        irradiance /= nodeArea

        return irradiance

    def plotQuadTree(self, rootIndex: int, title: str, ax=None) -> None:
        """
        Plot a QuadTree as heat map.
        - rootIndex: index of tree root (0, 1, 2, ...).
        """
        # Generate sampling position in a grid fashion
        depth = self.getMaxDepth(0)
        depth = min(depth, 10)
        numCell = pow(2, depth)

        cellSize = 1 / numCell
        cellCenterOffset = cellSize / 2
        x = dr.arange(mi.Float, numCell) / numCell
        y = dr.arange(mi.Float, numCell) / numCell
        x += cellCenterOffset
        y += cellCenterOffset

        x, y = dr.meshgrid(x, y)
        samplingPosition = mi.Vector2f(x, y)
        N = dr.width(samplingPosition)

        # Sample QuadTree irradiance
        rootIndexArray = dr.full(mi.UInt32, rootIndex, N)
        irradiance = self.sampleIrradiance(rootIndexArray, samplingPosition)
        irradiance_numpy = irradiance.numpy().reshape(numCell, numCell)

        # Plot heatmap
        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(irradiance_numpy, cmap='turbo', interpolation='nearest', extent=[0, 1, 0, 1], origin='lower')
        ax.set_xlabel('Normalized Φ (Phi)')
        ax.set_ylabel('Normalized Cos(θ) (Theta)')
        ax.set_title(title, pad=10)
        plt.colorbar(im, ax=ax, label='Irradiance (Log Scale)')


class KDTreePlotter:
    def __init__(self, fileName: str) -> None:
        # Load from file
        treeDataNumpy = np.load(fileName)

        # Init the KDTreeNode and load data into
        self.kdTreeNode = KDTreeNode()
        self.kdTreeNode.loadFromFile(treeDataNumpy)

        # Init QuadTreePlotter
        self.quadTreePlotter = QuadTreePlotter(fileName)

    def findLeafNode(self, position: mi.Vector3f) -> Tuple[mi.UInt32, mi.Bool]:
        """
        Find the corresponding leaf node index of the given position.
        """
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

        active, nodeIndex = dr.while_loop(
            state=(active, nodeIndex),
            cond=loop_cond,
            body=loop_body
        )
        return nodeIndex, isInsideRoot

    def findHottestPositions(self, top_n: int = 5) -> List[List[float]]:
        """
        Auto-discovers the most active spatial regions in the KD-Tree.
        """
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

    def plotQuadTreeAtPosition(self, position: List[float], title: str, ax=None) -> None:
        position_mi = mi.Vector3f(position)
        leafNodeIndex_mi, isValid_mi = self.findLeafNode(position_mi)
        leafNodeIndex = leafNodeIndex_mi.numpy()[0]
        isValid = isValid_mi.numpy()[0]

        if isValid:
            quadTreeRootIndex = dr.gather(mi.UInt32, self.kdTreeNode.quadTreeRootIndex, leafNodeIndex)
            self.quadTreePlotter.plotQuadTree(quadTreeRootIndex.numpy()[0], title, ax)


class MultiIterationTreePlotter:
    def __init__(self, sceneName: str, numIteration: int) -> None:
        self.sceneName = sceneName
        self.numIteration = numIteration
        self.kdTreePlotters: List[KDTreePlotter] = []

        self.to_world, self.fov, self.res_x, self.res_y = self._parse_scene_camera()

        for i in range(numIteration):
            fileName = FileNameManager.generateTreeDataFileName(i, withNpzEnding=True)
            kdTreePlotter = KDTreePlotter(fileName)
            self.kdTreePlotters.append(kdTreePlotter)

    def plotHottestRegions(self, top_n: int = 3, saveFig: bool = False) -> None:
        printTitle(f"Auto-Discovering Top {top_n} Hottest Regions")
        final_kd_tree = self.kdTreePlotters[-1]
        hot_positions = final_kd_tree.findHottestPositions(top_n=top_n)

        for idx, pos in enumerate(hot_positions):
            print(f"\nGenerating plot for Hotspot {idx + 1} at {pos}...")
            self.plotQuadTreeAtPosition(pos, saveFig=saveFig, file_suffix=f"_hotspot_{idx + 1}")

    def plotQuadTreeAtPosition(self, position: List[float], saveFig: bool = False, file_suffix: str = "") -> None:
        numIter = len(self.kdTreePlotters)
        fig, axes = plt.subplots(2, numIter, figsize=(numIter * 4, 8))
        if numIter == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        fig.suptitle(
            f'{self.sceneName} Evolution\nTarget Hotspot: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]',
            fontsize=16, fontweight='bold')

        for index, kdTreePlotter in enumerate(self.kdTreePlotters):
            ax_render = axes[0, index]
            ax_tree = axes[1, index]

            ax_render.set_title(f'Render (Iter {index})')
            ax_render.axis('off')

            exr_format = f"{self.sceneName}_iter-{index}_*.exr"
            exr_search_pattern = os.path.join(FileNameManager.IMAGE_FOLDER_PATH, exr_format)
            exr_matching_files = glob.glob(exr_search_pattern)

            file_format = f"{self.sceneName}_iter-{index}_*.png"
            search_pattern = os.path.join(FileNameManager.IMAGE_FOLDER_PATH, file_format)
            matching_files = glob.glob(search_pattern)

            if matching_files or exr_matching_files:
                if matching_files:
                    img_array = plt.imread(matching_files[0])
                else:
                    exr_bmp = mi.Bitmap(exr_matching_files[0])
                    converted_bmp = exr_bmp.convert(
                        mi.Bitmap.PixelFormat.RGB,
                        mi.Struct.Type.UInt8,
                        srgb_gamma=True
                    )
                    # THE FIX: Cast the Mitsuba Bitmap directly to a numpy array
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

            tree_title = f'QuadTree (Iter {index})'
            kdTreePlotter.plotQuadTreeAtPosition(position, tree_title, ax=ax_tree)

        fig.tight_layout()
        if saveFig:
            figFileName = os.path.join(FileNameManager.PLOT_FOLDER_PATH,
                                       f'{self.sceneName}{file_suffix}_evolution_pos_{position[0]:.2f}_{position[1]:.2f}_{position[2]:.2f}.png')
            plt.savefig(fname=figFileName, dpi=300, bbox_inches='tight')
            print(f"Saved timeline to {figFileName}")

    def _parse_scene_camera(self):
        run_root_dir = os.path.abspath(os.path.join(FileNameManager.PLOT_FOLDER_PATH, os.pardir))
        config_path = os.path.join(run_root_dir, 'camera_config.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"CRITICAL: camera_config.json not found at {config_path}")

        with open(config_path, 'r') as f:
            cam_data = json.load(f)

        # THE FIX: Force the 1D list back into a 4x4 square matrix
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
    iteration = 6

    FileNameManager.setSceneName(sceneName)
    multiIterationTreePlotter = MultiIterationTreePlotter(sceneName, iteration)
    multiIterationTreePlotter.plotHottestRegions(top_n=3, saveFig=True)
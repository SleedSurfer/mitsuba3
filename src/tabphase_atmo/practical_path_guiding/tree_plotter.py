from __future__ import annotations

import drjit as dr
import mitsuba as mi
mi.set_variant('llvm_spectral')

from practical_path_guiding.src.common import *

from practical_path_guiding.src.quadtree import QuadTreeNode
from practical_path_guiding.src.kdtree import KDTreeNode
from practical_path_guiding.src.file_name_manager import FileNameManager

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

class QuadTreePlotter:

	def __init__( self, fileName: str ) -> None:
		"""
			QuadTree data represented as numpy array fashion.
			Mainly use for offline data processing.
			- fileName: file name of the QuadTree data to be loaded.
		"""

		# Load from file
		treeDataNumpy = np.load(fileName)

		# Init the QuaddTreeNode and load data into
		self.quadTreeNode = QuadTreeNode()
		self.quadTreeNode.loadFromFile( treeDataNumpy )

	
	def getMaxDepth( self, rootIndex: int ) -> int:
		"""
			Get maximum depth of a QuadTree
			-
		"""
		leafNodeIndex = self.quadTreeNode.getAllLeafNodeIndex( mi.UInt32( rootIndex ) )
		leafNodeDepth = dr.gather( mi.UInt32, self.quadTreeNode.depth, leafNodeIndex )
		maxDepth = np.max(leafNodeDepth.numpy())

		return maxDepth

	
	def sampleIrradiance( self, rootIndex: mi.UInt32, position: mi.Vector2f ) -> mi.Float:
		"""
			Sample irradiance from a given postion. 'rootIndex' and 'position' MUST have
			the same size.
			- rootIndex: index of tree root (0, 1, 2, ...).
			- position: sampling position.
		"""
		# 	Start searching at root node index
		nodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

		# Test if data is within the root node bbox
		rootNodeBBox = self.quadTreeNode.getBBox( nodeIndex )
		isInsideRoot = rootNodeBBox.contains( position )
		active = mi.Bool( isInsideRoot )

		# Traverse the tree to find the corresponding leaf node
		def loop_cond(active, nodeIndex):
			return active

		def loop_body(active, nodeIndex):
			# If at leaf node then stop, if not then continue
			isLeafNode = dr.gather(mi.Bool, self.quadTreeNode.isLeaf, nodeIndex, active )
			isNotLeafNode = ~isLeafNode
			active &= isNotLeafNode

			# Else, check which child node it belongs to and store the node index
			child_1_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_1_index, index= nodeIndex, active= active)
			child_2_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_2_index, index= nodeIndex, active= active)
			child_3_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_3_index, index= nodeIndex, active= active)
			child_4_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_4_index, index= nodeIndex, active= active)

			child_1_bbox = self.quadTreeNode.getBBox( child_1_idx )
			child_1_bbox_test = child_1_bbox.contains( position )
			nodeIndex[child_1_bbox_test & active] = child_1_idx

			child_2_bbox = self.quadTreeNode.getBBox( child_2_idx )
			child_2_bbox_test = child_2_bbox.contains( position )
			nodeIndex[child_2_bbox_test & active] = child_2_idx

			child_3_bbox = self.quadTreeNode.getBBox( child_3_idx )
			child_3_bbox_test = child_3_bbox.contains( position )
			nodeIndex[child_3_bbox_test & active] = child_3_idx

			child_4_bbox = self.quadTreeNode.getBBox( child_4_idx )
			child_4_bbox_test = child_4_bbox.contains( position )
			nodeIndex[child_4_bbox_test & active] = child_4_idx

			return (active, nodeIndex)

		active, nodeIndex = dr.while_loop(
			state=(active, nodeIndex),
			cond=loop_cond,
			body=loop_body
		)

		# Get irradiance from the corresponding leaf node index
		irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, nodeIndex, isInsideRoot )
		# Compute area of the node
		nodeBBox = self.quadTreeNode.getBBox( nodeIndex )
		nodeBBoxExtent = (nodeBBox.max - nodeBBox.min)
		nodeArea = nodeBBoxExtent.x * nodeBBoxExtent.y

		# Normalized irradiance by the array size
		irradiance /= nodeArea

		return irradiance
	
	
	def plotQuadTree( self, rootIndex: int, title: str, ax = None ) -> None:
		"""
			Plot a QuadTree as heat map.
			- rootIndex: index of tree root (0, 1, 2, ...).
		"""

		# Generate sampling position in a grid fashion
		depth = self.getMaxDepth( 0 )
		depth = min( depth, 10 )
		numCell = pow( 2, depth )

		cellSize = 1 / numCell
		cellCenterOffset =  cellSize / 2
		x = dr.arange( mi.Float, numCell ) / numCell
		y = dr.arange( mi.Float, numCell ) / numCell
		x += cellCenterOffset
		y += cellCenterOffset

		x, y = dr.meshgrid( x, y )

		samplingPosition = mi.Vector2f( x, y )

		N = dr.width( samplingPosition )

		# Sample QuadTree irradiance
		rootIndexArray = dr.full( mi.UInt32, rootIndex, N )
		irradiance = self.sampleIrradiance( rootIndexArray, samplingPosition )
		irradiance_numpy = irradiance.numpy().reshape(numCell, numCell)

		# Draw QuadTree boxes visually
		leafNodeIndex = self.quadTreeNode.getAllLeafNodeIndex( mi.UInt32( rootIndex ) )
		leafBBoxes = self.quadTreeNode.getBBox( leafNodeIndex )
		min_x = leafBBoxes.min.x.numpy()
		min_y = leafBBoxes.min.y.numpy()
		max_x = leafBBoxes.max.x.numpy()
		max_y = leafBBoxes.max.y.numpy()

		# Plot heatmap
		if ax is None:
			fig, ax = plt.subplots()

		# Use Logarithmic bounds to better show high-dynamic range caustics versus diffuse areas
		vmin = max(1e-6, np.percentile(irradiance_numpy, 1))
		vmax = max(vmin + 1e-6, np.max(irradiance_numpy))
		
		im = ax.imshow( irradiance_numpy, cmap='turbo', interpolation='nearest', extent= [0, 1, 0, 1], origin= 'lower')
		ax.set_xlabel( 'Normalized Φ (Phi)' )
		ax.set_ylabel( 'Normalized Cos(θ) (Theta)' )
		ax.set_title( title, pad=10 )
		plt.colorbar( im, ax=ax, label='Irradiance (Log Scale)' )

		# samplingPos = mi.Vector2f( 0.25, 0.25 )
		# rootIndexArray = dr.full( mi.UInt32, rootIndex, 1 )
		# irradiance = self.sampleIrradiance( rootIndexArray, samplingPos )
		# print(irradiance)


class KDTreePlotter:

	def __init__(self, fileName: str) -> None:
		
		# Load from file
		treeDataNumpy = np.load(fileName)

		# Init the QuaddTreeNode and load data into
		self.kdTreeNode = KDTreeNode()
		self.kdTreeNode.loadFromFile( treeDataNumpy  )

		# Init QuadTreePlotter
		self.quadTreePlotter = QuadTreePlotter( fileName )


	def getSceneBBox( self ) -> Tuple[ Vec3, Vec3 ]:
		pass


	def findLeafNode( self, position: mi.Vector3f ) -> Tuple[ mi.UInt32, mi.Bool ]:
		"""
			Find the corresponding leaf node index of the given position.
			- position: query position.

			Return:
			- nodeIndex
			- isValid: 
		"""

		# Start traversing from root node
		nodeIndex = dr.zeros( mi.UInt32, dr.width( position ) )

		# 	Test if data is within the root node bbox
		rootNodeBBox = self.kdTreeNode.getBBox( 0 )
		isInsideRoot = rootNodeBBox.contains( position )
		active = mi.Bool( isInsideRoot )

		def loop_cond(active, nodeIndex):
			return active

		def loop_body(active, nodeIndex):
			# If at leaf node then stop. Otherwise, continue
			isLeafNode = dr.gather( mi.Bool, self.kdTreeNode.isLeaf, nodeIndex, active )
			
			# Else, check which children node it belongs to and store the node index
			isNotLeafNode = ~isLeafNode
			active &= isNotLeafNode 

			child_left_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_left_index, nodeIndex, active )
			child_right_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_right_index, nodeIndex, active )

			child_left_bbox = self.kdTreeNode.getBBox( child_left_idx )
			child_left_bbox_test = child_left_bbox.contains( position )
			nodeIndex[ child_left_bbox_test & active ] = child_left_idx

			child_right_bbox = self.kdTreeNode.getBBox( child_right_idx )
			child_right_bbox_test = child_right_bbox.contains( position )
			nodeIndex[ child_right_bbox_test & active ] = child_right_idx

			return (active, nodeIndex)

		active, nodeIndex = dr.while_loop(
			state=(active, nodeIndex),
			cond=loop_cond,
			body=loop_body
		)

		
		return nodeIndex, isInsideRoot

	def findHottestPositions(self, top_n: int = 5) -> List[Vec3]:
		"""
           Auto-discovers the most active spatial regions in the KD-Tree.
           Returns the 3D center coordinates of the top N leaf nodes with the highest vertex count.
        """
		# THE FIX: Just compress the isLeaf array directly from the struct
		leafNodeIndex_mi = dr.compress(self.kdTreeNode.isLeaf)
		leafNodeIndex_np = leafNodeIndex_mi.numpy()

		# Gather the energy/vertCount for those specific leaves
		vertCounts_mi = dr.gather(mi.Float, self.kdTreeNode.vertCount, leafNodeIndex_mi)
		vertCounts_np = vertCounts_mi.numpy()

		# Sort descending to find the highest traffic nodes
		top_indices_relative = np.argsort(vertCounts_np)[::-1][:top_n]
		top_leaf_indices = leafNodeIndex_np[top_indices_relative]

		# Fetch the bounding boxes for these hot nodes
		hot_bboxes = self.kdTreeNode.getBBox(top_leaf_indices)

		# Calculate the center points of these bounding boxes
		centers_x = (hot_bboxes.min.x.numpy() + hot_bboxes.max.x.numpy()) / 2.0
		centers_y = (hot_bboxes.min.y.numpy() + hot_bboxes.max.y.numpy()) / 2.0
		centers_z = (hot_bboxes.min.z.numpy() + hot_bboxes.max.z.numpy()) / 2.0

		hot_positions = []
		for i in range(len(centers_x)):
			pos = [float(centers_x[i]), float(centers_y[i]), float(centers_z[i])]
			print(f"Discovered Hotspot {i + 1}: {pos} (Traffic: {vertCounts_np[top_indices_relative[i]]})")
			hot_positions.append(pos)

		return hot_positions


	def plotQuadTreeAtPosition( self, position: Vec3, title: str, ax = None ) -> None:
		
		# Traverse the KDTree to find the corresponding rootnode
		position_mi = mi.Vector3f( position )
		leafNodeIndex_mi, isValid_mi = self.findLeafNode( position_mi )
		leafNodeIndex = leafNodeIndex_mi.numpy()[0]
		isValid = isValid_mi.numpy()[0]


		# If the query position is valid (withing tree bounding box) then plot the corresponding quadtree
		if isValid:
			# Get the corresponding root node index
			quadTreeRootIndex = dr.gather( mi.UInt32, self.kdTreeNode.quadTreeRootIndex, leafNodeIndex )

			# Plot the QuadTree
			self.quadTreePlotter.plotQuadTree( quadTreeRootIndex.numpy()[0], title, ax )


import glob
import os

class MultiIterationTreePlotter:

    def __init__(self, sceneName: str, numIteration: int) -> None:
       self.sceneName = sceneName
       self.numIteration = numIteration
       self.kdTreePlotters: List[KDTreePlotter] = []

       # Extract the camera transform from your standardized scene.xml
       self.to_world, self.fov, self.res_x, self.res_y = self._parse_scene_camera()

       for i in range(numIteration):
          fileName = FileNameManager.generateTreeDataFileName(i, withNpzEnding=True)
          kdTreePlotter = KDTreePlotter(fileName)
          self.kdTreePlotters.append(kdTreePlotter)

    def plotHottestRegions(self, top_n: int = 3, saveFig: bool = False) -> None:
       """
           Finds the most active regions from the final iteration's KD-Tree
           and generates timeline plots for each of them automatically.
        """
       printTitle(f"Auto-Discovering Top {top_n} Hottest Regions")

       # Use the fully mature KD-Tree from the final iteration to find the best spots
       final_kd_tree = self.kdTreePlotters[-1]
       hot_positions = final_kd_tree.findHottestPositions(top_n=top_n)

       for idx, pos in enumerate(hot_positions):
          print(f"\nGenerating plot for Hotspot {idx + 1} at {pos}...")

          # NO MORE MANGLING self.sceneName. Pass the suffix cleanly.
          self.plotQuadTreeAtPosition(pos, saveFig=saveFig, file_suffix=f"_hotspot_{idx + 1}")

    def plotQuadTreeAtPosition(self, position: Vec3, saveFig: bool = False, file_suffix: str = "") -> None:
       """
           Plots a dual-row timeline:
           Top Row: The actual rendered EXR for that iteration.
           Bottom Row: The QuadTree heatmap at the requested world position.
        """
       import math
       numIter = len(self.kdTreePlotters)

       # 2 rows: Top is Render PNG, Bottom is QuadTree Heatmap
       fig, axes = plt.subplots(2, numIter, figsize=(numIter * 4, 8))
       if numIter == 1:
          axes = np.array([[axes[0]], [axes[1]]])

       fig.suptitle(
          f'{self.sceneName} Evolution\nTarget Hotspot: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]',
          fontsize=16, fontweight='bold')

       for index, kdTreePlotter in enumerate(self.kdTreePlotters):
          ax_render = axes[0, index]
          ax_tree = axes[1, index]

          # ---------------------------------------------------------
          # 1. Plot the Rendered Image (from PNG)
          # ---------------------------------------------------------
          ax_render.set_title(f'Render (Iter {index})')
          ax_render.axis('off')

          # Snipe the PNGs you already generated.
          # file_format using wildcard for scene prefix in case it mismatches directory name
          file_format = f"*_iter-{index}_*.png"
          search_pattern = os.path.join(FileNameManager.IMAGE_FOLDER_PATH, file_format)

          matching_files = glob.glob(search_pattern)

          if matching_files:
             png_path = matching_files[0]

             # Zero math required, just load the damn picture
             img_array = plt.imread(png_path)
             ax_render.imshow(img_array)

             # PROJECT AND DRAW CROSSHAIRS
             px, py = self._project_world_to_pixel(position)

             if px is not None and py is not None:
                # Lime green crosshairs spanning the image
                ax_render.axvline(x=px, color='lime', linestyle='--', linewidth=1, alpha=0.6)
                ax_render.axhline(y=py, color='lime', linestyle='--', linewidth=1, alpha=0.6)

                # Red target ring scaled to 3% of your image width so it's always readable
                target_radius = self.res_x * 0.03
                target = patches.Circle((px, py), radius=target_radius, edgecolor='red', facecolor='none', linewidth=1.5)
                ax_render.add_patch(target)
             else:
                ax_render.text(0.5, 0.5, 'Hotspot Behind Camera', ha='center', va='center', color='red', transform=ax_render.transAxes)

          else:
             ax_render.text(0.5, 0.5, 'PNG Not Found', ha='center', va='center')

          # ---------------------------------------------------------
          # 2. Plot the QuadTree Heatmap
          # ---------------------------------------------------------
          tree_title = f'QuadTree (Iter {index})'
          kdTreePlotter.plotQuadTreeAtPosition(position, tree_title, ax=ax_tree)

       fig.tight_layout()

       if saveFig:
          # Inject the file suffix right after the scene name so your outputs are clean
          figFileName = os.path.join(FileNameManager.PLOT_FOLDER_PATH,
                               f'{self.sceneName}{file_suffix}_evolution_pos_{position[0]:.2f}_{position[1]:.2f}_{position[2]:.2f}.png')
          plt.savefig(fname=figFileName, dpi=300, bbox_inches='tight')
          print(f"Saved god-tier timeline to {figFileName}")

    def _parse_scene_camera(self):
       import xml.etree.ElementTree as ET
       import numpy as np

       # Path relative to your execution directory
       scene_xml_path = f'scenes/{self.sceneName}/scene.xml'
       tree = ET.parse(scene_xml_path)
       root = tree.getroot()

       # Parse defaults to resolve variables like $resx
       defaults = {}
       for default in root.findall('default'):
          defaults['$' + default.get('name')] = default.get('value')

       def resolve(val_str):
          return defaults.get(val_str, val_str)

       # Extract Film resolution
       sensor = root.find('.//sensor')
       film = sensor.find('film')
       res_x = float(resolve(film.find("integer[@name='width']").get('value')))
       res_y = float(resolve(film.find("integer[@name='height']").get('value')))

       # Extract FOV
       fov = float(resolve(sensor.find("float[@name='fov']").get('value')))

       # Extract 4x4 Transform Matrix
       matrix_str = sensor.find(".//transform/matrix").get('value')
       matrix_vals = list(map(float, matrix_str.split()))
       to_world = np.array(matrix_vals).reshape(4, 4)

       return to_world, fov, res_x, res_y

    def _project_world_to_pixel(self, p_world: list[float]) -> tuple[float, float]:
       import math
       import numpy as np

       # 1. World to Camera matrix (Inverse of to_world)
       w2c = np.linalg.inv(self.to_world)

       # 2. Transform point to camera space
       p_w = np.array([p_world[0], p_world[1], p_world[2], 1.0])
       p_cam = w2c @ p_w

       # If the Z value is behind the camera plane, skip projection
       # (Mitsuba camera conventions can be weird, flip this to > 0 if it culls visible stuff)
       if p_cam[2] <= 0:
          return None, None

       # Perspective divide
       x_cam = p_cam[0] / p_cam[2]
       y_cam = p_cam[1] / p_cam[2]

       # 3. Perspective Math
       fov_rad = math.radians(self.fov)
       tan_half_fov = math.tan(fov_rad / 2.0)
       aspect = self.res_x / self.res_y

       # Mitsuba camera: +X is left, +Y is up
       x_ndc = x_cam / tan_half_fov
       y_ndc = y_cam / (tan_half_fov / aspect)

       # 4. Map NDC [-1, 1] to Pixel Space (0,0 is top-left)
       pixel_x = (1.0 - x_ndc) * 0.5 * self.res_x
       pixel_y = (1.0 - y_ndc) * 0.5 * self.res_y

       return pixel_x, pixel_y

if __name__ == '__main__':
    sceneName = 'volumetric-caustic'
    # Make sure this matches the number of files generated by your main script exactly
    iteration = 6

    FileNameManager.setSceneName(sceneName)
    multiIterationTreePlotter = MultiIterationTreePlotter(sceneName, iteration)

    # Let the algorithm tell YOU where the interesting stuff happened.
    multiIterationTreePlotter.plotHottestRegions(top_n=10, saveFig=True)

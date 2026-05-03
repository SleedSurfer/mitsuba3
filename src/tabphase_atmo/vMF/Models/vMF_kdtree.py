from __future__ import annotations

import time

import drjit as dr
import mitsuba as mi

if __name__ == '__main__':
    mi.set_variant('llvm_spectral')

from vMF.common import *
import numpy as np
import math

# Import the standalone mixture class we just built
from vMF.Models.vMF_mixture import vMFMixture


class vMF_KDTreeNode:
    DRJIT_STRUCT = {
        'bbox': mi.BoundingBox3f,
        'depth': mi.UInt32,
        'vertCount': mi.Float,
        'isLeaf': mi.Bool,
        'child_left_index': mi.UInt32,
        'child_right_index': mi.UInt32,
        **{f'vmf_w{i}': mi.Float for i in range(8)},
        **{f'vmf_mu{i}': mi.Vector3f for i in range(8)},
        **{f'vmf_kappa{i}': mi.Float for i in range(8)},
        'fluence': mi.Float,
    }

    def __init__(self) -> None:
        self.bbox = mi.BoundingBox3f()
        self.depth = mi.UInt32()
        self.vertCount = mi.Float()
        self.isLeaf = mi.Bool()
        self.child_left_index = mi.UInt32()
        self.child_right_index = mi.UInt32()
        self.fluence = dr.zeros(mi.Float, 1)

        # Initialize all 8 lobes with uniform energy and slightly jittered directions
        # to prevent EM from collapsing on day one.
        for i in range(8):
            setattr(self, f'vmf_w{i}', dr.full(mi.Float, 1.0 / 8.0, 1))
            setattr(self, f'vmf_kappa{i}', dr.full(mi.Float, 1.0, 1))
            # Distribute mus across axes
            axis_val = [0.0, 0.0, 0.0]
            axis_val[i % 3] = 1.0
            setattr(self, f'vmf_mu{i}', mi.Vector3f(axis_val))

    def copyFrom(self, other: 'vMF_KDTreeNode') -> None:
        for key in self.DRJIT_STRUCT.keys():
            setattr(self, key, type(getattr(self, key))(getattr(other, key)))

    def resize(self, newSize: mi.UInt32) -> None:
        self.depth = resizeDrJitArray(self.depth, newSize)
        self.vertCount = resizeDrJitArray(self.vertCount, newSize)
        self.isLeaf = resizeDrJitArray(self.isLeaf, newSize, isDefaultZero=False)
        self.child_left_index = resizeDrJitArray(self.child_left_index, newSize)
        self.child_right_index = resizeDrJitArray(self.child_right_index, newSize)
        self.fluence = resizeDrJitArray(self.fluence, newSize)

        bbox_min = resizeDrJitArray(self.bbox.min, newSize)
        bbox_max = resizeDrJitArray(self.bbox.max, newSize)
        self.bbox = mi.BoundingBox3f(bbox_min, bbox_max)

        for i in range(8):
            setattr(self, f'vmf_w{i}', resizeDrJitArray(getattr(self, f'vmf_w{i}'), newSize))
            setattr(self, f'vmf_kappa{i}', resizeDrJitArray(getattr(self, f'vmf_kappa{i}'), newSize))
            mu = getattr(self, f'vmf_mu{i}')
            setattr(self, f'vmf_mu{i}', mi.Vector3f(
                resizeDrJitArray(mu.x, newSize),
                resizeDrJitArray(mu.y, newSize),
                resizeDrJitArray(mu.z, newSize)
            ))

    def getWidth(self) -> int:
        return dr.width(self.depth)

    def getBBox(self, idx: mi.UInt32) -> mi.BoundingBox3f:
        min_x = dr.gather(mi.Float, self.bbox.min.x, idx)
        min_y = dr.gather(mi.Float, self.bbox.min.y, idx)
        min_z = dr.gather(mi.Float, self.bbox.min.z, idx)
        bbox_min = mi.Vector3f(min_x, min_y, min_z)

        max_x = dr.gather(mi.Float, self.bbox.max.x, idx)
        max_y = dr.gather(mi.Float, self.bbox.max.y, idx)
        max_z = dr.gather(mi.Float, self.bbox.max.z, idx)
        bbox_max = mi.Vector3f(max_x, max_y, max_z)

        return mi.BoundingBox3f(bbox_min, bbox_max)

class vMF_KDTree:
    def __init__(self, max_leaf_size: float = 1, maxDepth: int = 20) -> None:
        # 1. ALLOCATE THE MEMORY FIRST
        self.kdTreeNode: vMF_KDTreeNode = dr.zeros(vMF_KDTreeNode, shape=1)

        # 2. NOW INITIALIZE THE ROOT NODE DATA
        self.kdTreeNode.isLeaf = mi.Bool(True)
        self.kdTreeNode.bbox = mi.BoundingBox3f([0, 0, 0], [1, 1, 1])
        self.kdTreeNode.fluence = dr.zeros(mi.Float, 1)

        # Unrolled initialization for 8 lobes
        for i in range(8):
            setattr(self.kdTreeNode, f'vmf_w{i}', dr.full(mi.Float, 1.0 / 8.0, 1))
            setattr(self.kdTreeNode, f'vmf_kappa{i}', dr.full(mi.Float, 1.0, 1))

            # Simple jitter for mus
            axis = [0.0, 0.0, 0.0]
            axis[i % 3] = 1.0
            setattr(self.kdTreeNode, f'vmf_mu{i}', mi.Vector3f(axis))

        # 3. SEAL THE GRAPH
        dr.eval(self.kdTreeNode)

        self.maxLeafSize = max_leaf_size
        self.maxDepth = maxDepth
        self.minLeafCountForSplit = 16.0
        self.maxSplitPerRefine = 4096

    def setup(self, bbox_min: mi.Vector3f, bbox_max: mi.Vector3f) -> None:
        self.kdTreeNode.bbox = mi.BoundingBox3f(bbox_min, bbox_max)

    def getAllLeafNodeIndex(self) -> mi.UInt32:
        return dr.compress(self.kdTreeNode.isLeaf)

    def copyFrom(self, other_tree: 'vMF_KDTree') -> None:
        """
        Clones the entire KD-Tree architecture and its underlying data.
        """
        self.kdTreeNode.copyFrom(other_tree.kdTreeNode)

        # Copy hyperparameters
        self.maxLeafSize = other_tree.maxLeafSize
        self.maxDepth = other_tree.maxDepth
        self.minLeafCountForSplit = other_tree.minLeafCountForSplit
        self.maxSplitPerRefine = other_tree.maxSplitPerRefine

    def split(self, idx: mi.UInt32) -> None:
        numSplitNode = dr.width(idx)
        numNewNode = numSplitNode * 2
        oldSize = self.kdTreeNode.getWidth()
        newsize = oldSize + numNewNode
        self.kdTreeNode.resize(newsize)
        dr.eval(self.kdTreeNode.vmf_mu0.x)

        childrenNodeIndex = dr.arange(mi.UInt32, numSplitNode)
        child_left_index = childrenNodeIndex * 2 + 0 + oldSize
        child_right_index = childrenNodeIndex * 2 + 1 + oldSize

        dr.scatter(self.kdTreeNode.child_left_index, child_left_index, idx)
        dr.scatter(self.kdTreeNode.child_right_index, child_right_index, idx)
        dr.scatter(self.kdTreeNode.isLeaf, False, idx)

        depth = dr.gather(mi.UInt32, self.kdTreeNode.depth, idx)
        childrenDepth = depth + 1
        dr.scatter(self.kdTreeNode.depth, childrenDepth, child_left_index)
        dr.scatter(self.kdTreeNode.depth, childrenDepth, child_right_index)

        vertCount = dr.gather(mi.Float, self.kdTreeNode.vertCount, idx)
        vertCount[vertCount > 0] = vertCount / 2
        dr.scatter(self.kdTreeNode.vertCount, vertCount, child_left_index)
        dr.scatter(self.kdTreeNode.vertCount, vertCount, child_right_index)

        # BBox Splitting Math
        min_x = dr.gather(mi.Float, self.kdTreeNode.bbox.min.x, idx)
        min_y = dr.gather(mi.Float, self.kdTreeNode.bbox.min.y, idx)
        min_z = dr.gather(mi.Float, self.kdTreeNode.bbox.min.z, idx)
        bbox_min = mi.Vector3f(min_x, min_y, min_z)

        max_x = dr.gather(mi.Float, self.kdTreeNode.bbox.max.x, idx)
        max_y = dr.gather(mi.Float, self.kdTreeNode.bbox.max.y, idx)
        max_z = dr.gather(mi.Float, self.kdTreeNode.bbox.max.z, idx)
        bbox_max = mi.Vector3f(max_x, max_y, max_z)

        bbox_mid = (bbox_min + bbox_max) / 2
        bbox_mid_plain: mi.Float = dr.ravel(bbox_mid)
        splitAxis = depth % 3
        bbox_mid_componentIndex = dr.arange(mi.UInt32, numSplitNode) * 3 + splitAxis
        bbox_mid_component = dr.gather(mi.Float, bbox_mid_plain, bbox_mid_componentIndex)

        bbox_left_min = mi.Vector3f(bbox_min)
        bbox_left_max = mi.Vector3f(bbox_max)
        bbox_left_max_plain = dr.ravel(bbox_left_max)
        dr.scatter(bbox_left_max_plain, bbox_mid_component, bbox_mid_componentIndex)
        bbox_left_max = dr.unravel(mi.Vector3f, bbox_left_max_plain)

        bbox_right_min = mi.Vector3f(bbox_min)
        bbox_right_max = mi.Vector3f(bbox_max)
        bbox_right_min_plain = dr.ravel(bbox_right_min)
        dr.scatter(bbox_right_min_plain, bbox_mid_component, bbox_mid_componentIndex)
        bbox_right_min = dr.unravel(mi.Vector3f, bbox_right_min_plain)

        dr.scatter(self.kdTreeNode.bbox.min.x, bbox_left_min.x, child_left_index)
        dr.scatter(self.kdTreeNode.bbox.min.y, bbox_left_min.y, child_left_index)
        dr.scatter(self.kdTreeNode.bbox.min.z, bbox_left_min.z, child_left_index)
        dr.scatter(self.kdTreeNode.bbox.max.x, bbox_left_max.x, child_left_index)
        dr.scatter(self.kdTreeNode.bbox.max.y, bbox_left_max.y, child_left_index)
        dr.scatter(self.kdTreeNode.bbox.max.z, bbox_left_max.z, child_left_index)

        dr.scatter(self.kdTreeNode.bbox.min.x, bbox_right_min.x, child_right_index)
        dr.scatter(self.kdTreeNode.bbox.min.y, bbox_right_min.y, child_right_index)
        dr.scatter(self.kdTreeNode.bbox.min.z, bbox_right_min.z, child_right_index)
        dr.scatter(self.kdTreeNode.bbox.max.x, bbox_right_max.x, child_right_index)
        dr.scatter(self.kdTreeNode.bbox.max.y, bbox_right_max.y, child_right_index)
        dr.scatter(self.kdTreeNode.bbox.max.z, bbox_right_max.z, child_right_index)

        # Copy parent's vMF mixture to children as a starting guess
        for child_idx in [child_left_index, child_right_index]:
            # Gather the parent's vMF state ONCE to avoid loop evaluation drift
            p_w0 = dr.gather(mi.Float, self.kdTreeNode.vmf_w0, idx)
            p_w1 = dr.gather(mi.Float, self.kdTreeNode.vmf_w1, idx)
            p_w2 = dr.gather(mi.Float, self.kdTreeNode.vmf_w2, idx)

            p_mu0_x, p_mu0_y, p_mu0_z = dr.gather(mi.Float, self.kdTreeNode.vmf_mu0.x, idx), dr.gather(mi.Float,
                                                                                                       self.kdTreeNode.vmf_mu0.y,
                                                                                                       idx), dr.gather(
                mi.Float, self.kdTreeNode.vmf_mu0.z, idx)
            p_mu1_x, p_mu1_y, p_mu1_z = dr.gather(mi.Float, self.kdTreeNode.vmf_mu1.x, idx), dr.gather(mi.Float,
                                                                                                       self.kdTreeNode.vmf_mu1.y,
                                                                                                       idx), dr.gather(
                mi.Float, self.kdTreeNode.vmf_mu1.z, idx)
            p_mu2_x, p_mu2_y, p_mu2_z = dr.gather(mi.Float, self.kdTreeNode.vmf_mu2.x, idx), dr.gather(mi.Float,
                                                                                                       self.kdTreeNode.vmf_mu2.y,
                                                                                                       idx), dr.gather(
                mi.Float, self.kdTreeNode.vmf_mu2.z, idx)

            p_k0 = dr.gather(mi.Float, self.kdTreeNode.vmf_kappa0, idx)
            p_k1 = dr.gather(mi.Float, self.kdTreeNode.vmf_kappa1, idx)
            p_k2 = dr.gather(mi.Float, self.kdTreeNode.vmf_kappa2, idx)

            # Inherit to children
            for child_idx in [child_left_index, child_right_index]:
                dr.scatter(self.kdTreeNode.vmf_w0, p_w0, child_idx)
                dr.scatter(self.kdTreeNode.vmf_w1, p_w1, child_idx)
                dr.scatter(self.kdTreeNode.vmf_w2, p_w2, child_idx)

                dr.scatter(self.kdTreeNode.vmf_kappa0, p_k0, child_idx)
                dr.scatter(self.kdTreeNode.vmf_kappa1, p_k1, child_idx)
                dr.scatter(self.kdTreeNode.vmf_kappa2, p_k2, child_idx)

                dr.scatter(self.kdTreeNode.vmf_mu0.x, p_mu0_x, child_idx)
                dr.scatter(self.kdTreeNode.vmf_mu0.y, p_mu0_y, child_idx)
                dr.scatter(self.kdTreeNode.vmf_mu0.z, p_mu0_z, child_idx)

                dr.scatter(self.kdTreeNode.vmf_mu1.x, p_mu1_x, child_idx)
                dr.scatter(self.kdTreeNode.vmf_mu1.y, p_mu1_y, child_idx)
                dr.scatter(self.kdTreeNode.vmf_mu1.z, p_mu1_z, child_idx)

                dr.scatter(self.kdTreeNode.vmf_mu2.x, p_mu2_x, child_idx)
                dr.scatter(self.kdTreeNode.vmf_mu2.y, p_mu2_y, child_idx)
                dr.scatter(self.kdTreeNode.vmf_mu2.z, p_mu2_z, child_idx)

            # Seal the graph
            dr.eval(self.kdTreeNode.vertCount, self.kdTreeNode.bbox.min, self.kdTreeNode.bbox.max)
            dr.eval(self.kdTreeNode.vmf_mu0, self.kdTreeNode.vmf_mu1, self.kdTreeNode.vmf_mu2)

    def fit_mixtures_to_record(self, leaf_indices: mi.UInt32,
                               directions: mi.Vector3f, weights: mi.Float,
                               directions_nee: mi.Vector3f, weights_nee: mi.Float,
                               active: mi.Bool, current_iteration: int, iterations: int = 12) -> None:
        num_leaves = self.kdTreeNode.getWidth()
        K = 8

        # 1. GATHER INITIAL STATE
        orig_ws = [getattr(self.kdTreeNode, f'vmf_w{i}') for i in range(K)]
        orig_mus = [getattr(self.kdTreeNode, f'vmf_mu{i}') for i in range(K)]
        orig_ks = [getattr(self.kdTreeNode, f'vmf_kappa{i}') for i in range(K)]

        ws, mus, ks = list(orig_ws), list(orig_mus), list(orig_ks)

        valid_ray = active & (weights > 1e-6)
        valid_nee = active & (weights_nee > 1e-6)

        # Calculate raw radiometric weight per leaf
        leaf_total_weight = dr.zeros(mi.Float, num_leaves)
        dr.scatter_reduce(dr.ReduceOp.Add, leaf_total_weight, weights, leaf_indices, valid_ray)
        dr.scatter_reduce(dr.ReduceOp.Add, leaf_total_weight, weights_nee, leaf_indices, valid_nee)

        # FIX 1: Calculate actual photon count (N) per leaf
        leaf_N = dr.zeros(mi.Float, num_leaves)
        dr.scatter_reduce(dr.ReduceOp.Add, leaf_N, mi.Float(1.0), leaf_indices, valid_ray)
        dr.scatter_reduce(dr.ReduceOp.Add, leaf_N, mi.Float(1.0), leaf_indices, valid_nee)
        dr.eval(leaf_total_weight, leaf_N)

        leaf_has_data = leaf_total_weight > 1e-6

        # FIX 1 (Cont.): Normalize weights to sum to N_leaf for the MAP-EM prior compatibility
        scale_factor = dr.select(leaf_has_data, leaf_N / leaf_total_weight, 0.0)

        # Gather the per-leaf scale factor back to the per-photon lanes
        gathered_scale = dr.gather(mi.Float, scale_factor, leaf_indices, valid_ray)
        gathered_scale_nee = dr.gather(mi.Float, scale_factor, leaf_indices, valid_nee)

        norm_weights = weights * gathered_scale
        norm_weights_nee = weights_nee * gathered_scale_nee

        def eval_vmf(mu, kappa, d):
            cos_t = dr.dot(mu, d)
            safe_kappa = dr.maximum(kappa, 1e-4)
            norm = safe_kappa / (dr.two_pi * (1.0 - dr.exp(-2.0 * safe_kappa)))
            return dr.select(kappa < 1e-4, dr.inv_four_pi, norm * dr.exp(safe_kappa * (cos_t - 1.0)))

        for _ in range(iterations):
            # --- E-STEP ---
            g_ws = [dr.gather(mi.Float, ws[i], leaf_indices, active) for i in range(K)]
            g_mus = [dr.gather(mi.Vector3f, mus[i], leaf_indices, active) for i in range(K)]
            g_ks = [dr.gather(mi.Float, ks[i], leaf_indices, active) for i in range(K)]

            pdfs_ray = [g_ws[i] * eval_vmf(g_mus[i], g_ks[i], directions) for i in range(K)]
            sum_pdf_ray = sum(pdfs_ray)
            resps_ray = [dr.select(sum_pdf_ray > 1e-8, p / sum_pdf_ray, 1.0 / K) for p in pdfs_ray]

            pdfs_nee = [g_ws[i] * eval_vmf(g_mus[i], g_ks[i], directions_nee) for i in range(K)]
            sum_pdf_nee = sum(pdfs_nee)
            resps_nee = [dr.select(sum_pdf_nee > 1e-8, p / sum_pdf_nee, 1.0 / K) for p in pdfs_nee]

            # --- M-STEP ---
            for i in range(K):
                # Use the statistically normalized weights here
                contrib_ray = resps_ray[i] * norm_weights
                contrib_nee = resps_nee[i] * norm_weights_nee

                sum_w = dr.zeros(mi.Float, num_leaves)
                R = dr.zeros(mi.Vector3f, num_leaves)

                dr.scatter_reduce(dr.ReduceOp.Add, sum_w, contrib_ray, leaf_indices, valid_ray)
                dr.scatter_reduce(dr.ReduceOp.Add, sum_w, contrib_nee, leaf_indices, valid_nee)

                for c in ['x', 'y', 'z']:
                    dr.scatter_reduce(dr.ReduceOp.Add, getattr(R, c), getattr(directions, c) * contrib_ray,
                                      leaf_indices, valid_ray)
                    dr.scatter_reduce(dr.ReduceOp.Add, getattr(R, c), getattr(directions_nee, c) * contrib_nee,
                                      leaf_indices, valid_nee)

                alpha_prior, beta_prior = 10.0, 0.8
                r_bar_λ = (alpha_prior * beta_prior + dr.norm(R)) / (alpha_prior + sum_w)

                r_safe = dr.minimum(r_bar_λ, 0.9999)
                new_k = (r_safe * (3.0 - dr.square(r_safe))) / (1.0 - dr.square(r_safe))
                new_k = dr.minimum(new_k, 30.0)
                new_mu = dr.normalize(dr.select(dr.norm(R) > 1e-6, R, mus[i]))

                # Weight update: proportion of energy in this lobe relative to leaf total
                new_w = sum_w / dr.select(leaf_has_data, leaf_N, 1.0)

                ws[i] = dr.select(leaf_has_data, new_w, ws[i])
                mus[i] = dr.select(leaf_has_data, new_mu, mus[i])
                ks[i] = dr.select(leaf_has_data, new_k, ks[i])

            # FIX 2: Evaluate the computational graph at the end of every EM iteration
            # Prevents LLVM from choking on a massive unrolled loop
            dr.eval(*ws, *mus, *ks)

        # 3. LERP INTO STORAGE
        lr = mi.Float(1.0 / (float(current_iteration) + 1.0))
        total_w = sum(ws)
        for i in range(K):
            final_w = dr.lerp(orig_ws[i], ws[i] / dr.select(total_w > 0, total_w, 1.0), lr)
            final_mu = dr.normalize(dr.lerp(orig_mus[i], mus[i], lr))
            final_k = dr.minimum(dr.lerp(orig_ks[i], ks[i], lr), 30.0)

            setattr(self.kdTreeNode, f'vmf_w{i}', final_w)
            setattr(self.kdTreeNode, f'vmf_mu{i}', final_mu)
            setattr(self.kdTreeNode, f'vmf_kappa{i}', final_k)

        dr.eval(self.kdTreeNode)

    def setRefinementThreshold(self, iteration: int) -> None:
        c = 1500
        self.maxLeafSize = c * math.sqrt(math.pow(2, iteration))

    def refine(self) -> None:
        leafNodeIndex = self.getAllLeafNodeIndex()
        vertCount = dr.gather(mi.Float, self.kdTreeNode.vertCount, leafNodeIndex)
        depth = dr.gather(mi.UInt32, self.kdTreeNode.depth, leafNodeIndex)
        effectiveLeafSize = dr.maximum(mi.Float(self.maxLeafSize), mi.Float(self.minLeafCountForSplit))
        condition = (vertCount > effectiveLeafSize) & (depth < self.maxDepth)

        if dr.any(condition):
            splitNodeIndex = dr.gather(mi.UInt32, leafNodeIndex, dr.compress(condition))
            nSplit = dr.width(splitNodeIndex)
            if nSplit > self.maxSplitPerRefine:
                splitNodeIndex = dr.gather(mi.UInt32, splitNodeIndex, dr.arange(mi.UInt32, self.maxSplitPerRefine))
            self.split(splitNodeIndex)

    def getLeafNodeIndex(self, position: mi.Vector3f, active: mi.Bool = True) -> mi.UInt32:
        size = dr.width(position)
        nodeIndex = dr.zeros(mi.UInt32, size)
        rootNodeBBox = self.kdTreeNode.getBBox(0)
        search_active = rootNodeBBox.contains(position) & active

        loop_state = (search_active, nodeIndex)

        def loop_cond(search_active, nodeIndex):
            return dr.detach(search_active)

        def loop_body(search_active, nodeIndex):
            isLeafNode = dr.gather(mi.Bool, self.kdTreeNode.isLeaf, nodeIndex, search_active)
            isNotLeafNode = ~isLeafNode
            search_active &= isNotLeafNode

            child_left_idx = dr.gather(mi.UInt32, self.kdTreeNode.child_left_index, nodeIndex, search_active)
            child_right_idx = dr.gather(mi.UInt32, self.kdTreeNode.child_right_index, nodeIndex, search_active)

            child_left_bbox = self.kdTreeNode.getBBox(child_left_idx)
            child_left_bbox_test = child_left_bbox.contains(position)
            nodeIndex[child_left_bbox_test & search_active] = child_left_idx

            child_right_bbox = self.kdTreeNode.getBBox(child_right_idx)
            child_right_bbox_test = child_right_bbox.contains(position)
            nodeIndex[child_right_bbox_test & search_active] = child_right_idx

            return (search_active, nodeIndex)

        search_active, nodeIndex = dr.while_loop(state=loop_state, cond=loop_cond, body=loop_body)
        return nodeIndex

    # =========================================================================
    # THE vMF BRIDGE
    # =========================================================================

    def query_mixture(self, position: mi.Vector3f, active: mi.Bool = True) -> vMFMixture:
        leaf_idx = self.getLeafNodeIndex(position, active)

        weights, mus, kappas = [], [], []
        for i in range(8):
            weights.append(dr.gather(mi.Float, getattr(self.kdTreeNode, f'vmf_w{i}'), leaf_idx, active))
            mus.append(dr.gather(mi.Vector3f, getattr(self.kdTreeNode, f'vmf_mu{i}'), leaf_idx, active))
            kappas.append(dr.gather(mi.Float, getattr(self.kdTreeNode, f'vmf_kappa{i}'), leaf_idx, active))

        return vMFMixture(weights=weights, mus=mus, kappas=kappas)

    def resetTreeVertCount(self) -> None:
        """
        Resets the recorded photon counters for the next data collection pass.
        No tree traversal needed—just instantly zero out the flat SIMD array.
        """
        num_nodes = self.kdTreeNode.getWidth()
        self.kdTreeNode.vertCount = dr.zeros(mi.Float, num_nodes)

        # Lock it in so the compiler doesn't ghost us again
        dr.eval(self.kdTreeNode.vertCount)

    def saveToFile(self, fileName: str) -> None:
        """
        Saves the KD-Tree topology and the unrolled vMF SoA arrays into a compressed numpy file.
        Dynamically captures all 8 lobes so we stop giving the integrator amnesia.
        """
        save_dict = {
            # Hyperparameters
            'kdtree_maxLeafSize': self.maxLeafSize,
            'kdtree_maxDepth': self.maxDepth,

            # KD-Tree Spatial Topology
            'kdtree_bbox_min_x': self.kdTreeNode.bbox.min.x.numpy(),
            'kdtree_bbox_min_y': self.kdTreeNode.bbox.min.y.numpy(),
            'kdtree_bbox_min_z': self.kdTreeNode.bbox.min.z.numpy(),
            'kdtree_bbox_max_x': self.kdTreeNode.bbox.max.x.numpy(),
            'kdtree_bbox_max_y': self.kdTreeNode.bbox.max.y.numpy(),
            'kdtree_bbox_max_z': self.kdTreeNode.bbox.max.z.numpy(),

            'kdtree_depth': self.kdTreeNode.depth.numpy(),
            'kdtree_vertCount': self.kdTreeNode.vertCount.numpy(),
            'kdtree_isLeaf': self.kdTreeNode.isLeaf.numpy(),
            'kdtree_child_left_index': self.kdTreeNode.child_left_index.numpy(),
            'kdtree_child_right_index': self.kdTreeNode.child_right_index.numpy(),
        }

        # The Brain (Dynamically save all K lobes)
        for i in range(8):
            save_dict[f'vmf_w{i}'] = getattr(self.kdTreeNode, f'vmf_w{i}').numpy()
            save_dict[f'vmf_kappa{i}'] = getattr(self.kdTreeNode, f'vmf_kappa{i}').numpy()

            mu = getattr(self.kdTreeNode, f'vmf_mu{i}')
            save_dict[f'vmf_mu{i}_x'] = mu.x.numpy()
            save_dict[f'vmf_mu{i}_y'] = mu.y.numpy()
            save_dict[f'vmf_mu{i}_z'] = mu.z.numpy()

        np.savez_compressed(file=fileName, **save_dict)

    def loadFromFile(self, dataNumpy: np.array) -> None:
        """
        Reconstructs the KD-Tree and its vMF memory banks from a numpy file.
        """
        self.maxLeafSize = int(dataNumpy['kdtree_maxLeafSize'])
        self.maxDepth = int(dataNumpy['kdtree_maxDepth'])

        # Reset the struct to avoid ghost data
        self.kdTreeNode = vMF_KDTreeNode()

        # Load Spatial Topology
        self.kdTreeNode.depth = mi.UInt32(dataNumpy['kdtree_depth'])
        self.kdTreeNode.vertCount = mi.Float(dataNumpy['kdtree_vertCount'])
        self.kdTreeNode.isLeaf = mi.Bool(dataNumpy['kdtree_isLeaf'])
        self.kdTreeNode.child_left_index = mi.UInt32(dataNumpy['kdtree_child_left_index'])
        self.kdTreeNode.child_right_index = mi.UInt32(dataNumpy['kdtree_child_right_index'])

        self.kdTreeNode.bbox = mi.BoundingBox3f(
            mi.Vector3f(dataNumpy['kdtree_bbox_min_x'], dataNumpy['kdtree_bbox_min_y'], dataNumpy['kdtree_bbox_min_z']),
            mi.Vector3f(dataNumpy['kdtree_bbox_max_x'], dataNumpy['kdtree_bbox_max_y'], dataNumpy['kdtree_bbox_max_z'])
        )

        # Load The Brain (Dynamically reconstruct all K lobes)
        for i in range(8):
            setattr(self.kdTreeNode, f'vmf_w{i}', mi.Float(dataNumpy[f'vmf_w{i}']))
            setattr(self.kdTreeNode, f'vmf_kappa{i}', mi.Float(dataNumpy[f'vmf_kappa{i}']))
            setattr(self.kdTreeNode, f'vmf_mu{i}', mi.Vector3f(
                dataNumpy[f'vmf_mu{i}_x'],
                dataNumpy[f'vmf_mu{i}_y'],
                dataNumpy[f'vmf_mu{i}_z']
            ))

        # Seal the memory graph so DrJit doesn't unalive itself
        dr.eval(self.kdTreeNode)

if __name__ == '__main__':
    from numpy.random import rand as np_rand
    import time

    def run_smoke_tests():
        printTitle("--- SMOKE TEST: vMF KD-Tree Architecture ---")

        # 1. INITIALIZATION TEST
        printBoldUnderLine("Test 1: Tree Setup")
        tree = vMF_KDTree()
        tree.setup(bbox_min=[0, 0, 0], bbox_max=[100, 100, 100])
        initial_size = tree.kdTreeNode.getWidth()
        print(f"Initial Tree Size (Should be 1): {initial_size}")

        # 2. SPLIT & INHERITANCE TEST
        printBoldUnderLine("\nTest 2: Array Resizing & vMF Inheritance")
        # Force a split on the root node
        leaf_nodes = tree.getAllLeafNodeIndex()
        tree.split(leaf_nodes)

        new_size = tree.kdTreeNode.getWidth()
        print(f"Tree Size after 1 split (Should be 3): {new_size}")

        # Verify the children inherited the default vMF parameters from the parent
        child_left = dr.gather(mi.UInt32, tree.kdTreeNode.child_left_index, 0)
        inherited_mu0_x = dr.gather(mi.Float, tree.kdTreeNode.vmf_mu0.x, child_left)
        print(f"Inherited Lobe 0 Mu.X on left child (Should be 1.0): {inherited_mu0_x[0]}")

        # 3. SPATIAL QUERY BRIDGE TEST
        printBoldUnderLine("\nTest 3: Vectorized Mixture Query")
        N_rays = 500000
        print(f"Shooting {N_rays} random rays to test SIMD gather throughput...")

        # Generate the numpy data
        np_pos = np_rand(N_rays, 3) * 100.0

        # Feed the components explicitly so Dr.Jit can pack them into SIMD lanes
        random_positions = mi.Vector3f(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
        active_mask = mi.Bool(True)

        start_time = time.perf_counter()

        # The ultimate test: Does the bridge reconstruct the mixture across SIMD lanes?
        queried_mixture = tree.query_mixture(random_positions, active_mask)

        # Evaluate a dummy PDF to force LLVM to actually execute the gather graph
        test_dir = mi.Vector3f(0, 0, 1)
        test_pdfs = queried_mixture.pdf(test_dir)
        dr.eval(test_pdfs)

        elapsed = time.perf_counter() - start_time

        print(f"Query & PDF Eval complete in {elapsed:.4f} seconds.")
        print(f"Max PDF value evaluated: {dr.max(test_pdfs)[0]:.4f}")
        print("\nRESULT: If you are reading this and the kernel didn't segfault, the architecture is working.")

    run_smoke_tests()
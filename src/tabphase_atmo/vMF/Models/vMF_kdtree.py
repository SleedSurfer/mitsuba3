from __future__ import annotations

import time

import drjit as dr
import mitsuba as mi

if __name__ == '__main__':
    mi.set_variant('llvm_spectral')

from practical_path_guiding.src.common import *
import numpy as np
import math

# Import the standalone mixture class we just built
from vMF.Models.vMF_mixture import vMFMixture


class vMF_KDTreeNode:
    # Explicitly unrolled K=3 vMF mixture arrays for perfect SIMD memory alignment
    DRJIT_STRUCT = {
        'bbox': mi.BoundingBox3f,
        'depth': mi.UInt32,
        'vertCount': mi.Float,
        'isLeaf': mi.Bool,
        'child_left_index': mi.UInt32,
        'child_right_index': mi.UInt32,

        # vMF Lobe 1
        'vmf_w0': mi.Float, 'vmf_mu0': mi.Vector3f, 'vmf_kappa0': mi.Float,
        # vMF Lobe 2
        'vmf_w1': mi.Float, 'vmf_mu1': mi.Vector3f, 'vmf_kappa1': mi.Float,
        # vMF Lobe 3
        'vmf_w2': mi.Float, 'vmf_mu2': mi.Vector3f, 'vmf_kappa2': mi.Float,
    }

    def __init__(self) -> None:
        self.bbox = mi.BoundingBox3f()
        self.depth = mi.UInt32()
        self.vertCount = mi.Float()
        self.isLeaf = mi.Bool()
        self.child_left_index = mi.UInt32()
        self.child_right_index = mi.UInt32()

        # THE FIX: Explicit dr.full allocation binds this to physical LLVM memory
        # so resizeDrJitArray can actually see and copy the values.
        self.vmf_w0 = dr.full(mi.Float, 1.0 / 3.0, 1)
        self.vmf_w1 = dr.full(mi.Float, 1.0 / 3.0, 1)
        self.vmf_w2 = dr.full(mi.Float, 1.0 / 3.0, 1)

        self.vmf_mu0 = mi.Vector3f(dr.full(mi.Float, 1.0, 1), dr.zeros(mi.Float, 1), dr.zeros(mi.Float, 1))
        self.vmf_mu1 = mi.Vector3f(dr.zeros(mi.Float, 1), dr.full(mi.Float, 1.0, 1), dr.zeros(mi.Float, 1))
        self.vmf_mu2 = mi.Vector3f(dr.zeros(mi.Float, 1), dr.zeros(mi.Float, 1), dr.full(mi.Float, 1.0, 1))

        self.vmf_kappa0 = dr.full(mi.Float, 5.0, 1)
        self.vmf_kappa1 = dr.full(mi.Float, 5.0, 1)
        self.vmf_kappa2 = dr.full(mi.Float, 5.0, 1)

        dr.eval(self.vmf_mu0.x)
        print(f"\n[DIAG 1 | Init] vmf_mu0.x width: {dr.width(self.vmf_mu0.x)} | data: {self.vmf_mu0.x.numpy()}")

    def copyFrom(self, other_node: 'vMF_KDTreeNode') -> None:
        """
        Deep copies the vectorized memory arrays from another node into this one.
        """
        self.bbox = mi.BoundingBox3f(other_node.bbox.min, other_node.bbox.max)
        self.depth = mi.UInt32(other_node.depth)
        self.vertCount = mi.Float(other_node.vertCount)
        self.isLeaf = mi.Bool(other_node.isLeaf)
        self.child_left_index = mi.UInt32(other_node.child_left_index)
        self.child_right_index = mi.UInt32(other_node.child_right_index)

        # Copy vMF Mixtures
        self.vmf_w0 = mi.Float(other_node.vmf_w0)
        self.vmf_w1 = mi.Float(other_node.vmf_w1)
        self.vmf_w2 = mi.Float(other_node.vmf_w2)

        # Vector3f copies component-wise safely
        self.vmf_mu0 = mi.Vector3f(other_node.vmf_mu0)
        self.vmf_mu1 = mi.Vector3f(other_node.vmf_mu1)
        self.vmf_mu2 = mi.Vector3f(other_node.vmf_mu2)

        self.vmf_kappa0 = mi.Float(other_node.vmf_kappa0)
        self.vmf_kappa1 = mi.Float(other_node.vmf_kappa1)
        self.vmf_kappa2 = mi.Float(other_node.vmf_kappa2)

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

    def resize(self, newSize: mi.UInt32) -> None:
        """Resizes the KDTree arrays to accommodate splits."""
        self.depth = resizeDrJitArray(self.depth, newSize)
        self.vertCount = resizeDrJitArray(self.vertCount, newSize)
        self.isLeaf = resizeDrJitArray(self.isLeaf, newSize, isDefaultZero=False)
        self.child_left_index = resizeDrJitArray(self.child_left_index, newSize)
        self.child_right_index = resizeDrJitArray(self.child_right_index, newSize)

        bbox_min = resizeDrJitArray(self.bbox.min, newSize)
        bbox_max = resizeDrJitArray(self.bbox.max, newSize)
        self.bbox = mi.BoundingBox3f(bbox_min, bbox_max)

        # Flat float arrays are fine
        self.vmf_w0 = resizeDrJitArray(self.vmf_w0, newSize)
        self.vmf_w1 = resizeDrJitArray(self.vmf_w1, newSize)
        self.vmf_w2 = resizeDrJitArray(self.vmf_w2, newSize)

        self.vmf_kappa0 = resizeDrJitArray(self.vmf_kappa0, newSize)
        self.vmf_kappa1 = resizeDrJitArray(self.vmf_kappa1, newSize)
        self.vmf_kappa2 = resizeDrJitArray(self.vmf_kappa2, newSize)

        # THE FIX: Explicitly unroll the Vector3f components to survive reallocation
        self.vmf_mu0 = mi.Vector3f(
            resizeDrJitArray(self.vmf_mu0.x, newSize),
            resizeDrJitArray(self.vmf_mu0.y, newSize),
            resizeDrJitArray(self.vmf_mu0.z, newSize)
        )
        self.vmf_mu1 = mi.Vector3f(
            resizeDrJitArray(self.vmf_mu1.x, newSize),
            resizeDrJitArray(self.vmf_mu1.y, newSize),
            resizeDrJitArray(self.vmf_mu1.z, newSize)
        )
        self.vmf_mu2 = mi.Vector3f(
            resizeDrJitArray(self.vmf_mu2.x, newSize),
            resizeDrJitArray(self.vmf_mu2.y, newSize),
            resizeDrJitArray(self.vmf_mu2.z, newSize)
        )

        dr.eval(self.depth, self.vertCount, self.isLeaf, self.child_left_index, self.child_right_index, self.bbox.min,
                self.bbox.max)
        dr.eval(self.vmf_w0, self.vmf_w1, self.vmf_w2, self.vmf_mu0, self.vmf_mu1, self.vmf_mu2, self.vmf_kappa0,
                self.vmf_kappa1, self.vmf_kappa2)


class vMF_KDTree:
    def __init__(self, max_leaf_size: float = 1, maxDepth: int = 10) -> None:
        # Dr.Jit allocates the memory and fills everything with 0.0
        self.kdTreeNode: vMF_KDTreeNode = dr.zeros(vMF_KDTreeNode, shape=1)

        self.kdTreeNode.isLeaf = mi.Bool(True)
        self.kdTreeNode.bbox = mi.BoundingBox3f([0, 0, 0], [1, 1, 1])

        # --- THE FIX: RESURRECT THE DATA ---
        # We explicitly inject the starting vMF values into the zeroed-out root node.
        self.kdTreeNode.vmf_w0 = dr.full(mi.Float, 1.0 / 3.0, 1)
        self.kdTreeNode.vmf_w1 = dr.full(mi.Float, 1.0 / 3.0, 1)
        self.kdTreeNode.vmf_w2 = dr.full(mi.Float, 1.0 / 3.0, 1)

        self.kdTreeNode.vmf_mu0 = mi.Vector3f(dr.full(mi.Float, 1.0, 1), dr.zeros(mi.Float, 1),
                                              dr.zeros(mi.Float, 1))
        self.kdTreeNode.vmf_mu1 = mi.Vector3f(dr.zeros(mi.Float, 1), dr.full(mi.Float, 1.0, 1),
                                              dr.zeros(mi.Float, 1))
        self.kdTreeNode.vmf_mu2 = mi.Vector3f(dr.zeros(mi.Float, 1), dr.zeros(mi.Float, 1),
                                              dr.full(mi.Float, 1.0, 1))

        self.kdTreeNode.vmf_kappa0 = dr.full(mi.Float, 5.0, 1)
        self.kdTreeNode.vmf_kappa1 = dr.full(mi.Float, 5.0, 1)
        self.kdTreeNode.vmf_kappa2 = dr.full(mi.Float, 5.0, 1)

        dr.eval(self.kdTreeNode.vmf_w0, self.kdTreeNode.vmf_mu0, self.kdTreeNode.vmf_kappa0)
        # -----------------------------------

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
        print(f"[DIAG 2 | Post-Resize] vmf_mu0.x width: {dr.width(self.kdTreeNode.vmf_mu0.x)} | data: {self.kdTreeNode.vmf_mu0.x.numpy()}")

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
            print(
                f"[DIAG 3 | Post-Inherit] vmf_mu0.x width: {dr.width(self.kdTreeNode.vmf_mu0.x)} | data: {self.kdTreeNode.vmf_mu0.x.numpy()}\n")

    def fit_mixtures_to_record(self, leaf_indices: mi.UInt32,
                               directions: mi.Vector3f, weights: mi.Float,
                               directions_nee: mi.Vector3f, weights_nee: mi.Float,
                               active: mi.Bool, current_iteration: int, iterations: int = 4) -> None:
        num_leaves = self.kdTreeNode.getWidth()

        w0, w1, w2 = self.kdTreeNode.vmf_w0, self.kdTreeNode.vmf_w1, self.kdTreeNode.vmf_w2
        mu0, mu1, mu2 = self.kdTreeNode.vmf_mu0, self.kdTreeNode.vmf_mu1, self.kdTreeNode.vmf_mu2
        k0, k1, k2 = self.kdTreeNode.vmf_kappa0, self.kdTreeNode.vmf_kappa1, self.kdTreeNode.vmf_kappa2

        orig_w0, orig_w1, orig_w2 = w0, w1, w2
        orig_mu0, orig_mu1, orig_mu2 = mu0, mu1, mu2
        orig_k0, orig_k1, orig_k2 = k0, k1, k2

        valid_ray = active & (weights > 1e-6)
        valid_nee = active & (weights_nee > 1e-6)

        leaf_total_weight = dr.zeros(mi.Float, num_leaves)
        dr.scatter_reduce(dr.ReduceOp.Add, leaf_total_weight, weights, leaf_indices, valid_ray)
        dr.scatter_reduce(dr.ReduceOp.Add, leaf_total_weight, weights_nee, leaf_indices, valid_nee)

        leaf_has_data = leaf_total_weight > 1e-6

        for _ in range(iterations):
            r_w0 = dr.gather(mi.Float, w0, leaf_indices, valid_ray)
            r_w1 = dr.gather(mi.Float, w1, leaf_indices, valid_ray)
            r_w2 = dr.gather(mi.Float, w2, leaf_indices, valid_ray)
            r_mu0 = dr.gather(mi.Vector3f, mu0, leaf_indices, valid_ray)
            r_mu1 = dr.gather(mi.Vector3f, mu1, leaf_indices, valid_ray)
            r_mu2 = dr.gather(mi.Vector3f, mu2, leaf_indices, valid_ray)
            r_k0 = dr.gather(mi.Float, k0, leaf_indices, valid_ray)
            r_k1 = dr.gather(mi.Float, k1, leaf_indices, valid_ray)
            r_k2 = dr.gather(mi.Float, k2, leaf_indices, valid_ray)

            # Re-gather for NEE to prevent width mismatches
            rn_w0 = dr.gather(mi.Float, w0, leaf_indices, valid_nee)
            rn_w1 = dr.gather(mi.Float, w1, leaf_indices, valid_nee)
            rn_w2 = dr.gather(mi.Float, w2, leaf_indices, valid_nee)
            rn_mu0 = dr.gather(mi.Vector3f, mu0, leaf_indices, valid_nee)
            rn_mu1 = dr.gather(mi.Vector3f, mu1, leaf_indices, valid_nee)
            rn_mu2 = dr.gather(mi.Vector3f, mu2, leaf_indices, valid_nee)
            rn_k0 = dr.gather(mi.Float, k0, leaf_indices, valid_nee)
            rn_k1 = dr.gather(mi.Float, k1, leaf_indices, valid_nee)
            rn_k2 = dr.gather(mi.Float, k2, leaf_indices, valid_nee)

            def eval_vmf(mu, kappa, d):
                cos_t = dr.dot(mu, d)
                safe_kappa = dr.maximum(kappa, 1e-4)
                norm = safe_kappa / (dr.two_pi * (1.0 - dr.exp(-2.0 * safe_kappa)))
                return dr.select(kappa < 1e-4, dr.inv_four_pi, norm * dr.exp(kappa * (cos_t - 1.0)))

            # E-STEP NORMAL
            pdf0 = r_w0 * eval_vmf(r_mu0, r_k0, directions)
            pdf1 = r_w1 * eval_vmf(r_mu1, r_k1, directions)
            pdf2 = r_w2 * eval_vmf(r_mu2, r_k2, directions)
            sum_pdf = pdf0 + pdf1 + pdf2
            valid_pdf = sum_pdf > 1e-8
            resp0 = dr.select(valid_pdf, pdf0 / sum_pdf, 1.0 / 3.0)
            resp1 = dr.select(valid_pdf, pdf1 / sum_pdf, 1.0 / 3.0)
            resp2 = dr.select(valid_pdf, pdf2 / sum_pdf, 1.0 / 3.0)

            # E-STEP NEE
            pdf0_nee = rn_w0 * eval_vmf(rn_mu0, rn_k0, directions_nee)
            pdf1_nee = rn_w1 * eval_vmf(rn_mu1, rn_k1, directions_nee)
            pdf2_nee = rn_w2 * eval_vmf(rn_mu2, rn_k2, directions_nee)
            sum_pdf_nee = pdf0_nee + pdf1_nee + pdf2_nee
            valid_pdf_nee = sum_pdf_nee > 1e-8
            resp0_nee = dr.select(valid_pdf_nee, pdf0_nee / sum_pdf_nee, 1.0 / 3.0)
            resp1_nee = dr.select(valid_pdf_nee, pdf1_nee / sum_pdf_nee, 1.0 / 3.0)
            resp2_nee = dr.select(valid_pdf_nee, pdf2_nee / sum_pdf_nee, 1.0 / 3.0)

            # M-STEP
            eff_w0, eff_w1, eff_w2 = resp0 * weights, resp1 * weights, resp2 * weights
            eff_w0_nee, eff_w1_nee, eff_w2_nee = resp0_nee * weights_nee, resp1_nee * weights_nee, resp2_nee * weights_nee

            sum_eff_w0, sum_eff_w1, sum_eff_w2 = dr.zeros(mi.Float, num_leaves), dr.zeros(mi.Float, num_leaves), dr.zeros(mi.Float, num_leaves)
            R0, R1, R2 = dr.zeros(mi.Vector3f, num_leaves), dr.zeros(mi.Vector3f, num_leaves), dr.zeros(mi.Vector3f, num_leaves)

            # Accumulate Normal
            dr.scatter_reduce(dr.ReduceOp.Add, sum_eff_w0, eff_w0, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, sum_eff_w1, eff_w1, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, sum_eff_w2, eff_w2, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R0.x, directions.x * eff_w0, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R0.y, directions.y * eff_w0, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R0.z, directions.z * eff_w0, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R1.x, directions.x * eff_w1, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R1.y, directions.y * eff_w1, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R1.z, directions.z * eff_w1, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R2.x, directions.x * eff_w2, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R2.y, directions.y * eff_w2, leaf_indices, valid_ray)
            dr.scatter_reduce(dr.ReduceOp.Add, R2.z, directions.z * eff_w2, leaf_indices, valid_ray)

            # Accumulate NEE
            dr.scatter_reduce(dr.ReduceOp.Add, sum_eff_w0, eff_w0_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, sum_eff_w1, eff_w1_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, sum_eff_w2, eff_w2_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R0.x, directions_nee.x * eff_w0_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R0.y, directions_nee.y * eff_w0_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R0.z, directions_nee.z * eff_w0_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R1.x, directions_nee.x * eff_w1_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R1.y, directions_nee.y * eff_w1_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R1.z, directions_nee.z * eff_w1_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R2.x, directions_nee.x * eff_w2_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R2.y, directions_nee.y * eff_w2_nee, leaf_indices, valid_nee)
            dr.scatter_reduce(dr.ReduceOp.Add, R2.z, directions_nee.z * eff_w2_nee, leaf_indices, valid_nee)

            def update_lobe(old_w, old_mu, old_k, sum_eff_w, R_vec):
                new_w = sum_eff_w / dr.select(leaf_has_data, leaf_total_weight, 1.0)
                r_vec = dr.select(sum_eff_w > 1e-6, R_vec / sum_eff_w, old_mu)
                r_bar = dr.norm(r_vec)
                new_mu = dr.select(r_bar > 1e-6, r_vec / r_bar, old_mu)
                r_safe = dr.minimum(r_bar, 0.9999)
                r2 = r_safe * r_safe
                new_k = dr.select(sum_eff_w > 1e-6, (r_safe * (3.0 - r2)) / (1.0 - r2), 0.0)
                return (
                    dr.select(leaf_has_data, new_w, old_w),
                    dr.select(leaf_has_data, new_mu, old_mu),
                    dr.select(leaf_has_data, new_k, old_k)
                )

            w0, mu0, k0 = update_lobe(w0, mu0, k0, sum_eff_w0, R0)
            w1, mu1, k1 = update_lobe(w1, mu1, k1, sum_eff_w1, R1)
            w2, mu2, k2 = update_lobe(w2, mu2, k2, sum_eff_w2, R2)

        MAX_KAPPA = mi.Float(50.0)
        k0 = dr.minimum(k0, MAX_KAPPA)
        k1 = dr.minimum(k1, MAX_KAPPA)
        k2 = dr.minimum(k2, MAX_KAPPA)

        lr = mi.Float(1.0 / (float(current_iteration) + 1.0))

        w0 = dr.lerp(orig_w0, w0, lr)
        w1 = dr.lerp(orig_w1, w1, lr)
        w2 = dr.lerp(orig_w2, w2, lr)

        mu0 = dr.normalize(dr.lerp(orig_mu0, mu0, lr))
        mu1 = dr.normalize(dr.lerp(orig_mu1, mu1, lr))
        mu2 = dr.normalize(dr.lerp(orig_mu2, mu2, lr))

        k0 = dr.lerp(orig_k0, k0, lr)
        k1 = dr.lerp(orig_k1, k1, lr)
        k2 = dr.lerp(orig_k2, k2, lr)

        tot_w = w0 + w1 + w2
        w0 = dr.select(leaf_has_data & (tot_w > 0), w0 / tot_w, w0)
        w1 = dr.select(leaf_has_data & (tot_w > 0), w1 / tot_w, w1)
        w2 = dr.select(leaf_has_data & (tot_w > 0), w2 / tot_w, w2)

        self.kdTreeNode.vmf_w0, self.kdTreeNode.vmf_w1, self.kdTreeNode.vmf_w2 = w0, w1, w2
        self.kdTreeNode.vmf_mu0, self.kdTreeNode.vmf_mu1, self.kdTreeNode.vmf_mu2 = mu0, mu1, mu2
        self.kdTreeNode.vmf_kappa0, self.kdTreeNode.vmf_kappa1, self.kdTreeNode.vmf_kappa2 = k0, k1, k2

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
        """
        Takes a world position, finds the corresponding KD-Tree leaf,
        and reconstructs the full vMFMixture on the fly for the integrator.
        """
        leaf_idx = self.getLeafNodeIndex(position, active)

        # Gather all 3 lobes using pure Structure of Arrays layout
        w0 = dr.gather(mi.Float, self.kdTreeNode.vmf_w0, leaf_idx, active)
        w1 = dr.gather(mi.Float, self.kdTreeNode.vmf_w1, leaf_idx, active)
        w2 = dr.gather(mi.Float, self.kdTreeNode.vmf_w2, leaf_idx, active)

        mu0 = dr.gather(mi.Vector3f, self.kdTreeNode.vmf_mu0, leaf_idx, active)
        mu1 = dr.gather(mi.Vector3f, self.kdTreeNode.vmf_mu1, leaf_idx, active)
        mu2 = dr.gather(mi.Vector3f, self.kdTreeNode.vmf_mu2, leaf_idx, active)

        kappa0 = dr.gather(mi.Float, self.kdTreeNode.vmf_kappa0, leaf_idx, active)
        kappa1 = dr.gather(mi.Float, self.kdTreeNode.vmf_kappa1, leaf_idx, active)
        kappa2 = dr.gather(mi.Float, self.kdTreeNode.vmf_kappa2, leaf_idx, active)

        return vMFMixture(
            weights=[w0, w1, w2],
            mus=[mu0, mu1, mu2],
            kappas=[kappa0, kappa1, kappa2]
        )

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
        """
        np.savez_compressed(
            file=fileName,

            # Hyperparameters
            kdtree_maxLeafSize=self.maxLeafSize,
            kdtree_maxDepth=self.maxDepth,

            # KD-Tree Spatial Topology
            kdtree_bbox_min_x=self.kdTreeNode.bbox.min.x.numpy(),
            kdtree_bbox_min_y=self.kdTreeNode.bbox.min.y.numpy(),
            kdtree_bbox_min_z=self.kdTreeNode.bbox.min.z.numpy(),
            kdtree_bbox_max_x=self.kdTreeNode.bbox.max.x.numpy(),
            kdtree_bbox_max_y=self.kdTreeNode.bbox.max.y.numpy(),
            kdtree_bbox_max_z=self.kdTreeNode.bbox.max.z.numpy(),

            kdtree_depth=self.kdTreeNode.depth.numpy(),
            kdtree_vertCount=self.kdTreeNode.vertCount.numpy(),
            kdtree_isLeaf=self.kdTreeNode.isLeaf.numpy(),
            kdtree_child_left_index=self.kdTreeNode.child_left_index.numpy(),
            kdtree_child_right_index=self.kdTreeNode.child_right_index.numpy(),

            # The Brain (vMF Lobes)
            vmf_w0=self.kdTreeNode.vmf_w0.numpy(),
            vmf_w1=self.kdTreeNode.vmf_w1.numpy(),
            vmf_w2=self.kdTreeNode.vmf_w2.numpy(),

            vmf_kappa0=self.kdTreeNode.vmf_kappa0.numpy(),
            vmf_kappa1=self.kdTreeNode.vmf_kappa1.numpy(),
            vmf_kappa2=self.kdTreeNode.vmf_kappa2.numpy(),

            vmf_mu0_x=self.kdTreeNode.vmf_mu0.x.numpy(),
            vmf_mu0_y=self.kdTreeNode.vmf_mu0.y.numpy(),
            vmf_mu0_z=self.kdTreeNode.vmf_mu0.z.numpy(),

            vmf_mu1_x=self.kdTreeNode.vmf_mu1.x.numpy(),
            vmf_mu1_y=self.kdTreeNode.vmf_mu1.y.numpy(),
            vmf_mu1_z=self.kdTreeNode.vmf_mu1.z.numpy(),

            vmf_mu2_x=self.kdTreeNode.vmf_mu2.x.numpy(),
            vmf_mu2_y=self.kdTreeNode.vmf_mu2.y.numpy(),
            vmf_mu2_z=self.kdTreeNode.vmf_mu2.z.numpy(),
        )

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

        # Load The Brain (vMF Lobes)
        self.kdTreeNode.vmf_w0 = mi.Float(dataNumpy['vmf_w0'])
        self.kdTreeNode.vmf_w1 = mi.Float(dataNumpy['vmf_w1'])
        self.kdTreeNode.vmf_w2 = mi.Float(dataNumpy['vmf_w2'])

        self.kdTreeNode.vmf_kappa0 = mi.Float(dataNumpy['vmf_kappa0'])
        self.kdTreeNode.vmf_kappa1 = mi.Float(dataNumpy['vmf_kappa1'])
        self.kdTreeNode.vmf_kappa2 = mi.Float(dataNumpy['vmf_kappa2'])

        self.kdTreeNode.vmf_mu0 = mi.Vector3f(dataNumpy['vmf_mu0_x'], dataNumpy['vmf_mu0_y'], dataNumpy['vmf_mu0_z'])
        self.kdTreeNode.vmf_mu1 = mi.Vector3f(dataNumpy['vmf_mu1_x'], dataNumpy['vmf_mu1_y'], dataNumpy['vmf_mu1_z'])
        self.kdTreeNode.vmf_mu2 = mi.Vector3f(dataNumpy['vmf_mu2_x'], dataNumpy['vmf_mu2_y'], dataNumpy['vmf_mu2_z'])

        # Seal the memory graph
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
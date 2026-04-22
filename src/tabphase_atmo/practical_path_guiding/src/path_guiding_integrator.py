from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
if __name__ == '__main__':
    mi.set_variant('llvm_ad_rgb')

from practical_path_guiding.src.common import *

import numpy as np

from practical_path_guiding.src.kdtree import KDTree

epsilon = 0.00001

def mis_weight(pdf_a, pdf_b):
    """
        Compute the Multiple Importance Sampling (MIS) weight given the densities
        of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    result = dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0)
    result[ dr.isnan( result ) ] = 0
    return result


class PathGuidingIntegrator(mi.SamplingIntegrator):

    def __init__(self: mi.SamplingIntegrator, props: mi.Properties) -> None:
        super().__init__(props)

        # Read properties
        # 	Max Depth
        self.max_depth = props.get('max_depth', 30)
        if self.max_depth < 0 and self.max_depth != -1:
            raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # 	Russian Roulette
        self.rr_depth = props.get('rr_depth', 8)
        if self.rr_depth < 0:
            raise Exception("\"rr_depth\" must be set to >= 0")

        # Declare properties
        self.numRays: int = 0
        self.array_size: int = 0
        self.isStoreNEERadiance: bool = False
        """
            When guiding, we perform MIS with the balance heuristic between the guiding
            distribution and the BSDF, combined with probabilistically choosing one of the
            two sampling methods. This factor controls how often the BSDF is sampled
            vs. how often the guiding distribution is sampled.
            Default = 0.5 (50%)
        """
        self.bsdfSamplingFraction = 0.5
        self.iteration = 0
        self.isFinalIter = False

        # Data recorder for transferring into the SD-Tree
        self.surfaceInteractionRecord: SurfaceInteractionRecord = None

        """
            Two SDTrees.
            There are two SDTrees: sdTree_current, and sdTree_prev.
            In each iteration, we sample from sdTree_prev and store into sdTree_current.
            At the end of the iteration, we refine the sdTree_current, copy from sdTree_current into sdTree_prev,
            and then remove data in the sdTree_current but keeps the structure, ready to be use again.
        """
        self.sdTree_prev = KDTree()
        self.sdTree_current = KDTree()


        # For variance calculation
        self.sumL = mi.Spectrum(0)
        self.sumL2 = mi.Spectrum(0)


    def setup( self: mi.SamplingIntegrator,
        numRays: int,
        bbox_min: mi.Vector3f,
        bbox_max: mi.Vector3f,
        sdTreeMaxDepth: int = 10,
        quadTreeMaxDepth: int = 30,
        isStoreNEERadiance: bool = True,
        bsdfSamplingFraction: float = 0.5
    ) -> None:
        """
            Setting SDTree and other self attributes.
            This function need to be called BEFORE starting the rendering process.
        """

        # Self properties
        self.numRays = numRays
        self.array_size = self.numRays * self.max_depth
        self.isStoreNEERadiance = isStoreNEERadiance
        self.bsdfSamplingFraction = bsdfSamplingFraction
        self.resetRayPathData()

        # SDTrees
        # 	SDTree current
        self.sdTree_current.setup( bbox_min, bbox_max )
        self.sdTree_current.maxDepth = sdTreeMaxDepth
        self.sdTree_current.quadTree.maxDepth = quadTreeMaxDepth
        self.sdTree_current.quadTree.isStoreNEERadiance = isStoreNEERadiance
        # 	SDTree prev
        self.sdTree_prev.copyFrom( self.sdTree_current )


    def resetVarianceCounter( self ) -> None:
        self.sumL = mi.Spectrum(0)
        self.sumL2 = mi.Spectrum(0)


    def resetRayPathData(self: mi.SamplingIntegrator):

        if not self.isFinalIter:
            self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape= self.array_size)
        else:
            self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape= 1)


    def setIteration(self, iteration: int, isFinalIter: bool) -> None:
        self.iteration = iteration
        self.isFinalIter = isFinalIter


    def sample(self: mi.SamplingIntegrator,
                scene: mi.Scene,
                sampler: mi.Sampler,
                ray: mi.RayDifferential3f,
                medium: mi.Medium = None,
                active: bool = True,
                aovs: mi.Float = None,
                ) -> Tuple[mi.Color3f, bool, List[float]]:

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        #
        # 	Configure Loop State
        #

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(ray)
        ray_width = dr.width(ray)

        #         Path throughput weight
        throughput_weight = dr.full(mi.Spectrum, 1.0, ray_width)
        #         Depth of current vertex
        depth = dr.zeros(mi.UInt32, ray_width)
        #         Radiance accumulator
        L = dr.zeros(mi.Spectrum, ray_width)
        #         Index of refraction
        ior = dr.full(mi.Float, 1.0, ray_width)
        #         Active SIMD lanes
        active = mi.Bool(active)

        self.resetRayPathData()

        # Generate ray index
        ray_index = dr.arange( mi.UInt32, dr.width(ray) )

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        isFinalIter = mi.Bool( self.isFinalIter )
        # =========== MEDIUM INIT ================
        # Create a properly typed Dr.Jit null medium array so the compiler doesn't optimize the fog away
        dummy_si = dr.zeros(mi.SurfaceInteraction3f, ray_width)
        null_medium = dummy_si.target_medium(ray.d)

        # Override with environment if it exists, otherwise use the typed null array
        env_medium = scene.environment() if hasattr(scene, 'environment') else None
        medium = env_medium if env_medium is not None else null_medium
        # (Assuming the scene has a global medium. If your fog is inside a specific shape, we grab it from the ray intersect later)

        mei = dr.zeros(mi.MediumInteraction3f, ray_width)

        # Spectral channel selection for tracking MIS correctly
        channel = mi.UInt32(0)
        n_channels = mi.UInt32(dr.size_v(mi.Spectrum))
        channel = dr.minimum(sampler.next_1d(active) * n_channels, n_channels - 1)
        channel = mi.UInt32(channel)

        valid_ray = mi.Bool(False)

        loop_state = (sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                      isFinalIter, medium, mei, channel, valid_ray)

        def loop_cond(sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                      isFinalIter, medium, mei, channel, valid_ray):
            return active

        def loop_body(sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta, isFinalIter, medium, mei, channel, valid_ray):
            #
            # 	Direct Emission
            #

            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=depth == 0)
            # =========================================================================
            # PHASE 2: SAMPLING THE RTE
            # =========================================================================
            active_medium = active & (medium != None)

            # 1. Sample the medium
            mei = medium.sample_interaction(ray, sampler.next_1d(active_medium), channel, active_medium)

            # 2. Check if we hit the surface before the medium particle
            mei.t[active_medium & (si.t < mei.t)] = dr.inf

            # 3. Transmittance Evaluation (Spectral)
            is_spectral = active_medium & medium.has_spectral_extinction()
            tr, free_flight_pdf = medium.transmittance_eval_pdf(mei, si, is_spectral)

            # Safe dynamic spectral channel routing (bypasses gather indexing errors)
            tr_pdf = dr.select(channel == 0, free_flight_pdf[0],
                               dr.select(channel == 1, free_flight_pdf[1],
                                         dr.select(channel == 2, free_flight_pdf[2], free_flight_pdf[3])))

            throughput_weight[is_spectral] *= dr.select(tr_pdf > 0.0, tr / tr_pdf, 0.0)

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()

            # 4. Handle Null vs Real Scatter
            null_scatter_prob = dr.mean(mei.sigma_n / mei.combined_extinction)
            null_scatter = sampler.next_1d(active_medium) < null_scatter_prob

            act_null_scatter = null_scatter & active_medium
            act_medium_scatter = ~act_null_scatter & active_medium

            throughput_weight[is_spectral & act_null_scatter] *= mei.sigma_n / null_scatter_prob

            depth[act_medium_scatter] += 1

            # Update Ray for null scatter (moves forward, skips surface eval)
            ray.o[act_null_scatter] = mei.p
            si.t[act_null_scatter] = si.t - mei.t

            # Throughput for real scatter
            throughput_weight[is_spectral & act_medium_scatter] *= mei.sigma_s / dr.mean(
                mei.sigma_t / mei.combined_extinction)

            not_spectral = active_medium & ~is_spectral
            throughput_weight[not_spectral & act_medium_scatter] *= mei.sigma_s / mei.sigma_t

            # Define what actually hits the surface now
            active_surface = (active & (medium == None)) | escaped_medium
            # =========================================================================

            # =========================================================================
            # PHASE 3: THE UNIFIED SCATTER & NEE (Surface + Medium)
            # =========================================================================
            bsdf = si.bsdf()
            phase = medium.phase_function() if medium is not None else None
            phase_ctx = mi.PhaseFunctionContext(sampler)

            # 1. Unified Position for the Spotter
            eval_p = dr.select(act_medium_scatter, mei.p, si.p)

            # 2. Direct Emission (Only surfaces emit light directly when hit)
            ds_direct = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            emitter_pdf = scene.pdf_emitter_direction(prev_si, ds_direct, ~prev_bsdf_delta)
            mis = mis_weight(prev_bsdf_pdf, emitter_pdf)
            em_radiance = ds_direct.emitter.eval(si)
            Le = dr.select(active_surface, throughput_weight * mis * em_radiance, 0.0)

            # -------------------------------------------------------------------------
            # Next Event Estimation (NEE) - THE SHADOW RAY BUG FIX
            # -------------------------------------------------------------------------
            active_next = (depth + 1 < self.max_depth) & (si.is_valid() | act_medium_scatter)
            active_em_surface = active_next & active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            active_em_medium = active_next & act_medium_scatter

            # Evaluate NEE securely using native structs so geometric normals offset shadow rays correctly
            ds_surface, em_weight_surface = scene.sample_emitter_direction(si, sampler.next_2d(), test_visibility=True,
                                                                           active=active_em_surface)
            ds_medium, em_weight_medium = scene.sample_emitter_direction(mei, sampler.next_2d(), test_visibility=True,
                                                                         active=active_em_medium)

            # Merge outputs dynamically
            ds = dr.select(act_medium_scatter, ds_medium, ds_surface)
            em_weight = dr.select(act_medium_scatter, em_weight_medium, em_weight_surface)

            active_em = (active_em_surface | active_em_medium) & (ds.pdf > 0.0)

            # Evaluate BSDF
            wo_local_surface = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo_local_surface, active_em & active_surface)

            # Evaluate Phase Function (Using Coordinate Space Fix)
            if phase is not None:
                wo_local_medium = mei.to_local(ds.d)
                phase_value_em, phase_pdf_em = phase.eval_pdf(phase_ctx, mei, wo_local_medium,
                                                              active_em & act_medium_scatter)
            else:
                phase_value_em, phase_pdf_em = mi.Spectrum(0), mi.Float(0)

            # Unify the Evaluation
            scatter_value_em = dr.select(act_medium_scatter, phase_value_em, bsdf_value_em)
            scatter_pdf_em = dr.select(act_medium_scatter, phase_pdf_em, bsdf_pdf_em)

            active_sdtree_and_em = active_em & (self.iteration > 1)

            # SD-Tree Guiding PDF for NEE
            sdtree_pdf_em = self.sdTree_prev.pdf(eval_p, ds.d, active_sdtree_and_em)

            # MIS PDF: 50% Tree, 50% Material/Phase
            surface_pdf_em = self.bsdfSamplingFraction * scatter_pdf_em + (
                        1 - self.bsdfSamplingFraction) * sdtree_pdf_em
            surface_pdf_em = dr.select((self.iteration <= 1), scatter_pdf_em, surface_pdf_em)

            # MIS Weight & NEE Accumulation
            mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, surface_pdf_em))
            Lr_dir = throughput_weight * mis_em * scatter_value_em * em_weight

            L += Le + Lr_dir

            # -------------------------------------------------------------------------
            # Next Outgoing Ray Sampling (The Spotter MIS)
            # -------------------------------------------------------------------------
            # Sample BSDF natively
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active_surface),
                                                   sampler.next_2d(active_surface), active_surface)

            # Sample Phase Function natively (Coordinate Space Fix)
            if phase is not None:
                phase_wo_local, phase_weight, phase_pdf = phase.sample(phase_ctx, mei,
                                                                       sampler.next_1d(act_medium_scatter),
                                                                       sampler.next_2d(act_medium_scatter),
                                                                       act_medium_scatter)
                phase_wo_world = mei.to_world(phase_wo_local)
            else:
                phase_wo_world, phase_weight, phase_pdf = mi.Vector3f(0), mi.Spectrum(0), mi.Float(0)

            # Unify Initial Samples
            wo_world = dr.select(act_medium_scatter, phase_wo_world, si.to_world(bsdf_sample.wo))
            scatter_weight_sampled = dr.select(act_medium_scatter, phase_weight, bsdf_weight)
            scatter_pdf_sampled = dr.select(act_medium_scatter, phase_pdf, bsdf_sample.pdf)
            delta = active_surface & mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            do_sdtree_mis = active_next & ~delta & (self.iteration > 1)
            active_sample_sdtree_mis = (sampler.next_1d(active_next) > self.bsdfSamplingFraction) & do_sdtree_mis
            active_sample_scatter_mis = do_sdtree_mis & ~active_sample_sdtree_mis

            # Sample the Spotter (SD-Tree)
            sdtree_dir, sdtree_pdf = self.sdTree_prev.sample(position=eval_p, sampler=sampler,
                                                             active=active_sample_sdtree_mis)
            wo_world[active_sample_sdtree_mis] = sdtree_dir

            # Re-evaluate Material/Phase for the tree's chosen direction
            wo_local_sdtree = si.to_local(sdtree_dir)
            bsdf_value_sd, bsdf_pdf_sd = bsdf.eval_pdf(bsdf_ctx, si, wo_local_sdtree,
                                                       active_sample_sdtree_mis & active_surface)

            if phase is not None:
                wo_local_medium_sd = mei.to_local(sdtree_dir)
                phase_value_sd, phase_pdf_sd = phase.eval_pdf(phase_ctx, mei, wo_local_medium_sd,
                                                              active_sample_sdtree_mis & act_medium_scatter)
            else:
                phase_value_sd, phase_pdf_sd = mi.Spectrum(0), mi.Float(0)

            # Unify Tree Evaluation
            scatter_value_sd = dr.select(act_medium_scatter, phase_value_sd, bsdf_value_sd)
            scatter_pdf_sd = dr.select(act_medium_scatter, phase_pdf_sd, bsdf_pdf_sd)

            # Overwrite variables if SDTree was used (safe division)
            valid_sd_pdf = scatter_pdf_sd > 0.0
            scatter_weight_sampled[active_sample_sdtree_mis] = dr.select(valid_sd_pdf,
                                                                         scatter_value_sd / scatter_pdf_sd, 0.0)
            scatter_pdf_sampled[active_sample_sdtree_mis] = scatter_pdf_sd

            # Fetch SD-Tree PDF for the Material/Phase's originally chosen direction
            sdtree_pdf[active_sample_scatter_mis] = self.sdTree_prev.pdf(eval_p, wo_world, active_sample_scatter_mis)

            # Finalize Unified PDF and MIS Weight
            woPdf = mi.Float(scatter_pdf_sampled)
            woPdf[active_next & do_sdtree_mis] = (self.bsdfSamplingFraction * scatter_pdf_sampled) + (
                        (1 - self.bsdfSamplingFraction) * sdtree_pdf)

            # NaN Protection for throughput division
            valid_woPdf = woPdf > 0.0
            mis_weight_final = dr.select(valid_woPdf, (scatter_weight_sampled * scatter_pdf_sampled) / woPdf, 0.0)
            scatter_weight_final = dr.select(do_sdtree_mis, mis_weight_final, scatter_weight_sampled)

            # Overwrite for storage block
            bsdf_weight = scatter_weight_final
            bsdf_sample.pdf = scatter_pdf_sampled

            #
            #     Store surface interaction data
            #
            globalIndex = (ray_index * self.max_depth) + depth
            storeFlag = active & (si.is_valid() | act_medium_scatter)
            storeFlag &= ~isFinalIter

            # --- THE DECOMPOSED SOA SCATTERS ---
            dr.scatter(self.surfaceInteractionRecord.position.x, value=eval_p.x, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.position.y, value=eval_p.y, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.position.z, value=eval_p.z, index=globalIndex, active=storeFlag)

            canon_wo = dirToCanonical(wo_world)
            dr.scatter(self.surfaceInteractionRecord.direction.x, value=canon_wo.x, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.direction.y, value=canon_wo.y, index=globalIndex, active=storeFlag)

            dr.scatter(self.surfaceInteractionRecord.active, value=storeFlag, index=globalIndex, active=storeFlag)

            for c in range(len(L)):
                dr.scatter(self.surfaceInteractionRecord.bsdf[c], value=bsdf_weight[c], index=globalIndex,
                           active=storeFlag)
                dr.scatter(self.surfaceInteractionRecord.throughputBsdf[c], value=throughput_weight[c],
                           index=globalIndex, active=storeFlag)
                dr.scatter(self.surfaceInteractionRecord.throughputRadiance[c], value=L[c], index=globalIndex,
                           active=storeFlag)

            isStoreNEERadiance = self.isStoreNEERadiance & storeFlag
            nee_rad = Lr_dir / dr.select(throughput_weight > 0, throughput_weight, 1.0)  # Prevent 0 division

            for c in range(len(nee_rad)):
                dr.scatter(self.surfaceInteractionRecord.radiance_nee[c], value=nee_rad[c], index=globalIndex,
                           active=isStoreNEERadiance)

            canon_ds = dirToCanonical(ds.d)
            dr.scatter(self.surfaceInteractionRecord.direction_nee.x, value=canon_ds.x, index=globalIndex,
                       active=isStoreNEERadiance)
            dr.scatter(self.surfaceInteractionRecord.direction_nee.y, value=canon_ds.y, index=globalIndex,
                       active=isStoreNEERadiance)

            dr.scatter(self.surfaceInteractionRecord.woPdf, value=woPdf, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.bsdfPdf, value=bsdf_sample.pdf, index=globalIndex,
                       active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.isDelta, value=delta, index=globalIndex, active=storeFlag)

            #
            #     Update loop variables based on current interaction
            #

            # Safely spawn rays depending on if we hit a surface or fog
            ray_surface = si.spawn_ray(wo_world)
            ray_medium = mei.spawn_ray(wo_world)

            ray = mi.Ray3f(
                dr.select(act_medium_scatter, ray_medium.o, ray_surface.o),
                wo_world,
                si.time,
                ray.wavelengths
            )

            ior *= dr.select(act_medium_scatter, 1.0, bsdf_sample.eta)
            throughput_weight *= bsdf_weight

            # =========================================================================
            # BOUNDARY TRACKING FIX
            # If the ray transmits through a boundary (like glass), update the medium!
            # =========================================================================
            has_medium_trans = active_surface & si.is_valid() & si.is_medium_transition()
            medium = dr.select(has_medium_trans, si.target_medium(ray.d), medium)

            prev_si = si
            prev_bsdf_pdf = woPdf
            prev_bsdf_delta = delta

            #
            #     Stopping criterion
            #
            throughput_weight_max = dr.max(throughput_weight)
            active_next &= (throughput_weight_max != 0)

            rr_prob = dr.minimum(throughput_weight_max * ior ** 2, 0.95)
            rr_active = depth >= self.rr_depth
            throughput_weight_max[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            active = active_next
            # Only increment surface depth if it hit the surface
            depth[active_surface & si.is_valid()] += 1

            return (sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                    isFinalIter, medium, mei, channel, valid_ray)

        sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta, isFinalIter, medium, mei, channel, valid_ray = dr.while_loop(
            state=loop_state,
            cond=loop_cond,
            body=loop_body,
            max_iterations=self.max_depth
        )

        dr.schedule(L)
        sampler.schedule_state()
        dr.schedule(self.surfaceInteractionRecord)

        # No need to update tree in the final iteration
        if not self.isFinalIter:

            # Process path data
            Lfinal = mi.Spectrum( L )
            self.processPathData( Lfinal )

            # Scatter the stored data into SD-Tree
            self.scatterDataIntoSDTree()


        # Accumulate Radiance for variance calculation

        spp_per_pass = sampler.sample_count()

        if spp_per_pass == 1:

            self.sumL += L
            self.sumL2 += L * L

        else:

            # Process sample for each SPP
            total_size = dr.width( L )

            one_sample_size = int( total_size / spp_per_pass )

            # Each sample within render pass is lay out in a concatination fashion next to eachc other.
            # For example, if SPP per pass is 4, then:
            # 	pixel_1_sample_1, pixel_1_sample_2, pixel_1_sample_3, pixel_1_sample_4, pixel_2_sample_1, pixel_2_sample_2, ...
            base_index = dr.arange( mi.UInt32, one_sample_size ) * spp_per_pass		#	0 4 8 12 ...

            for i in range( spp_per_pass ):

                index = base_index + i		# 	e.g. i=1: 1 5 9 13 ...

                one_spp_L = dr.gather( mi.Spectrum, L, index )

                self.sumL += one_spp_L
                self.sumL2 += one_spp_L * one_spp_L


        dr.schedule( self.sumL, self.sumL2 )

        return (L, depth != 0, [1])

    def processPathData(self: mi.SamplingIntegrator, Lfinal: mi.Spectrum) -> None:
        """    Calculate incoming radiance at each vertex
        """

        # This launches one thread per vertex:
        globalVertexIndex = dr.arange(mi.UInt32, self.array_size)
        rayIndex = globalVertexIndex // self.max_depth  # 000001111122222...
        LfinalPerVertex = dr.gather(mi.Spectrum, Lfinal, rayIndex)

        outgoingRadiance = (
                                       LfinalPerVertex - self.surfaceInteractionRecord.throughputRadiance) / self.surfaceInteractionRecord.throughputBsdf

        # Functional NaN selection (You already fixed this one)
        outgoingRadiance = dr.select(dr.isnan(outgoingRadiance), 0.0, outgoingRadiance)

        # ---------------------------------------------------------
        # THE FIX: Decomposed dynamic scatter for Spectral/RGB
        # ---------------------------------------------------------
        for c in range(len(outgoingRadiance)):
            dr.scatter(self.surfaceInteractionRecord.product[c], outgoingRadiance[c], globalVertexIndex)

        #incomingRadiance = outgoingRadiance / self.surfaceInteractionRecord.bsdf


        # ---------------------------------------------------------
        # THE SECOND FIX: Functional NaN selection for incomingRadiance
        # ---------------------------------------------------------
        #incomingRadiance = dr.select(dr.isnan(incomingRadiance), 0.0, incomingRadiance)

        # Convert from Spectrum to Luminance (Luminance is a 1D float, so this single scatter is safe)
        # Bypass mi.luminance by explicitly averaging the spectral channels.
        # This gives the SD-Tree a valid 1D scalar energy array without needing the ray's wavelength data.
        #incomingRadiance_lum = sum(incomingRadiance) / len(incomingRadiance)
        #incomingRadiance_lum = dr.log(1.0 + incomingRadiance_lum)
        outgoingRadiance_lum = sum(outgoingRadiance) / len(outgoingRadiance)
        dr.scatter(self.surfaceInteractionRecord.radiance, outgoingRadiance_lum, globalVertexIndex)

    def scatterDataIntoSDTree(self: mi.SamplingIntegrator) -> None:
        """
            Scatter surface interaction data into current SDTree.
            This also filter out invalid surface interactions before scattering.
        """

        # Remove surfaceInteractionRecord that are inactive. This modifies the array size.
        isActive = self.surfaceInteractionRecord.active

        # Filter NaN radiance using functional selection
        self.surfaceInteractionRecord.radiance = dr.select(dr.isnan(self.surfaceInteractionRecord.radiance), 0.0,
                                                           self.surfaceInteractionRecord.radiance)

        # Note: In spectral mode, radiance_nee is a Spectrum, so we do it channel by channel or use the broadcasted select
        self.surfaceInteractionRecord.radiance_nee = dr.select(dr.isnan(self.surfaceInteractionRecord.radiance_nee),
                                                               0.0, self.surfaceInteractionRecord.radiance_nee)

        # ---> THE MIE FIREFLY MUZZLE <---
        # Compress massive NEE spikes so they don't nuke the QuadTree PDF
        #self.surfaceInteractionRecord.radiance_nee = dr.log(1.0 + self.surfaceInteractionRecord.radiance_nee)

        # Check if both radiance is zero then also filter out
        radiance_zero = self.surfaceInteractionRecord.radiance == 0
        radiance_nee_lum = sum(self.surfaceInteractionRecord.radiance_nee) / len(
            self.surfaceInteractionRecord.radiance_nee)
        radiance_nee_zero = radiance_nee_lum == 0
        bothRadianceZero = radiance_zero & radiance_nee_zero

        # Remove woPdf = 0 or NaN
        woPdf_zero = self.surfaceInteractionRecord.woPdf == 0
        woPdf_nan = dr.isnan(self.surfaceInteractionRecord.woPdf)
        activeIndex = dr.compress(isActive & ~bothRadianceZero & ~woPdf_zero & ~woPdf_nan)


        # If there is no active element then exit
        if( dr.width( activeIndex ) == 0 ):
            return

        self.surfaceInteractionRecord.position = dr.gather( type(self.surfaceInteractionRecord.position), self.surfaceInteractionRecord.position, activeIndex )

        self.surfaceInteractionRecord.direction = dr.gather( type(self.surfaceInteractionRecord.direction), self.surfaceInteractionRecord.direction, activeIndex )
        self.surfaceInteractionRecord.radiance = dr.gather( type(self.surfaceInteractionRecord.radiance), self.surfaceInteractionRecord.radiance, activeIndex )

        if self.isStoreNEERadiance:
            self.surfaceInteractionRecord.radiance_nee = dr.gather( type(self.surfaceInteractionRecord.radiance_nee), self.surfaceInteractionRecord.radiance_nee, activeIndex )
            self.surfaceInteractionRecord.direction_nee = dr.gather( type(self.surfaceInteractionRecord.direction_nee), self.surfaceInteractionRecord.direction_nee, activeIndex )

        self.surfaceInteractionRecord.product = dr.gather( type(self.surfaceInteractionRecord.product), self.surfaceInteractionRecord.product, activeIndex )
        self.surfaceInteractionRecord.woPdf = dr.gather( type(self.surfaceInteractionRecord.woPdf), self.surfaceInteractionRecord.woPdf, activeIndex )
        self.surfaceInteractionRecord.bsdfPdf = dr.gather( type(self.surfaceInteractionRecord.bsdfPdf), self.surfaceInteractionRecord.bsdfPdf, activeIndex )
        self.surfaceInteractionRecord.isDelta = dr.gather( type(self.surfaceInteractionRecord.isDelta), self.surfaceInteractionRecord.isDelta, activeIndex )

        # Scatter into current SDTree
        self.sdTree_current.addDataPropagate( self.surfaceInteractionRecord )

    def computeMSE(self, spp: float, groundTruth: mi.Color3f) -> float:
        """
            Compute Mean Square Error with respect to ground truth.
            If ground truth is not provided then compare to itself.
        """
        L = self.sumL / spp

        # Convert both arrays to 1D scalar energy before math
        L_scalar = sum(L) / len(L)
        gt_scalar = mi.luminance(groundTruth)  # groundTruth is strictly Color3f from the EXR

        # MSE compare to ground truth
        mse = (L_scalar - gt_scalar) ** 2

        mse = dr.minimum(mse, 10000.0)  # Cutoff outlier for more stable mse
        mse = dr.mean(mse)[0]

        return mse

    def computeVariance(self, spp: float, groundTruth: mi.Color3f = None) -> float:
        """
            Compute Variance
            If ground truth is not provided then compare to itself.
        """

        if groundTruth is not None:
            L2 = self.sumL2 / spp

            # Convert both to scalar
            L2_scalar = sum(L2) / len(L2)
            gt_scalar = mi.luminance(groundTruth)

            variance = L2_scalar - (gt_scalar * gt_scalar)

            variance = dr.minimum(variance, 10000.0)
            variance = dr.mean(variance)[0]

            # Population variance
            variance /= spp

        else:
            L = self.sumL / spp
            L2 = self.sumL2 / spp

            # Convert to scalar
            L_scalar = sum(L) / len(L)
            L2_scalar = sum(L2) / len(L2)

            variance = L2_scalar - (L_scalar * L_scalar)

            variance = dr.minimum(variance, 10000.0)
            variance = dr.mean(variance)[0]

            if spp > 1:
                # Sample variance
                variance /= spp - 1

        return variance


    def refine(self) -> None:
        """
            Refine the current SDTree.
        """

        # Refine the KDTree
        self.sdTree_current.refine()

        # Refind QuadTree
        self.sdTree_current.setQuadTreeRefinementThreshold()
        self.sdTree_current.refineAllQuadTree()


    def refineAndPrepareSDTreeForNextIteration(self) -> None:
        """
            Prepare SDTree for the next iteration.
            1. Refine the current SDTree
            2. Clean unused quadtree in current SDTree
            3. Copy from current SDTree to previous SDTree
            4. Reset data of the current SDTree
        """
        # 1. Set refinement threshold and refine the current SDTree
        self.sdTree_current.setRefinementThreshold( self.iteration )
        self.refine()

        # 2. Clean unused quadtree in current SDTree
        self.sdTree_current.cleanUnusedQuadTree()

        # 3. Copy from current SDTree to previous SDTree
        self.sdTree_prev.copyFrom( self.sdTree_current )

        self.sdTree_current.resetTreeVertCount()
        self.sdTree_current.resetAllQuadTreeIrradiance()


    def saveSDTreeToFile(self, fileName: str) -> None:
        """
            Save previous SDTree data which contains both values and structure of the tree,
            into a file.
        """
        self.sdTree_prev.saveToFile(fileName)


    def loadSDTreeFromFile(self, fileName: str) -> None:
        """
            Load sdTree data from a given .npz file
        """
        dataNumpy = np.load( fileName )
        self.sdTree_prev.loadFromFile( dataNumpy )

        self.sdTree_current.copyFrom( self.sdTree_prev )

        # 4. Reset current SDTree data
        self.sdTree_current.resetTreeVertCount()
        self.sdTree_current.resetAllQuadTreeIrradiance()


    def saveSDTreeOBJ( self, fileName: str ) -> None:
        """
            Write KDTree bounding boxes to a Wavefront Obj file
        """
        self.sdTree_prev.saveOBJ( fileName )


    def aov_names(self):
        # Not really sure what it's for but it's required
        return ["depth.Y"]


    def to_string(self):
        return "path_guiding_integrator"



mi.register_integrator('path_guiding_integrator', lambda props: PathGuidingIntegrator(props))
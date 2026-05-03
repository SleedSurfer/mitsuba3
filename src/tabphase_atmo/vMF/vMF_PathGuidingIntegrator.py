from __future__ import annotations as __annotations__  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
from typing import Tuple, List

from vMF.common import *
import numpy as np

epsilon = 0.00001


def mis_weight(pdf_a, pdf_b):
    """
        Compute the Multiple Importance Sampling (MIS) weight given the densities
        of two sampling strategies according to the power heuristic.
    """
    a2 = dr.square(pdf_a)
    result = dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0.0)
    result = dr.select(dr.isnan(result), 0.0, result)
    return result


class vMF_PathGuidingIntegrator(mi.SamplingIntegrator):

    def __init__(self: mi.SamplingIntegrator, props: mi.Properties) -> None:
        super().__init__(props)

        # Read properties
        self.max_depth = props.get('max_depth', 30)
        if self.max_depth < 0 and self.max_depth != -1:
            raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        self.rr_depth = props.get('rr_depth', 8)
        if self.rr_depth < 0:
            raise Exception("\"rr_depth\" must be set to >= 0")

        self.phase_kappa = mi.Float(props.get('phase_kappa', 0.0))

        # Declare properties
        self.numRays: int = 0
        self.array_size: int = 0
        self.isStoreNEERadiance: bool = False

        # Spatial Hash Map (Injected by Benchmark Script)
        self.photon_map = None

        self.guidingActive = False
        self.iteration = 0
        self.isFinalIter = False

        # For variance calculation
        self.sumL = mi.Spectrum(0)
        self.sumL2 = mi.Spectrum(0)

        # Dummy medium with zero extinction for safe NEE tracking
        self._null_medium = mi.load_dict({
            "type": "homogeneous",
            "sigma_t": 0.0,
            "albedo": 0.0
        })

        # Dummy shape/bsdf for safe bsdf() calls
        self._null_shape = mi.load_dict({
            "type": "sphere",
            "radius": 0.0,
            "bsdf": {"type": "null"}
        })

        # A dummy surface interaction tied to the null shape
        null_ray = mi.Ray3f(o=[0, 0, 0], d=[1, 0, 0])
        self._null_si = self._null_shape.ray_intersect(null_ray, True)

    def setup(self: mi.SamplingIntegrator,
              numRays: int,
              bbox_min: mi.Vector3f,
              bbox_max: mi.Vector3f,
              sdTreeMaxDepth: int = 10,
              isStoreNEERadiance: bool = True,
              bsdfSamplingFraction: float = 0.5,
              guidingStartIteration: int = 3,
              guidingRampIterations: int = 4,
              sdtreePdfPrior: float = 0.05,
              trainingRadianceClamp: float = 250.0,
              trainingLogCompression: bool = True
              ) -> None:
        """
            Setup is preserved so external scripts passing kwargs do not crash.
            Data recorder and KDTree arrays have been purged.
        """
        self.numRays = numRays
        self.array_size = self.numRays * self.max_depth
        self.isStoreNEERadiance = isStoreNEERadiance

    def resetVarianceCounter(self) -> None:
        self.sumL = mi.Spectrum(0)
        self.sumL2 = mi.Spectrum(0)

    def setIteration(self, iteration: int, isFinalIter: bool) -> None:
        self.iteration = iteration
        self.isFinalIter = isFinalIter

    def _index_spectrum(self, spec, idx):
        """Select one spectral component using a lane-wise index."""
        value = spec[0]
        for i in range(1, len(spec)):
            value = dr.select(idx == i, spec[i], value)
        return value

    def _eval_shadow_transmittance(self, scene: mi.Scene, sampler: mi.Sampler, shadow_ray: mi.Ray3f, shadow_medium,
                                   channel: mi.UInt32, active: mi.Bool) -> mi.Spectrum:
        ray = mi.Ray3f(shadow_ray)
        max_dist = ray.maxt
        transmittance = dr.full(mi.Spectrum, 1.0, dr.width(ray))
        total_dist = dr.zeros(mi.Float, dr.width(ray))
        needs_intersection = mi.Bool(True)
        si = dr.zeros(mi.SurfaceInteraction3f, dr.width(ray))
        medium = shadow_medium

        loop_state = (active, ray, total_dist, needs_intersection, si, transmittance, medium, sampler)

        def cond(active, ray, total_dist, needs_intersection, si, transmittance, medium, sampler):
            return dr.detach(active)

        def body(active, ray, total_dist, needs_intersection, si, transmittance, medium, sampler):
            remaining_dist = max_dist - total_dist
            ray.maxt = remaining_dist
            active &= remaining_dist > 0.0

            active_medium = active & (medium != None)
            active_surface = active & ~active_medium

            # --- 1. THE VOLUMETRIC RATIO TRACKER ---
            mei = medium.sample_interaction(ray, sampler.next_1d(active_medium), channel, active_medium)
            ray.maxt = dr.select(active_medium & medium.is_homogeneous() & mei.is_valid(),
                                 dr.minimum(mei.t, remaining_dist), ray.maxt)
            intersect = needs_intersection & active_medium
            si = dr.select(intersect, scene.ray_intersect(ray, active=intersect), si)
            needs_intersection &= ~active_medium
            mei.t = dr.select(active_medium & (si.t < mei.t), dr.inf, mei.t)

            is_spectral = active_medium & medium.has_spectral_extinction()
            not_spectral = active_medium & ~is_spectral

            t_cap = dr.minimum(remaining_dist, dr.minimum(mei.t, si.t)) - mei.mint
            tr_eval = dr.exp(-t_cap * mei.combined_extinction)

            free_flight_pdf = dr.select((si.t < mei.t) | (mei.t > remaining_dist), tr_eval,
                                        tr_eval * mei.combined_extinction)
            tr_pdf = dr.select(channel == 0, free_flight_pdf[0], dr.select(channel == 1, free_flight_pdf[1],
                                                                           dr.select(channel == 2, free_flight_pdf[2],
                                                                                     free_flight_pdf[3])))

            tr_weight = dr.select(tr_pdf > 0.0, tr_eval / tr_pdf, 0.0)
            transmittance = dr.select(is_spectral, transmittance * tr_weight, transmittance)

            passed_light = active_medium & (mei.t > remaining_dist)
            total_dist = dr.select(passed_light & mei.is_valid(), max_dist, total_dist)
            mei.t = dr.select(passed_light, dr.inf, mei.t)

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()
            is_spectral &= active_medium
            not_spectral &= active_medium

            total_dist = dr.select(active_medium, total_dist + mei.t, total_dist)

            ray.o = dr.select(active_medium, mei.p, ray.o)
            si.t = dr.select(active_medium, si.t - mei.t, si.t)

            transmittance = dr.select(is_spectral, transmittance * mi.Spectrum(mei.sigma_n), transmittance)
            transmittance = dr.select(not_spectral, transmittance * (mei.sigma_n / mei.combined_extinction),
                                      transmittance)

            # --- 2. THE SURFACE WALKER ---
            intersect = active_surface & needs_intersection
            si = dr.select(intersect, scene.ray_intersect(ray, active=intersect), si)
            needs_intersection &= ~intersect

            active_surface |= escaped_medium
            total_dist = dr.select(active_surface, total_dist + si.t, total_dist)

            active_surface &= si.is_valid() & active & ~active_medium

            bsdf_shadow = si.bsdf(ray)
            null_trans = bsdf_shadow.eval_null_transmission(si, active_surface)
            null_trans = si.to_world_mueller(null_trans, si.wi, si.wi)
            transmittance = dr.select(active_surface, transmittance * null_trans, transmittance)

            ray = dr.select(active_surface, si.spawn_ray(ray.d), ray)
            ray.maxt = remaining_dist
            needs_intersection |= active_surface

            trans_max = dr.max(transmittance)
            active &= (active_medium | active_surface) & (trans_max > 0.0)

            has_medium_trans = active_surface & si.is_medium_transition()
            medium = dr.select(has_medium_trans, si.target_medium(ray.d), medium)

            return (active, ray, total_dist, needs_intersection, si, transmittance, medium, sampler)

        active, ray, total_dist, needs_intersection, si, transmittance, medium, sampler = dr.while_loop(
            state=loop_state, cond=cond, body=body, max_iterations=self.max_depth * 2
        )

        return transmittance

    def evaluate_nee_transmittance(self, scene: mi.Scene, sampler: mi.Sampler, ray_in: mi.Ray3f,
                                   medium_in: mi.Medium, channel: mi.UInt32,
                                   active_in: mi.Bool, max_dist: mi.Float) -> mi.Spectrum:

        active = mi.Bool(active_in)
        ray = mi.Ray3f(ray_in)
        total_dist = dr.zeros(mi.Float, dr.width(active))

        has_medium = (medium_in != None)
        medium = dr.select(has_medium, medium_in, self._null_medium)
        transmittance = dr.full(mi.Spectrum, 1.0, dr.width(active))

        loop_state = (sampler, active, ray, total_dist, medium, transmittance)

        def loop_cond(sampler, active, ray, total_dist, medium, transmittance):
            return active

        def loop_body(sampler, active, ray, total_dist, medium, transmittance):
            remaining_dist = max_dist - total_dist
            ray.maxt = remaining_dist
            active &= (remaining_dist > 0.0)

            active_medium = active & (medium != None)
            mei = medium.sample_interaction(ray, sampler.next_1d(active_medium), channel, active_medium)
            ray.maxt = dr.select(active_medium & medium.is_homogeneous() & mei.is_valid(),
                                 dr.minimum(mei.t, remaining_dist), ray.maxt)

            si = scene.ray_intersect(ray, active)
            mei.t = dr.select(active_medium & (si.t < mei.t), dr.inf, mei.t)

            t_cap = dr.minimum(remaining_dist, dr.minimum(mei.t, si.t)) - mei.mint
            tr_eval = dr.exp(-t_cap * mei.combined_extinction)
            tr_eval = dr.select(dr.isfinite(tr_eval), tr_eval, 0.0)

            is_spectral = active_medium & medium.has_spectral_extinction()
            free_flight_pdf = dr.select((si.t < mei.t) | (mei.t > remaining_dist), tr_eval,
                                        tr_eval * mei.combined_extinction)
            tr_pdf = self._index_spectrum(free_flight_pdf, channel)
            tr_weight = dr.select(tr_pdf > 0.0, tr_eval / tr_pdf, 0.0)
            transmittance = dr.select(is_spectral, transmittance * tr_weight, transmittance)

            passed_light = active_medium & (mei.t > remaining_dist)
            total_dist = dr.select(passed_light & mei.is_valid(), max_dist, total_dist)
            mei.t = dr.select(passed_light, dr.inf, mei.t)

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()
            total_dist = dr.select(active_medium, total_dist + mei.t, total_dist)

            transmittance = dr.select(active_medium & is_spectral, transmittance * mei.sigma_n, transmittance)
            safe_ext = dr.maximum(mei.combined_extinction, 1e-8)
            transmittance = dr.select(active_medium & ~is_spectral, transmittance * (mei.sigma_n / safe_ext),
                                      transmittance)

            active_surface = (active & ~active_medium) | escaped_medium
            valid_surface = active_surface & si.is_valid()

            total_dist = dr.select(valid_surface, total_dist + si.t,
                                   dr.select(active_surface & ~valid_surface, max_dist, total_dist))

            bsdf_shadow = si.bsdf(ray)
            null_trans = bsdf_shadow.eval_null_transmission(si, valid_surface)
            transmittance = dr.select(valid_surface, transmittance * null_trans, transmittance)

            has_medium_trans = valid_surface & si.is_medium_transition()
            medium = dr.select(has_medium_trans, si.target_medium(ray.d), medium)

            ray_spawned = si.spawn_ray(ray.d)
            ray.o = dr.select(active_medium, mei.p, dr.select(valid_surface, ray_spawned.o, ray.o))
            ray.maxt = remaining_dist

            active &= (active_medium | valid_surface) & (dr.max(transmittance) > 1e-6)
            transmittance = mi.Spectrum(transmittance)

            return (sampler, active, ray, total_dist, medium, transmittance)

        sampler, active, ray, total_dist, medium, transmittance = dr.while_loop(
            state=loop_state,
            cond=loop_cond,
            body=loop_body
        )

        return transmittance

    def sample(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.RayDifferential3f,
               medium: mi.Medium = None,
               active: bool = True,
               aovs: mi.Float = None,
               ) -> Tuple[mi.Color3f, bool, List[float]]:

        bsdf_ctx = mi.BSDFContext()
        ray = mi.Ray3f(ray)
        ray_width = dr.width(ray)

        throughput_weight = dr.full(mi.Spectrum, 1.0, ray_width)
        depth = dr.zeros(mi.UInt32, ray_width)
        L = dr.zeros(mi.Spectrum, ray_width)
        ior = dr.full(mi.Float, 1.0, ray_width)
        active = mi.Bool(active)
        ray_index = dr.arange(mi.UInt32, dr.width(ray))

        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        prev_event_was_medium = mi.Bool(False)

        isFinalIter = mi.Bool(self.isFinalIter)
        if medium is None:
            dummy_si = dr.zeros(mi.SurfaceInteraction3f, ray_width)
            medium = dummy_si.target_medium(ray.d)

        mei = dr.zeros(mi.MediumInteraction3f, ray_width)

        channel = mi.UInt32(0)
        if dr.size_v(mi.Spectrum) == 3:
            n_channels = mi.UInt32(dr.size_v(mi.Spectrum))
            channel = dr.minimum(sampler.next_1d(active) * n_channels, n_channels - 1)
            channel = mi.UInt32(channel)

        valid_ray = mi.Bool(False)

        loop_state = (sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                      prev_event_was_medium, isFinalIter, medium, mei, channel, valid_ray)

        def loop_cond(sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                      prev_event_was_medium, isFinalIter, medium, mei, channel, valid_ray):
            return active

        def loop_body(sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                      prev_event_was_medium, isFinalIter, medium, mei, channel, valid_ray):

            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=depth == 0)

            # =========================================================================
            # PHASE 2: SAMPLING THE RTE
            # =========================================================================
            active_medium = active & (medium != None)
            mei = medium.sample_interaction(ray, sampler.next_1d(active_medium), channel, active_medium)
            mei.t = dr.select(active_medium & (si.t < mei.t), dr.inf, mei.t)

            is_spectral = active_medium & medium.has_spectral_extinction()
            tr, free_flight_pdf = medium.transmittance_eval_pdf(mei, si, active_medium)
            tr_pdf = self._index_spectrum(free_flight_pdf, channel)

            tr_attenuation = dr.select(tr_pdf > 0.0, tr / tr_pdf, 0.0)
            throughput_weight = dr.select(is_spectral, throughput_weight * tr_attenuation, throughput_weight)

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()

            null_scatter_prob = dr.mean(mei.sigma_n / mei.combined_extinction)
            null_scatter = sampler.next_1d(active_medium) < null_scatter_prob

            act_null_scatter = null_scatter & active_medium
            act_medium_scatter = ~act_null_scatter & active_medium

            throughput_weight = dr.select(is_spectral & act_null_scatter,
                                          throughput_weight * (mei.sigma_n / null_scatter_prob), throughput_weight)
            ray.o = dr.select(act_null_scatter, mei.p, ray.o)
            si.t = dr.select(act_null_scatter, si.t - mei.t, si.t)

            spectral_scatter_atten = mei.sigma_s / dr.mean(mei.sigma_t / mei.combined_extinction)
            throughput_weight = dr.select(is_spectral & act_medium_scatter, throughput_weight * spectral_scatter_atten,
                                          throughput_weight)

            not_spectral = active_medium & ~is_spectral
            throughput_weight = dr.select(not_spectral & act_medium_scatter,
                                          throughput_weight * (mei.sigma_s / mei.sigma_t), throughput_weight)

            active_surface = (active & (medium == None)) | escaped_medium

            # =========================================================================
            # PHASE 3: NEE (Standard Path Tracing MIS)
            # =========================================================================
            bsdf = si.bsdf()
            phase = medium.phase_function() if medium is not None else None
            phase_ctx = mi.PhaseFunctionContext(sampler)
            bsdf_ctx = mi.BSDFContext()

            ds_direct = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            emitter_pdf = scene.pdf_emitter_direction(prev_si, ds_direct, ~prev_bsdf_delta)
            mis = dr.select(prev_bsdf_delta, 1.0, mis_weight(prev_bsdf_pdf, emitter_pdf))
            em_radiance = ds_direct.emitter.eval(si)
            Le = dr.select(active_surface, throughput_weight * mis * em_radiance, mi.Spectrum(0.0))

            active_next = (depth + 1 < self.max_depth) & (si.is_valid() | act_medium_scatter)

            active_em_surface = active_next & active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            active_em_medium = active_next & act_medium_scatter

            ds_surface, em_weight_surface = scene.sample_emitter_direction(si, sampler.next_2d(), test_visibility=False,
                                                                           active=active_em_surface)
            ds_medium, em_weight_medium = scene.sample_emitter_direction(mei, sampler.next_2d(), test_visibility=False,
                                                                         active=active_em_medium)

            ds = dr.select(act_medium_scatter, ds_medium, ds_surface)
            em_weight = dr.select(act_medium_scatter, em_weight_medium, em_weight_surface)

            active_em = (active_em_surface | active_em_medium) & (ds.pdf > 0.0)

            shadow_ray = dr.select(act_medium_scatter, mei.spawn_ray_to(ds.p), si.spawn_ray_to(ds.p))
            shadow_medium = dr.select(act_medium_scatter, medium,
                                      dr.select(si.is_medium_transition(), si.target_medium(shadow_ray.d), medium))

            Tr = self.evaluate_nee_transmittance(scene, sampler, shadow_ray, shadow_medium, channel, active_em,
                                                 shadow_ray.maxt)
            em_weight *= Tr
            active_em &= (dr.max(Tr) > 0.0)

            wo_local_surface = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo_local_surface, active_em & active_surface)

            if phase is not None:
                phase_value_em, phase_pdf_em = phase.eval_pdf(phase_ctx, mei, ds.d, active_em & act_medium_scatter)
            else:
                phase_value_em, phase_pdf_em = mi.Spectrum(0), mi.Float(0)

            scatter_value_em = dr.select(act_medium_scatter, phase_value_em, bsdf_value_em)
            scatter_pdf_em = dr.select(act_medium_scatter, phase_pdf_em, bsdf_pdf_em)

            # Standard Path Tracer NEE MIS
            mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, scatter_pdf_em))
            Lr_dir = throughput_weight * mis_em * scatter_value_em * em_weight

            L = mi.Spectrum(L + Le + dr.select(dr.isnan(Lr_dir), mi.Spectrum(0.0), Lr_dir))

            # =========================================================================
            # PHASE 4: SCATTER & SHM CAUSTICS
            # =========================================================================
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active_surface),
                                                   sampler.next_2d(active_surface), active_surface)

            if phase is not None:
                phase_wo_world, phase_weight, phase_pdf = phase.sample(phase_ctx, mei,
                                                                       sampler.next_1d(act_medium_scatter),
                                                                       sampler.next_2d(act_medium_scatter),
                                                                       act_medium_scatter)
            else:
                phase_wo_world, phase_weight, phase_pdf = mi.Vector3f(0), mi.Spectrum(0), mi.Float(0)

            wo_world = dr.select(act_medium_scatter, phase_wo_world, si.to_world(bsdf_sample.wo))

            bounce_weight = dr.select(act_medium_scatter, phase_weight, bsdf_weight)

            delta = active_surface & mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            non_null_bsdf = active_surface & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
            valid_bounce = act_medium_scatter | non_null_bsdf

            throughput_weight *= dr.select(valid_bounce, bounce_weight, 1.0)
            final_pdf = dr.select(act_medium_scatter, phase_pdf, bsdf_sample.pdf)

            # --- Inject SHM Caustics ---
            is_diffuse = active_surface & ~delta

            # The density gather
            search_radius = mi.Float(0.05)
            caustic_irradiance = dr.zeros(mi.Spectrum, dr.width(si.p))

            if self.photon_map is not None:
                caustic_irradiance = self.photon_map.evaluate_surface_caustics(si, search_radius, is_diffuse)

            albedo = bsdf.eval(bsdf_ctx, si, si.wi, is_diffuse)
            L_caustics = throughput_weight * albedo * caustic_irradiance
            L += dr.select(dr.isnan(L_caustics), mi.Spectrum(0.0), L_caustics)

            # =========================================================================
            # PHASE 5: UPDATE & RUSSIAN ROULETTE
            # =========================================================================
            ray_surface = si.spawn_ray(wo_world)
            ray_medium = mei.spawn_ray(wo_world)

            scatter_event = act_medium_scatter | active_surface

            ray.o = dr.select(scatter_event, dr.select(act_medium_scatter, ray_medium.o, ray_surface.o), ray.o)
            ray.d = dr.select(scatter_event, wo_world, ray.d)

            ior *= dr.select(active_surface, bsdf_sample.eta, 1.0)

            has_medium_trans = active_surface & si.is_medium_transition()
            medium = dr.select(has_medium_trans, si.target_medium(ray.d), medium)

            mei_si = dr.zeros(mi.SurfaceInteraction3f, dr.width(active))
            mei_si.p = mei.p
            mei_si.time = mei.time
            mei_si.wi = mei.wi

            current_si = dr.select(act_medium_scatter, mei_si, si)
            prev_si = dr.select(scatter_event, current_si, prev_si)
            prev_bsdf_pdf = dr.select(scatter_event, final_pdf, prev_bsdf_pdf)
            prev_bsdf_delta = dr.select(scatter_event, delta, prev_bsdf_delta)
            prev_event_was_medium = dr.select(scatter_event, act_medium_scatter, prev_event_was_medium)

            throughput_weight_max = dr.max(throughput_weight)
            active_next &= (throughput_weight_max > 0.0)

            rr_prob = dr.minimum(throughput_weight_max * ior ** 2, 0.95)
            rr_prob = dr.maximum(rr_prob, 1e-6)
            rr_active = (depth >= self.rr_depth) & active_next
            throughput_weight = dr.select(rr_active, throughput_weight * dr.rcp(rr_prob), throughput_weight)
            rr_continue = sampler.next_1d(active_next) < rr_prob
            active_next &= ~rr_active | rr_continue

            active = active_next
            depth = dr.select(valid_bounce, depth + 1, depth)

            return (sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                    prev_event_was_medium, isFinalIter, medium, mei, channel, valid_ray)

        sampler, ray, depth, L, throughput_weight, ior, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta, prev_event_was_medium, isFinalIter, medium, mei, channel, valid_ray = dr.while_loop(
            state=loop_state,
            cond=loop_cond,
            body=loop_body,
            max_iterations=self.max_depth
        )

        dr.schedule(L)
        sampler.schedule_state()

        spp_per_pass = sampler.sample_count()
        if spp_per_pass == 1:
            self.sumL += L
            self.sumL2 += L * L
        else:
            total_size = dr.width(L)
            one_sample_size = int(total_size / spp_per_pass)
            base_index = dr.arange(mi.UInt32, one_sample_size) * spp_per_pass
            for i in range(spp_per_pass):
                index = base_index + i
                one_spp_L = dr.gather(mi.Spectrum, L, index)
                self.sumL += one_spp_L
                self.sumL2 += one_spp_L * one_spp_L

        dr.schedule(self.sumL, self.sumL2)
        return (L, depth != 0, [1])

    def computeMSE(self, spp: float, groundTruth: mi.Color3f) -> float:
        L = self.sumL / spp
        L_scalar = sum(L) / len(L)
        gt_scalar = mi.luminance(groundTruth)
        mse = (L_scalar - gt_scalar) ** 2
        mse = dr.minimum(mse, 10000.0)
        mse = dr.mean(mse)[0]
        return mse

    def computeVariance(self, spp: float, groundTruth: mi.Color3f = None) -> float:
        if groundTruth is not None:
            L2 = self.sumL2 / spp
            L2_scalar = sum(L2) / len(L2)
            gt_scalar = mi.luminance(groundTruth)
            variance = L2_scalar - (gt_scalar * gt_scalar)
            variance = dr.minimum(variance, 10000.0)
            variance = dr.mean(variance)[0]
            variance /= spp
        else:
            L = self.sumL / spp
            L2 = self.sumL2 / spp
            L_scalar = sum(L) / len(L)
            L2_scalar = sum(L2) / len(L2)
            variance = L2_scalar - (L_scalar * L_scalar)
            variance = dr.minimum(variance, 10000.0)
            variance = dr.mean(variance)[0]
            if spp > 1:
                variance /= spp - 1
        return variance

    def aov_names(self):
        return ["depth.Y"]

    def to_string(self):
        return "vmf_path_guiding_integrator"


mi.register_integrator('vmf_path_guiding_integrator', lambda props: vMF_PathGuidingIntegrator(props))
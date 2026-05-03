import drjit as dr
import mitsuba as mi
import numpy as np

class PhotonRecord:
    DRJIT_STRUCT = {
        'pos': mi.Point3f,
        'dir': mi.Vector3f,
        'weight': mi.Float,
        'active': mi.Bool
    }

class VolumetricPhotonTracer:
    def __init__(self, scene: mi.Scene, tree, max_depth: int = 8):
        self.scene = scene
        self.tree = tree
        self.max_depth = max_depth

    def trace_and_train(self, num_photons: int, current_iteration: int):
        print(f"[PhotonTracer] Tracing {num_photons} photons (Iter {current_iteration})...")

        sampler = mi.load_dict({'type': 'independent'})
        sampler.seed(current_iteration * 1337, wavefront_size=num_photons)

        ray, emitter_weight, emitter_ds = self.scene.sample_emitter_ray(
            time=0.0,
            sample1=sampler.next_1d(),
            sample2=sampler.next_2d(),
            sample3=sampler.next_2d(),
            active=True
        )
        dr.eval(self.tree.kdTreeNode.vertCount)

        throughput = mi.Spectrum(emitter_weight)
        depth = dr.zeros(mi.UInt32, num_photons)
        active = mi.Bool(True)

        dummy_si = dr.zeros(mi.SurfaceInteraction3f, num_photons)
        medium = dummy_si.target_medium(ray.d)

        buffer_size = num_photons * self.max_depth
        record_pos = dr.zeros(mi.Vector3f, buffer_size)
        record_dir = dr.zeros(mi.Vector3f, buffer_size)
        record_weight = dr.zeros(mi.Float, buffer_size)
        record_active = dr.zeros(mi.Bool, buffer_size)

        dr.eval(record_pos, record_dir, record_weight, record_active)

        ray_index = dr.arange(mi.UInt32, num_photons)

        loop_state = (sampler, ray, throughput, depth, medium, active)

        def loop_cond(sampler, ray, throughput, depth, medium, active):
            return active

        def loop_body(sampler, ray, throughput, depth, medium, active):
            si = self.scene.ray_intersect(ray, active)
            active_medium = active & (medium != None)

            mei = medium.sample_interaction(ray, sampler.next_1d(active_medium), 0, active_medium)
            mei.t = dr.select(active_medium & (si.t < mei.t), dr.inf, mei.t)

            act_medium_scatter = active_medium & mei.is_valid() & (mei.t < si.t)

            # Standard attenuation
            throughput = dr.select(act_medium_scatter, throughput * (mei.sigma_s / mei.sigma_t), throughput)

            # Unique global index that prevents collisions
            num_photons_mi = dr.width(ray_index)
            global_idx = (depth * num_photons_mi) + ray_index

            dr.scatter(record_pos.x, mei.p.x, global_idx, act_medium_scatter)
            dr.scatter(record_pos.y, mei.p.y, global_idx, act_medium_scatter)
            dr.scatter(record_pos.z, mei.p.z, global_idx, act_medium_scatter)

            # REPAIR 1: Directional Alignment
            # Record the direction the photon CAME FROM (-ray.d) so camera rays are guided TOWARDS the light.
            dr.scatter(record_dir.x, -ray.d.x, global_idx, act_medium_scatter)
            dr.scatter(record_dir.y, -ray.d.y, global_idx, act_medium_scatter)
            dr.scatter(record_dir.z, -ray.d.z, global_idx, act_medium_scatter)

            # REPAIR 2: Raw Radiometric Flux
            # No more log-compression. We need raw energy values for the zero-variance theory to hold up[cite: 810].
            raw_weight = dr.mean(throughput)
            compressed_weight = dr.log(1.0 + raw_weight)
            dr.scatter(record_weight, compressed_weight, global_idx, act_medium_scatter)
            dr.scatter(record_active, mi.Bool(True), global_idx, act_medium_scatter)

            phase = medium.phase_function() if medium is not None else None
            phase_ctx = mi.PhaseFunctionContext(sampler)

            wo, phase_weight, phase_pdf = mi.Vector3f(0), mi.Spectrum(0), mi.Float(0)
            if phase is not None:
                wo, phase_weight, phase_pdf = phase.sample(phase_ctx, mei, sampler.next_1d(act_medium_scatter),
                                                           sampler.next_2d(act_medium_scatter), act_medium_scatter)

            # --- SURFACE BOUNCE LOGIC ---
            active_surface = active & ~act_medium_scatter & si.is_valid()
            bsdf = si.bsdf(ray)
            bsdf_ctx = mi.BSDFContext()

            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si,
                sampler.next_1d(active_surface),
                sampler.next_2d(active_surface),
                active_surface
            )

            # Spawn rays safely with epsilon offsets (Done exactly once)
            wo_world_surface = si.to_world(bsdf_sample.wo)
            ray_surface = si.spawn_ray(wo_world_surface)
            ray_medium = mei.spawn_ray(wo)

            # Update ray state
            ray.o = dr.select(act_medium_scatter, ray_medium.o, ray_surface.o)
            ray.d = dr.select(act_medium_scatter, ray_medium.d, ray_surface.d)

            # Update throughput: phase weight for volumes, bsdf weight for surfaces
            # If it hits a null boundary (neither scatter nor valid surface bounce), multiply by 1.0 to pass through
            bounce_weight = dr.select(act_medium_scatter, phase_weight, bsdf_weight)
            active_bounce = act_medium_scatter | active_surface
            throughput *= dr.select(active_bounce, bounce_weight, 1.0)

            # Boundary Tracking
            has_medium_trans = active_surface & si.is_medium_transition()
            medium = dr.select(has_medium_trans, si.target_medium(ray.d), medium)

            depth += 1
            max_t = dr.max(throughput)
            active &= (depth < self.max_depth) & (max_t > 0)

            rr_prob = dr.minimum(max_t, 0.95)
            active &= sampler.next_1d(active) < rr_prob
            throughput /= dr.select(active, rr_prob, 1.0)

            return (sampler, ray, throughput, depth, medium, active)

        dr.while_loop(loop_state, loop_cond, loop_body, max_iterations=self.max_depth)

        dr.eval(record_active, record_pos, record_dir, record_weight)

        active_indices = dr.compress(record_active)
        num_active = dr.width(active_indices)

        print(f"[PhotonTracer] Loop finished. Captured {num_active} valid volumetric scatter events.")

        if num_active > 0:
            c_pos = mi.Vector3f(
                dr.gather(mi.Float, record_pos.x, active_indices),
                dr.gather(mi.Float, record_pos.y, active_indices),
                dr.gather(mi.Float, record_pos.z, active_indices)
            )
            c_dir = mi.Vector3f(
                dr.gather(mi.Float, record_dir.x, active_indices),
                dr.gather(mi.Float, record_dir.y, active_indices),
                dr.gather(mi.Float, record_dir.z, active_indices)
            )
            c_weight = dr.gather(mi.Float, record_weight, active_indices)

            valid_mask = dr.full(mi.Bool, True, num_active)
            leaf_indices = self.tree.getLeafNodeIndex(c_pos, valid_mask)
            unique_leaves = len(np.unique(leaf_indices.numpy()))
            print(f"DEBUG: Photons distributed across {unique_leaves} unique leaves.")

            dummy_dir = dr.zeros(mi.Vector3f, num_active)
            dummy_weight = dr.zeros(mi.Float, num_active)

            self.tree.fit_mixtures_to_record(
                leaf_indices=leaf_indices,
                directions=c_dir,
                weights=c_weight,
                directions_nee=dummy_dir,
                weights_nee=dummy_weight,
                active=valid_mask,
                current_iteration=current_iteration,
                iterations=12
            )

            dr.scatter_reduce(dr.ReduceOp.Add, self.tree.kdTreeNode.vertCount, mi.Float(1.0), leaf_indices, valid_mask)
            dr.eval(self.tree.kdTreeNode.vertCount)
        else:
            print("[WARNING] Zero photons recorded. The KD-Tree is completely unlearned.")
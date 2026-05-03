import drjit as dr
import mitsuba as mi


class vMFMixture:
    def __init__(self, weights: list[mi.Float], mus: list[mi.Vector3f], kappas: list[mi.Float]):
        self.K = len(weights)
        self.weights = weights
        self.mus = mus
        self.kappas = kappas

    def pdf(self, omega: mi.Vector3f) -> mi.Float:
        total_pdf = mi.Float(0.0)
        for k in range(self.K):
            safe_k = dr.maximum(self.kappas[k], 1e-4)
            cos_theta = dr.dot(self.mus[k], omega)

            norm_factor = safe_k / (dr.two_pi * (1.0 - dr.exp(-2.0 * safe_k)))
            exp_term = dr.exp(safe_k * (cos_theta - 1.0))
            lobe_pdf = dr.select(self.kappas[k] < 1e-4, dr.inv_four_pi, norm_factor * exp_term)
            total_pdf += self.weights[k] * lobe_pdf
        return total_pdf

    def sample(self, sample3: mi.Point3f) -> tuple[mi.Vector3f, mi.Float]:
        xi_lobe = sample3.x
        xi_dir = mi.Point2f(sample3.y, sample3.z)

        sampled_dirs = []
        for k in range(self.K):
            safe_k = dr.maximum(self.kappas[k], 1e-4)
            W = 1.0 + (1.0 / safe_k) * dr.log(xi_dir.x + (1.0 - xi_dir.x) * dr.exp(-2.0 * safe_k))
            W = dr.select(self.kappas[k] < 1e-4, 1.0 - 2.0 * xi_dir.x, W)

            sin_theta = dr.safe_sqrt(1.0 - W * W)
            phi = dr.two_pi * xi_dir.y
            s, c = dr.sincos(phi)

            local_dir = mi.Vector3f(sin_theta * c, sin_theta * s, W)
            frame = mi.Frame3f(self.mus[k])
            world_dir = frame.to_world(local_dir)
            sampled_dirs.append(world_dir)

        final_dir = mi.Vector3f(0.0, 0.0, 0.0)
        cdf = mi.Float(0.0)

        for k in range(self.K):
            next_cdf = cdf + self.weights[k]
            if k == self.K - 1:
                is_selected = (xi_lobe >= cdf)
            else:
                is_selected = (xi_lobe >= cdf) & (xi_lobe < next_cdf)

            final_dir = dr.select(is_selected, sampled_dirs[k], final_dir)
            cdf = next_cdf

        return final_dir, self.pdf(final_dir)

    @classmethod
    def fit_em(cls, directions: mi.Vector3f, rad_weights: mi.Float, K: int = 3, iterations: int = 12) -> 'vMFMixture':
        initial_dirs = [
            mi.Vector3f(1, 1, 1), mi.Vector3f(1, 1, -1), mi.Vector3f(1, -1, 1), mi.Vector3f(1, -1, -1),
            mi.Vector3f(-1, 1, 1), mi.Vector3f(-1, 1, -1), mi.Vector3f(-1, -1, 1), mi.Vector3f(-1, -1, -1)
        ]
        mus = [dr.normalize(initial_dirs[k % 8]) for k in range(K)]
        weights = [mi.Float(1.0 / K) for _ in range(K)]
        kappas = [mi.Float(5.0) for _ in range(K)]

        active_data = rad_weights > 1e-6
        total_rad_weight = dr.sum(dr.select(active_data, rad_weights, 0.0))

        # Calculate raw photon count for the MAP prior
        N_samples = dr.sum(dr.select(active_data, 1.0, 0.0))
        scale_factor = dr.select(total_rad_weight > 1e-6, N_samples / total_rad_weight, 0.0)
        norm_rad_weights = rad_weights * scale_factor

        for _ in range(iterations):
            # --- E-STEP ---
            lobe_pdfs = []
            for k in range(K):
                safe_k = dr.maximum(kappas[k], 1e-4)
                is_unif = kappas[k] < 1e-4
                cos_t = dr.dot(mus[k], directions)
                norm = safe_k / (dr.two_pi * (1.0 - dr.exp(-2.0 * safe_k)))
                val = dr.select(is_unif, dr.inv_four_pi, norm * dr.exp(safe_k * (cos_t - 1.0)))
                lobe_pdfs.append(weights[k] * val)

            sum_pdfs = lobe_pdfs[0]
            for k in range(1, K):
                sum_pdfs += lobe_pdfs[k]

            responsibilities = [
                dr.select(sum_pdfs > 1e-8, lobe_pdfs[k] / sum_pdfs, 1.0 / K)
                for k in range(K)
            ]

            # --- M-STEP ---
            for k in range(K):
                # Apply the statistically normalized weights
                eff_weight = responsibilities[k] * norm_rad_weights
                sum_eff_weight = dr.sum(dr.select(active_data, eff_weight, 0.0))

                # Weight update uses actual photon count N
                weights[k] = dr.select(N_samples > 0, sum_eff_weight / N_samples, 1.0 / K)

                R_x = dr.sum(dr.select(active_data, directions.x * eff_weight, 0.0))
                R_y = dr.sum(dr.select(active_data, directions.y * eff_weight, 0.0))
                R_z = dr.sum(dr.select(active_data, directions.z * eff_weight, 0.0))
                R = mi.Vector3f(R_x, R_y, R_z)

                # Inject MAP-EM Prior (alpha=10.0, beta=0.8)
                alpha_prior, beta_prior = 10.0, 0.8
                r_bar_λ = (alpha_prior * beta_prior + dr.norm(R)) / (alpha_prior + sum_eff_weight)

                r_safe = dr.minimum(r_bar_λ, 0.9999)
                r2 = r_safe * r_safe

                mus[k] = dr.normalize(dr.select(dr.norm(R) > 1e-6, R, mus[k]))
                kappas[k] = dr.select(sum_eff_weight > 1e-6, (r_safe * (3.0 - r2)) / (1.0 - r2), 0.0)

            # Prevent the graph from blowing up memory
            dr.eval(*weights, *mus, *kappas)

        total_w = sum(weights)
        weights = [dr.select(total_w > 0, w / total_w, 1.0 / K) for w in weights]

        return cls(weights, mus, kappas)
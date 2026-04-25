import drjit as dr
import mitsuba as mi


class vMFMixture:
    def __init__(self, weights: list[mi.Float], mus: list[mi.Vector3f], kappas: list[mi.Float]):
        self.K = len(weights)
        self.weights = weights
        self.mus = mus
        self.kappas = kappas

    def pdf(self, omega: mi.Vector3f) -> mi.Float:
        """
        Evaluates the combined PDF of the mixture for a given direction.
        Uses numerically stable evaluation to prevent Inf/NaN blowouts at high kappas.
        """
        total_pdf = mi.Float(0.0)
        for k in range(self.K):
            is_uniform = self.kappas[k] < 1e-4
            cos_theta = dr.dot(self.mus[k], omega)

            norm_factor = self.kappas[k] / (dr.two_pi * (1.0 - dr.exp(-2.0 * self.kappas[k])))
            exp_term = dr.exp(self.kappas[k] * (cos_theta - 1.0))

            lobe_pdf = dr.select(is_uniform, dr.inv_four_pi, norm_factor * exp_term)
            total_pdf += self.weights[k] * lobe_pdf

        return total_pdf

    def sample(self, sample3: mi.Point3f) -> tuple[mi.Vector3f, mi.Float]:
        """
        Samples a direction from the mixture.
        sample3.x picks the lobe via CDF. sample3.y and .z sample the direction.
        """
        xi_lobe = sample3.x
        xi_dir = mi.Point2f(sample3.y, sample3.z)

        sampled_dirs = []
        for k in range(self.K):
            W = 1.0 + (1.0 / self.kappas[k]) * dr.log(xi_dir.x + (1.0 - xi_dir.x) * dr.exp(-2.0 * self.kappas[k]))
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

        # Select the sampled direction based on the mixture weights
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
    def fit_em(cls, directions: mi.Vector3f, rad_weights: mi.Float, K: int = 3, iterations: int = 5) -> 'vMFMixture':
        """
        Expectation-Maximization algorithm to fit K lobes to a batch of recorded ray directions.
        rad_weights should be the collapsed scalar luminance of the incoming rays.
        """
        # Spread lobes across orthogonal axes to prevent them from merging initially
        initial_dirs = [
            mi.Vector3f(1, 0, 0),
            mi.Vector3f(0, 1, 0),
            mi.Vector3f(0, 0, 1)
        ]

        weights = [mi.Float(1.0 / K) for _ in range(K)]
        mus = [initial_dirs[k % 3] for k in range(K)]
        kappas = [mi.Float(5.0) for _ in range(K)]

        active_data = rad_weights > 1e-6
        total_rad_weight = dr.sum(dr.select(active_data, rad_weights, 0.0))

        for _ in range(iterations):
            # --- E-STEP ---
            lobe_pdfs = []
            for k in range(K):
                is_unif = kappas[k] < 1e-4
                cos_t = dr.dot(mus[k], directions)
                norm = kappas[k] / (dr.two_pi * (1.0 - dr.exp(-2.0 * kappas[k])))
                val = dr.select(is_unif, dr.inv_four_pi, norm * dr.exp(kappas[k] * (cos_t - 1.0)))
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
                eff_weight = responsibilities[k] * rad_weights
                sum_eff_weight = dr.sum(dr.select(active_data, eff_weight, 0.0))

                weights[k] = dr.select(total_rad_weight > 0, sum_eff_weight / total_rad_weight, 1.0 / K)

                R_x = dr.sum(dr.select(active_data, directions.x * eff_weight, 0.0))
                R_y = dr.sum(dr.select(active_data, directions.y * eff_weight, 0.0))
                R_z = dr.sum(dr.select(active_data, directions.z * eff_weight, 0.0))
                R = mi.Vector3f(R_x, R_y, R_z)

                r_vec = dr.select(sum_eff_weight > 1e-6, R / sum_eff_weight, mus[k])
                r_bar = dr.norm(r_vec)

                mus[k] = dr.select(r_bar > 1e-6, r_vec / r_bar, mus[k])

                r_safe = dr.minimum(r_bar, 0.9999)
                r2 = r_safe * r_safe
                kappas[k] = dr.select(sum_eff_weight > 1e-6, (r_safe * (3.0 - r2)) / (1.0 - r2), 0.0)

        # Force normalization at the end to prevent LLVM folding drift
        total_w = sum(weights)
        weights = [dr.select(total_w > 0, w / total_w, 1.0 / K) for w in weights]

        return cls(weights, mus, kappas)
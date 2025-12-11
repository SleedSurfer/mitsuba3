import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant('llvm_ad_spectral')

def test_eval_vs_sample(phase_props, num_samples=1000):
    phase_fn = mi.load_dict({
        'type': 'atmosphericphase',
        **phase_props
    })

    key = mi.PhaseFunctionContext()
    W = mi.Float(550.0)

    mi_rec = mi.MediumInteraction3f()
    mi_rec.wi = mi.Vector3f(0, 0, 1)
    mi_rec.wavelengths = W

    rng = np.random.RandomState(1)
    ratios = []

    for _ in range(num_samples):
        sample2 = mi.Point2f(rng.rand(), rng.rand())
        sample1 = mi.Float(0.0)

        wo, w, pdf_sample = phase_fn.sample(key, mi_rec, sample1, sample2, mi.Bool(True))
        val, pdf_eval = phase_fn.eval_pdf(key, mi_rec, wo, mi.Bool(True))

        # We expect pdf_eval to match pdf_sample up to the known constant factor issue
        # 💥 FIX: Index [0] is required to extract the scalar float from the Dr.Jit array before casting to Python float
        ratios.append(float((pdf_eval / (pdf_sample + 1e-12))[0]))

    ratios = np.array(ratios)
    print("pdf_eval / pdf_sample: mean", ratios.mean(), "std", ratios.std())

def test_phase_normalization(phase_props, w_nm=550.0, n_mu=128, n_phi=128):
    phase_fn = mi.load_dict({
        'type': 'atmosphericphase',
        **phase_props
    })

    key = mi.PhaseFunctionContext()
    W = mi.Float(w_nm)

    mi_rec = mi.MediumInteraction3f()
    mi_rec.wi = mi.Vector3f(0, 0, 1)
    mi_rec.wavelengths = W

    mu_vals = np.linspace(-1.0, 1.0, n_mu)
    d_mu   = 2.0 / (n_mu - 1)
    d_phi  = 2.0 * np.pi / n_phi

    integral = 0.0

    for mu in mu_vals:
        sin_theta = np.sqrt(max(0.0, 1.0 - mu * mu))

        for j in range(n_phi):
            phi = j * d_phi

            # Cast to Mitsuba scalar type
            x = mi.Float(sin_theta * np.cos(phi))
            y = mi.Float(sin_theta * np.sin(phi))
            z = mi.Float(mu)

            wo = mi.Vector3f(x, y, z)

            val, pdf = phase_fn.eval_pdf(key, mi_rec, wo, mi.Bool(True))

            # Take first spectral component
            val0 = float(val[0])
            integral += val0 * d_mu * d_phi

    print("Estimated ∫_S^2 p(ω) dω ≈", integral)

def test_phase_vs_isotropic_or_hg(phase_props, num_samples=1000):
    # Build your custom phase function phase_mist from properties
    phase_fn = mi.load_dict({
        'type': 'atmosphericphase',  # match plugin name
        **phase_props
    })

    # Also build an HG phase with g=0.8 or isotropic for comparison
    hg_phase = mi.load_dict({'type': 'hg', 'g': 0.8})
    iso_phase = mi.load_dict({'type': 'isotropic'})

    key = mi.PhaseFunctionContext()  # dummy context

    # Set up a dummy MediumInteraction: incoming direction wi and wavelength
    # Choose wi = +Z, and a fixed spectral triplet (center of range)

    # For now, we just pick a simple, constant Wavelength struct:
    W = mi.Float(550.0)

    mi_rec = mi.MediumInteraction3f()
    mi_rec.wi = mi.Vector3f(0, 0, 1)
    mi_rec.wavelengths = W

    # Sample directions and gather cos(theta) and weights
    rng = np.random.RandomState(0)
    cos_t_samples = []
    weights = []

    for _ in range(num_samples):
        # draw 2D samples
        sample2 = mi.Point2f(rng.rand(), rng.rand())
        sample1 = mi.Float(0.0)

        wo, w, pdf = phase_fn.sample(key, mi_rec, sample1, sample2, mi.Bool(True))
        cos_t = dr.dot(wo, mi_rec.wi)

        cos_t_samples.append(float(cos_t[0]))  # first lane
        # Optionally record w and pdf, p_eval etc.

    cos_t_samples = np.array(cos_t_samples)

    print("Mean cos(theta) ~ g_est:", cos_t_samples.mean())
    print("Should be strongly forward-peaked (close to 1).")


# Example usage:
phase_props = {
    'filename': 'cache/150um_mean_10um_std_32bins_1024ang_cornell_rainbow.bin'
}

if __name__ == "__main__":
    test_phase_vs_isotropic_or_hg(phase_props)
    test_eval_vs_sample(phase_props)
    test_phase_normalization(phase_props)
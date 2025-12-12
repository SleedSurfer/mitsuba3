import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

mi.set_variant('llvm_ad_spectral')

PLUGIN_FILENAME = Path("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/cache/150um_mean_10um_std_32bins_1024ang_cornell_rainbow.bin")

def make_wavelength_packet(nm: float):
    # Use the same approach as other tests in the repo: create a Spectrum
    # from a single wavelength value. Some Mitsuba builds expose `Spectrum`
    # rather than a `Wavelength` constructor.
    return mi.Spectrum(nm)

def audit_sampling(num_samples=2_000_00, bins=200):
    phase_func = mi.load_dict({
        'type': 'atmosphericphase',
        'filename': str(PLUGIN_FILENAME),
    })

    ctx = mi.PhaseFunctionContext(None)
    mei = mi.MediumInteraction3f()
    mei.wi = mi.Vector3f(0, 0, 1)
    mei.wavelengths = make_wavelength_packet(550.0)

    # Random samples
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(0, num_samples)

    s1 = sampler.next_1d()
    s2 = sampler.next_2d()

    wo, weight, pdf = phase_func.sample(ctx, mei, s1, s2)

    cos_theta = dr.dot(wo, mei.wi)
    value, pdf2 = phase_func.eval_pdf(ctx, mei, wo)

    # Consistency check: pdf returned by sample vs recomputed should match
    pdf_np = pdf.numpy()
    pdf2_np = pdf2.numpy()
    print(f"[audit_sampling] pdf consistency: max |pdf-pdf2| = {np.max(np.abs(pdf_np - pdf2_np)):.3e}")

    # Histogram of sampled mu
    mu = cos_theta.numpy()
    hist, edges = np.histogram(mu, bins=bins, range=(-1, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Compare sampled density in mu-space against analytic p_mu derived from pdf_omega:
    # pdf_omega = p_mu / (2π) => p_mu ≈ pdf_omega * 2π
    # We'll evaluate pdf_omega at those centers by calling eval_pdf with wo constructed from mu.
    mu_dr = mi.Float(centers)
    wo_test = mi.Vector3f(dr.sqrt(dr.maximum(0.0, 1 - mu_dr*mu_dr)), 0.0, mu_dr)
    v_test, pdf_test = phase_func.eval_pdf(ctx, mei, wo_test)
    p_mu = (pdf_test * (2.0 * dr.pi)).numpy()

    # Weight tail stats (lane 0)
    w0 = (value[0] / pdf2).numpy()
    q = np.quantile(w0[np.isfinite(w0)], [0.5, 0.9, 0.99, 0.999])
    print(f"[audit_sampling] weight quantiles (lane0): 50%={q[0]:.3e}, 90%={q[1]:.3e}, 99%={q[2]:.3e}, 99.9%={q[3]:.3e}")
    print(f"[audit_sampling] weight max (finite): {np.max(w0[np.isfinite(w0)]):.3e}")

    plt.figure(figsize=(12, 9))

    plt.subplot(2, 1, 1)
    plt.title("Sampling in µ: histogram vs p_mu (from eval_pdf)")
    plt.plot(centers, hist, label="hist (density in µ)", color="black")
    plt.plot(centers, p_mu, label="p_mu = pdf_omega*2π", color="red", linestyle="--")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Weight distribution (lane 0): log histogram")
    w0f = w0[np.isfinite(w0)]
    plt.hist(np.log10(w0f + 1e-30), bins=200, color="blue", alpha=0.7)
    plt.xlabel("log10(value/pdf)")
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path("audit_sampling.png")
    plt.savefig(out)
    print(f"Saved {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    audit_sampling()
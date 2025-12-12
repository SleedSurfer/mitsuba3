import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from tools.atmo_phase.lut import read_atmphase_file, lookup_phase_reference
except Exception:
    # Fallback when running from different cwd / import root used in this repo
    from python.atmospheric.tests.tools.atmo_phase.lut import read_atmphase_file, lookup_phase_reference

mi.set_variant('llvm_ad_spectral')  # keep it simple for auditing

PLUGIN_FILENAME = Path("/home/speedlord/bachelors/mitsuba3/src/tabphase_atmo/python/cache/150um_mean_10um_std_32bins_1024ang_cornell_rainbow.bin")


def make_wavelength_packet(nm: float):
    # Typical spectral lane count is 4; replicate same nm to all lanes.
    # This avoids ambiguity in eval path: all lanes see same wavelength.
    # Some Mitsuba builds expose `Spectrum` instead of `Wavelength`.
    return mi.Spectrum(nm)


def audit_eval_pdf():
    phase_table, num_angles, num_wl, min_wl, max_wl = read_atmphase_file(PLUGIN_FILENAME)

    phase_func = mi.load_dict({
        'type': 'atmosphericphase',
        'filename': str(PLUGIN_FILENAME),
    })

    num_points = 2000
    theta = dr.linspace(mi.Float, 0, dr.pi, num_points)

    wi = mi.Vector3f(0, 0, 1)
    wo = mi.Vector3f(dr.sin(theta), dr.zeros(mi.Float, num_points), dr.cos(theta))

    ctx = mi.PhaseFunctionContext(None)
    mei = mi.MediumInteraction3f()
    mei.wi = wi

    # Choose one wavelength to test
    test_nm = 550.0
    mei.wavelengths = make_wavelength_packet(test_nm)

    value, pdf = phase_func.eval_pdf(ctx, mei, wo)

    # Extract lane 0 of value for plotting/comparison
    value0 = value[0]

    # Python reference evaluation (same test wavelength)
    cos_theta = dr.dot(wo, wi).numpy()
    ref = lookup_phase_reference(
        phase_table, cos_theta=cos_theta,
        wavelength_nm=np.full_like(cos_theta, test_nm),
        min_wl=min_wl, max_wl=max_wl
    )

    # Convert to numpy
    deg = np.rad2deg(theta.numpy())
    y_val = value0.numpy()
    y_pdf = pdf.numpy()

    # Compare value vs reference
    abs_err = np.abs(y_val - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-20)

    print(f"[audit_eval_pdf] value vs reference: "
          f"abs_err max={abs_err.max():.3e}, mean={abs_err.mean():.3e} | "
          f"rel_err max={rel_err.max():.3e}, mean={rel_err.mean():.3e}")

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.title(f"Eval vs Reference @ {test_nm} nm")
    plt.plot(deg, ref, label="Reference LUT", color="green", linewidth=2)
    plt.plot(deg, y_val, label="Plugin eval", color="black", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title("Plugin pdf")
    plt.plot(deg, y_pdf, label="pdf", color="red")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("Relative error |eval-reference|/|reference|")
    plt.plot(deg, rel_err, color="purple")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    out = Path("audit_eval_pdf.png")
    plt.savefig(out)
    print(f"Saved {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    audit_eval_pdf()
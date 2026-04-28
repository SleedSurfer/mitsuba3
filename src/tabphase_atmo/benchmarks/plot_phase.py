import struct
import numpy as np
import matplotlib.pyplot as plt
import sys

# ==========================================
# 🧪 TWEAK THESE FOR YOUR SPECIFIC MIE PEAK
# ==========================================
G_PARAM = 0.985  # How sharp the HG forward peak is (0.0 to 0.999)
ALPHA_WEIGHT = 0.85  # How much of the rays use HG vs Uniform (0.85 = 85% HG, 15% Uniform)


# ==========================================

def hg_pdf(theta, g):
    """Analytical Henyey-Greenstein Directional PDF"""
    cos_theta = np.cos(theta)
    denom = (1.0 + g ** 2 - 2.0 * g * cos_theta) ** 1.5
    return (1.0 - g ** 2) / (4.0 * np.pi * denom)


def mix_pdf(theta, g, alpha):
    """The new Steering Wheel: HG + Uniform Mixture"""
    uniform_pdf = 1.0 / (4.0 * np.pi)
    return alpha * hg_pdf(theta, g) + (1.0 - alpha) * uniform_pdf


def analyze_lut_vs_mixture(filepath):
    print(f"🔬 Loading LUT: {filepath}")
    print(f"🧪 Testing Mixture PDF: g={G_PARAM}, alpha={ALPHA_WEIGHT}")

    with open(filepath, 'rb') as f:
        magic = f.read(8)
        if magic != b'ATMPHASE':
            raise ValueError("Not a valid ATMPHASE binary.")

        version = struct.unpack('I', f.read(4))[0]
        if version < 2:
            raise ValueError("Legacy V1 binary detected.")

        theta_bins, phi_bins, num_channels = struct.unpack('III', f.read(12))
        min_wvl, max_wvl = struct.unpack('ff', f.read(8))

        total_floats = theta_bins * phi_bins * num_channels
        raw_data = np.frombuffer(f.read(total_floats * 4), dtype=np.float32)
        data = raw_data.reshape((theta_bins, phi_bins, num_channels))

    c = 0  # Testing first wavelength channel

    d_theta = np.pi / (theta_bins - 1)
    d_phi = (2.0 * np.pi / (phi_bins - 1)) if phi_bins > 1 else (2.0 * np.pi)

    phi_integrals = np.zeros(theta_bins)
    for t in range(theta_bins):
        if phi_bins > 1:
            integral_phi = 0.0
            for p in range(1, phi_bins):
                v0 = data[t, p - 1, c]
                v1 = data[t, p, c]
                integral_phi += 0.5 * (v0 + v1) * d_phi
            phi_integrals[t] = integral_phi
        else:
            phi_integrals[t] = data[t, 0, c] * 2.0 * np.pi

    marg_cdf = np.zeros(theta_bins)
    integral_theta = 0.0

    for t in range(1, theta_bins):
        mu0 = np.cos((t - 1) * d_theta)
        mu1 = np.cos(t * d_theta)
        d_mu = mu0 - mu1
        integral_theta += 0.5 * (phi_integrals[t - 1] + phi_integrals[t]) * d_mu
        marg_cdf[t] = integral_theta

    norm_theta = integral_theta if integral_theta > 0.0 else 1.0
    marg_cdf /= norm_theta
    marg_cdf[-1] = 1.0

    theta_angles = np.linspace(0, np.pi, theta_bins)
    theta_degrees = np.degrees(theta_angles)

    # ---------------------------------------------------------
    # 1. The True Energy: Raw Directional PDF from LUT
    # p(omega) = phi_integral / (2 * pi * norm_theta)
    # ---------------------------------------------------------
    raw_p_omega = (phi_integrals / norm_theta) / (2.0 * np.pi)

    # ---------------------------------------------------------
    # 2. The Crime: Discrete Directional PDF (What C++ was doing)
    # p_discrete(omega) = d(CDF) / d(Solid_Angle)
    # ---------------------------------------------------------
    discrete_p_omega = np.zeros(theta_bins)
    for t in range(1, theta_bins):
        dp = marg_cdf[t] - marg_cdf[t - 1]
        # Solid angle of this theta slice = 2 * pi * (cos(t0) - cos(t1))
        d_omega = 2.0 * np.pi * (np.cos(theta_angles[t - 1]) - np.cos(theta_angles[t]))
        discrete_p_omega[t - 1] = dp / max(d_omega, 1e-12)
    discrete_p_omega[-1] = discrete_p_omega[-2]

    # ---------------------------------------------------------
    # 3. The Fix: Analytical Mixture Directional PDF
    # ---------------------------------------------------------
    mix_p_omega = mix_pdf(theta_angles, G_PARAM, ALPHA_WEIGHT)

    # --- PLOTTING ---
    plt.figure(figsize=(14, 7))

    # Let's zoom in on the first 10 degrees to see the forward peak behavior
    zoom_bins = int(theta_bins * (10.0 / 180.0))
    if zoom_bins < 5: zoom_bins = 10

    x_zoomed = theta_degrees[:zoom_bins]

    # Use logarithmic Y scale because Mie scattering spans orders of magnitude
    plt.plot(x_zoomed, raw_p_omega[:zoom_bins], 'o-', label="Raw LUT Energy (The Target)", color='blue', linewidth=2)
    plt.step(x_zoomed, discrete_p_omega[:zoom_bins], where='post', label="Old Discrete PDF (The Crime)", color='red',
             linestyle='--')
    plt.plot(x_zoomed, mix_p_omega[:zoom_bins], label="New Analytical Mixture PDF (The Fix)", color='green',
             linewidth=2.5)

    plt.yscale('log')  # Log scale exposes the variance disconnect instantly

    plt.title(f"Directional PDF Autopsy (Log Scale) | g={G_PARAM}, alpha={ALPHA_WEIGHT}")
    plt.xlabel("Theta (Degrees)")
    plt.ylabel("Probability / Energy (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    analyze_lut_vs_mixture("cache/miepython/droplet_4096ang-360az-6wl_sphe50um-100w-10pvar.bin")
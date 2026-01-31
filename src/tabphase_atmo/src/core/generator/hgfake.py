import numpy as np
import struct
from dataclasses import dataclass


# 1. Configuration matching your C++ expectation
@dataclass
class MieConfig:
    num_angles: int = 4096  # Matches your 4k resolution
    num_wavelengths: int = 64  # Standard spectral resolution
    min_wavelength: float = 300.0
    max_wavelength: float = 800.0
    g: float = 0.8  # Anisotropy: 0.8 is forward-scattering but SMOOTH


def henyey_greenstein(theta, g):
    """
    Standard HG phase function (normalized to 1 over sphere).
    p(theta) = (1/4pi) * (1 - g^2) / (1 + g^2 - 2g cos(theta))^(3/2)
    """
    mu = np.cos(theta)
    denom = (1 + g ** 2 - 2 * g * mu) ** 1.5
    # Mitsuba usually expects raw values, PDF normalization happens in the integrator/plugin.
    # We output the raw phase function value here.
    return (1.0 / (4.0 * np.pi)) * (1 - g ** 2) / denom


def save_fake_mie(filename, config):
    # 1. Generate Angles (0 to Pi)
    theta = np.linspace(0, np.pi, config.num_angles)

    # 2. Calculate HG Values
    # We use the same 'g' for all wavelengths to verify stability first.
    # If this works, the "Color Noise" is definitely mostly MIS weight variance.
    hg_values = henyey_greenstein(theta, config.g)

    # 3. Create Phase Table [Wavelengths, Angles]
    # (Matches the shape expected by your generate.py logic before saving)
    phase_table = np.zeros((config.num_wavelengths, config.num_angles), dtype=np.float32)

    for i in range(config.num_wavelengths):
        phase_table[i, :] = hg_values

    # 4. Save Binary (Exact replica of your save_binary_file)
    # The C++ loader expects interleaved data: [Angle0_Wvl0, Angle0_Wvl1, ... Angle1_Wvl0...]
    # So we Transpose the [Wvl, Angle] array.
    phase_interleaved = phase_table.T
    data_flat = phase_interleaved.flatten().astype(np.float32)

    print(f"[GEN] Generating Fake Mie HG (g={config.g})...")
    print(f"      File: {filename}")
    print(f"      Res:  {config.num_angles} angles x {config.num_wavelengths} wvl")

    with open(filename, "wb") as f:
        # Header
        f.write(b"ATMPHASE")
        f.write(struct.pack("<I", 1))  # Version
        f.write(struct.pack("<I", config.num_angles))
        f.write(struct.pack("<I", config.num_wavelengths))
        f.write(struct.pack("<f", float(config.min_wavelength)))
        f.write(struct.pack("<f", float(config.max_wavelength)))

        # Data
        f.write(data_flat.tobytes())

    # Note: We skip the MIS metadata. Your C++ constructor provided earlier
    # stops reading after 'm_data', so appending it is unnecessary for the render test.
    print("[GEN] Saved successfully.")


if __name__ == "__main__":
    # Generate a strong forward scattering (g=0.8) similar to water droplets
    # but strictly smooth (no glory, no rainbow, no interference ripples).
    conf = MieConfig(g=0.8)
    save_fake_mie("../../../benchmarks/cache/miepython/0um_mean_0um_std_0bins_0ang_0.bin", conf)
import struct
from pathlib import Path
import numpy as np
import pytest


def _read_atmphase_file(path: Path):
    """
    Read the custom ATMPHASE binary header and return the phase table as
    a numpy array shaped (num_wavelengths, num_angles) with dtype float32.

    The file layout (as written by `save_binary_file`):
      - 8 bytes: ASCII "ATMPHASE"
      - uint32: version
      - uint32: num_angles
      - uint32: num_wavelengths
      - float32: min_wavelength
      - float32: max_wavelength
      - (num_angles * num_wavelengths) float32 values: data interleaved as
        phase_interleaved = phase_table.T flattened (i.e. saved shape is
        (num_angles, num_wavelengths)).
      - (optional) appended MIS metadata (ignored by this reader)
    """
    with path.open("rb") as f:
        header = f.read(8)
        if header != b"ATMPHASE":
            raise ValueError(f"File {path} does not start with ATMPHASE header")

        meta = f.read(20)  # 5 fields: I I I f f
        version, num_angles, num_wavelengths, min_wl, max_wl = struct.unpack("<IIIff", meta)

        count = int(num_angles) * int(num_wavelengths)
        data_bytes = count * 4
        data = f.read(data_bytes)
        if len(data) < data_bytes:
            raise ValueError(f"File {path} is truncated: expected {data_bytes} bytes of float data")

        arr = np.frombuffer(data, dtype=np.float32, count=count).copy()
        # Saved layout was (num_angles, num_wavelengths) flattened; transpose back
        arr = arr.reshape((int(num_angles), int(num_wavelengths))).T
        return arr, float(min_wl), float(max_wl)


def test_lut_integrates_to_one_for_all_bins():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if not data_dir.exists():
        pytest.skip("No data directory found; skipping LUT integration test")

    bin_files = list(data_dir.glob("*.bin"))
    if not bin_files:
        pytest.skip("No .bin LUT files found in data directory")

    failures = []
    for p in bin_files:
        phase_table, min_wl, max_wl = _read_atmphase_file(p)
        num_wavelengths, num_angles = phase_table.shape
        # Use same d_mu as generator: mu = linspace(1, -1, num_angles) => d_mu = 2/(num_angles-1)
        d_mu = 2.0 / (num_angles - 1)

        # compute per-wavelength integral (including 2π azimuthal factor)
        integrals = np.sum(phase_table, axis=1) * d_mu * 2.0 * np.pi

        # Check integrals are close to 1.0
        try:
            np.testing.assert_allclose(integrals, np.ones_like(integrals), rtol=1e-3, atol=1e-4)
        except AssertionError as e:
            failures.append((p, integrals))

    if failures:
        msg_lines = ["Some LUT files failed the normalization check:"]
        for p, integrals in failures:
            msg_lines.append(f"  {p}: min={integrals.min():.6f}, max={integrals.max():.6f}")
        pytest.fail("\n".join(msg_lines))


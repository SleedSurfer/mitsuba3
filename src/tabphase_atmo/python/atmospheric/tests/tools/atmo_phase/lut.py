"""Utilities for reading ATMPHASE LUT files and evaluating a reference phase
function from the loaded table.

Provides:
- read_atmphase_file(path) -> (phase_table, num_angles, num_wavelengths, min_wl, max_wl)
- lookup_phase_reference(phase_table, cos_theta, wavelength_nm, min_wl, max_wl)

The binary layout is the one produced by `generate.save_binary_file`:
- 8 bytes: ASCII "ATMPHASE"
- uint32: version
- uint32: num_angles (resolution)
- uint32: num_wavelengths (channels)
- float32: min_wavelength (nm)
- float32: max_wavelength (nm)
- float32[data_count]: data in (num_angles, num_wavelengths) layout flattened
  where the writer used `phase_interleaved = phase_table.T` before writing.

The returned `phase_table` has shape (num_wavelengths, num_angles).
"""
from pathlib import Path
import struct
import numpy as np
from typing import Tuple


def read_atmphase_file(path) -> Tuple[np.ndarray, int, int, float, float]:
    """Read ATMPHASE binary and return (phase_table, num_angles, num_wavelengths, min_wl, max_wl).

    phase_table is returned as float32 numpy array shaped (num_wavelengths, num_angles).
    """
    p = Path(path)
    with p.open("rb") as f:
        magic = f.read(8)
        if magic != b"ATMPHASE":
            raise ValueError(f"File {p} does not start with ATMPHASE magic")

        meta = f.read(4 + 4 + 4 + 4 + 4)  # version, num_angles, num_wl, min_wl, max_wl
        if len(meta) < 20:
            raise ValueError(f"File {p} header is truncated")

        version, num_angles, num_wavelengths, min_wl, max_wl = struct.unpack("<IIIff", meta)

        count = int(num_angles) * int(num_wavelengths)
        data_bytes = count * 4
        data = f.read(data_bytes)
        if len(data) < data_bytes:
            raise ValueError(f"File {p} is truncated: expected {data_bytes} bytes of float data")

        arr = np.frombuffer(data, dtype=np.float32, count=count).copy()
        # Writer stored phase_interleaved = phase_table.T with shape (num_angles, num_wavelengths)
        arr = arr.reshape((int(num_angles), int(num_wavelengths))).T

    return arr, int(num_angles), int(num_wavelengths), float(min_wl), float(max_wl)


def lookup_phase_reference(phase_table: np.ndarray, *, cos_theta, wavelength_nm, min_wl: float, max_wl: float):
    """Evaluate the phase reference from the LUT using bilinear interpolation.

    Inputs:
      - phase_table: numpy array shaped (num_wavelengths, num_angles)
      - cos_theta: numpy array-like of cos(theta) values in [-1, 1]
      - wavelength_nm: numpy array-like of wavelengths in nm (same shape as cos_theta)
      - min_wl, max_wl: LUT wavelength range

    Returns:
      - values: numpy array of same shape as inputs with interpolated phase values.

    Notes:
      - Values outside wavelength range are clamped to the endpoint channels.
      - cos_theta outside [-1,1] is clamped to that range.
    """
    cos_theta = np.asarray(cos_theta)
    wavelength_nm = np.asarray(wavelength_nm)

    if cos_theta.shape != wavelength_nm.shape:
        # allow scalar wavelength to broadcast
        try:
            wavelength_nm = np.broadcast_to(wavelength_nm, cos_theta.shape)
        except Exception:
            raise ValueError("cos_theta and wavelength_nm must have the same shape or be broadcastable")

    num_wl, num_mu = phase_table.shape

    # Map cos_theta (mu) into mu index: mu grid is linspace(1, -1, num_mu)
    # normalized t_mu = (1 - mu)/(2) maps mu=1->0, mu=-1->1
    mu = np.clip(cos_theta, -1.0, 1.0)
    t_mu = (1.0 - mu) * 0.5
    idx_mu = t_mu * (num_mu - 1)

    # Map wavelength into wl index
    if max_wl == min_wl:
        t_wl = np.zeros_like(wavelength_nm)
    else:
        t_wl = (wavelength_nm - min_wl) / (max_wl - min_wl)
    t_wl = np.clip(t_wl, 0.0, 1.0)
    idx_wl = t_wl * (num_wl - 1)

    # Indices
    i_mu0 = np.floor(idx_mu).astype(int)
    i_mu1 = np.minimum(i_mu0 + 1, num_mu - 1)
    w_mu1 = idx_mu - i_mu0
    w_mu0 = 1.0 - w_mu1

    i_w0 = np.floor(idx_wl).astype(int)
    i_w1 = np.minimum(i_w0 + 1, num_wl - 1)
    w_w1 = idx_wl - i_w0
    w_w0 = 1.0 - w_w1

    # Gather values
    # phase_table shape: (num_wl, num_mu)
    v00 = phase_table[i_w0, i_mu0]
    v01 = phase_table[i_w0, i_mu1]
    v10 = phase_table[i_w1, i_mu0]
    v11 = phase_table[i_w1, i_mu1]

    # Bilinear interpolation
    val = (w_w0 * (w_mu0 * v00 + w_mu1 * v01) +
           w_w1 * (w_mu0 * v10 + w_mu1 * v11))

    return val


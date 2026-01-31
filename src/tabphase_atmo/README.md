# Atmospheric Phase Function Plugin

This plugin provides phase functions for rendering atmospheric optical phenomena such as rainbows and halos using Mie scattering theory.

## Structure

- `python/atmospheric/` - Python scripts for generating phase function data
  - `generate.py` - Mie scattering computation using miepython
  - `visualize.py` - Visualization tools for verification
  - `test_generation.py` - Test script

## Requirements

Install the required Python package:
```bash
pip install miepython numpy scipy matplotlib
```

## Usage

### Generate Phase Function Data

```python
from tabphase_atmo.python.atmospheric.generate import generate_mie_table, save_binary_file

# Generate table
phase_table = generate_mie_table(
    radius_mean_um=50.0,      # Mean droplet radius in micrometers
    radius_std_um=10.0,        # Standard deviation
    num_angles=1000,           # Angle resolution
    num_wavelengths=16,        # Number of wavelength channels
    min_wavelength=360.0,      # Minimum wavelength (nm)
    max_wavelength=830.0       # Maximum wavelength (nm)
)

# Save to binary file
save_binary_file(
    filename="phase_function.bin",
    phase_table=phase_table,
    num_angles=1000,
    num_wavelengths=16,
    min_wavelength=360.0,
    max_wavelength=830.0
)
```

### Visualize Generated Data

```python
from tabphase_atmo.python.atmospheric.visualize import visualize_binary_file

visualize_binary_file("phase_function.bin", "visualization.png")
```

### Run Test

```bash
cd src/tabphase_atmo/python/atmospheric
python test_generation.py
```

## Binary File Format

- Magic: "ATMPHASE" (8 bytes)
- Version: 1 (uint32)
- Resolution: number of angle bins (uint32)
- Channels: number of wavelength channels (uint32)
- Min_wavelength: minimum wavelength in nm (float32)
- Max_wavelength: maximum wavelength in nm (float32)
- Data: float32[resolution * channels] (angle-fastest layout)


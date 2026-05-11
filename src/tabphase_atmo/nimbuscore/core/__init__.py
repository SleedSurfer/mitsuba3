"""
Atmospheric phase function generation and visualization tools.

This package provides tools for generating Mie scattering phase functions
for atmospheric optical phenomena (rainbows, halos, etc.) and saving them
to a binary format for use in Mitsuba rendering.
"""
from .generator.visualize import visualize_anisotropic, load_binary_file
from .gen_manager import create_atmospheric_phase


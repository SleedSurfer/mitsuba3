"""
Atmospheric phase function generation and visualization tools.

This package provides tools for generating Mie scattering phase functions
for atmospheric optical phenomena (rainbows, halos, etc.) and saving them
to a binary format for use in Mitsuba rendering.
"""

from .generate import generate_mie_table, save_binary_file, get_water_ior
from .visualize import visualize_binary_file, load_binary_file
from .wrapper import create_atmospheric_phase
from .config import MieConfig

__all__ = [
    'generate_mie_table',
    'save_binary_file',
    'get_water_ior',
    'visualize_binary_file',
    'load_binary_file',
    'create_atmospheric_phase',
]


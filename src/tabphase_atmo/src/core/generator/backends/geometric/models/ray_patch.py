import drjit as dr
from drjit.auto import UInt32
from dataclasses import dataclass

@dataclass
class RayPatch:
    """
    Wavefront quad representation of 4 adjacent rays used by the collection sphere
    """
    v0: UInt32
    v1: UInt32
    v2: UInt32
    v3: UInt32
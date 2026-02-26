import drjit as dr
from drjit.auto import Float, Int, Complex2f, Array3f
from dataclasses import dataclass

@dataclass
class PhasorRay():
    """
    Ray wrapper bundling orthogonal wave phasors
    """
    o: Array3f  # Ray origin
    d: Array3f  # Ray direction
    Ex: Complex2f  # Perpendicular wave amplitude & phase
    Ey: Complex2f  # Parallel wave amplitude & phase
    l: Float  # Traversed optical path length in mm
    f: Int  # Number of focal lines traversed
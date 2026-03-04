from drjit.auto import Float, Int, Complex2f, Array3f
from dataclasses import dataclass

@dataclass
class PhasorRay():
    """
    Ray wrapper bundling orthogonal wave phasors
    """
    origin: Array3f
    direction: Array3f
    Ex: Complex2f
    Ey: Complex2f
    opt_path_length: Float
    focal_lines_crossed: Int  # Number of focal lines traversed
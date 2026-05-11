from drjit.llvm.ad import Float, Int, Complex2f, Array3f
from dataclasses import dataclass

@dataclass
class PhasorRay():
    """
    Ray wrapper bundling orthogonal wave phasors
    """
    origin: Array3f
    direction: Array3f
    Ex_X: Complex2f
    Ey_X: Complex2f
    Ex_Y: Complex2f
    Ey_Y: Complex2f
    opt_path_length: Float
    focal_lines_crossed: Int
    basis_x: Array3f
    basis_y: Array3f
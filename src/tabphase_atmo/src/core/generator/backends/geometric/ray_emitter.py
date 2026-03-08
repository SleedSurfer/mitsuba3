import drjit as dr
from drjit.auto import Float, Int, UInt32, Array3f, Complex2f
from .models.phasor_ray import PhasorRay
from .models.ray_patch import RayPatch


class GridEmitter:
    @staticmethod
    def emit(grid_res: int, width_mm: float) -> tuple[PhasorRay, RayPatch]:
        """
        Emits a grid of parallel rays
        """
        num_rays = grid_res * grid_res

        index = dr.arange(UInt32, num_rays)
        ix = index % grid_res
        iy = index // grid_res

        step = width_mm / (grid_res - 1) if grid_res > 1 else 0.0
        x = Float(ix) * step - (width_mm / 2.0)
        y = Float(iy) * step - (width_mm / 2.0)
        z = dr.zeros(Float, num_rays) - (width_mm / 2.0)

        rays = dr.zeros(PhasorRay, num_rays)

        rays.origin = Array3f(x, y, z)
        rays.direction = Array3f(0.0, 0.0, 1.0)  #+Z
        rays.Ex = Complex2f(1.0, 0.0)
        rays.Ey = Complex2f(1.0, 0.0)
        rays.opt_path_length = Float(0.0)
        rays.basis_x = Array3f(1.0, 0.0, 0.0)
        rays.basis_y = Array3f(0.0, 1.0, 0.0)

        num_patches = (grid_res - 1) * (grid_res - 1)
        p_index = dr.arange(UInt32, num_patches)
        px = p_index % (grid_res - 1)
        py = p_index // (grid_res - 1)

        top_left = py * grid_res + px

        patches = dr.zeros(RayPatch, num_patches)
        patches.v0 = top_left  # Top-left
        patches.v1 = top_left + 1  # Top-right
        patches.v2 = top_left + grid_res  # Bottom-left
        patches.v3 = top_left + grid_res + 1  # Bottom-right

        return rays, patches
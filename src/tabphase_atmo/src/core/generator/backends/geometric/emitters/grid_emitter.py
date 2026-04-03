import numpy as np
import drjit as dr
from drjit.auto import Float, UInt32, Array3f, Complex2f
from models.phasor_ray import PhasorRay
from models.ray_patch import RayPatch


class GridEmitter:
    @staticmethod
    def emit(grid_res: int, width_mm: float, offset_x: float, offset_y: float, rot_angle: float,
             pol: str = 'X') -> tuple:
        num_rays = grid_res * grid_res
        index = dr.arange(UInt32, num_rays)
        ix = index % grid_res
        iy = index // grid_res

        step = width_mm / (grid_res - 1) if grid_res > 1 else 0.0

        logical_x = Float(ix) * step - (width_mm / 2.0) + offset_x
        logical_y = Float(iy) * step - (width_mm / 2.0) + offset_y

        cos_a = float(np.cos(rot_angle))
        sin_a = float(np.sin(rot_angle))
        world_x = logical_x * cos_a - logical_y * sin_a
        world_y = logical_x * sin_a + logical_y * cos_a
        z = dr.zeros(Float, num_rays) - (width_mm / 2.0)

        rays = dr.zeros(PhasorRay, num_rays)
        rays.origin = Array3f(world_x, world_y, z)
        rays.direction = Array3f(0.0, 0.0, 1.0)

        rays.basis_x = Array3f(1.0, 0.0, 0.0)
        rays.basis_y = Array3f(0.0, 1.0, 0.0)

        if pol == 'X':
            rays.Ex = Complex2f(1.0, 0.0)
            rays.Ey = Complex2f(0.0, 0.0)
        else:
            rays.Ex = Complex2f(0.0, 0.0)
            rays.Ey = Complex2f(1.0, 0.0)

        rays.opt_path_length = Float(0.0)

        num_patches = (grid_res - 1) * (grid_res - 1)
        p_index = dr.arange(UInt32, num_patches)
        px = p_index % (grid_res - 1)
        py = p_index // (grid_res - 1)

        top_left = py * grid_res + px

        patches = dr.zeros(RayPatch, num_patches)
        patches.v0 = top_left
        patches.v1 = top_left + 1
        patches.v2 = top_left + grid_res
        patches.v3 = top_left + grid_res + 1

        return rays, patches, logical_x, logical_y
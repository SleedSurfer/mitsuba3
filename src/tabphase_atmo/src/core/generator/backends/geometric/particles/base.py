from typing import Protocol, Tuple
import drjit as dr
from drjit.cuda import Float, Bool, Array3f, Complex2f
from ..models.phasor_ray import PhasorRay


class Particle(Protocol):
    """
    Contract for any scattering particle geometry, feel free to add your own geometry.
    """

    def intersect(self, rays: PhasorRay) -> Tuple[Bool, Float]:
        """
        Calculates the intersection of rays with the particle.
        Returns a mask of which rays hit, and the distance 't' to the hit point.
        """
        ...

    def scatter(self, rays: PhasorRay, hit_mask: Bool, t: Float, ior_water: float) -> Tuple[
        Array3f, Array3f, Complex2f, Complex2f, Complex2f, Complex2f, Array3f]:
        """
        Calculates surface normals and offloads to optics to get new trajectories
        and Fresnel coefficients.
        Returns: (d_reflected, d_refracted, r_perp, r_para, t_perp, t_para, normals)
        """
        ...

    def get_focal_crossings(self, bounce_p: int) -> int:
        """
        Returns the number of focal lines crossed for a specific exit bounce 'p'.
        """
        ...
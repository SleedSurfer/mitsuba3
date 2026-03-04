from typing import Protocol, Tuple
import drjit as dr
from drjit.auto import Float, Bool, Array3f, Complex2f
from ..models.phasor_ray import PhasorRay

class Particle(Protocol):
    """
    Contract for any scattering particle geometry.
    """
    def intersect(self, rays: PhasorRay) -> Tuple[Bool, Float]:
        ...

    def scatter(self, rays: PhasorRay, hit_mask: Bool, t: Float, ior_water: float) -> Tuple[
        Array3f, Array3f, Complex2f, Complex2f, Complex2f, Complex2f, Array3f]:
        ...
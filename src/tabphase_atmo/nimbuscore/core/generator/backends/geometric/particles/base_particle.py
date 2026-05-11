from typing import Protocol, Tuple, runtime_checkable
import drjit as dr
from drjit.llvm.ad import Float, Bool, Array3f, Complex2f
from ..models.phasor_ray import PhasorRay

@runtime_checkable
class Particle(Protocol):
    def intersect(self, rays: PhasorRay) -> Tuple[Bool, Float]: ...
    def scatter(self, rays: PhasorRay, hit_mask: Bool, t: Float, ior_water: float) -> Tuple[Array3f, Array3f, Complex2f, Complex2f, Complex2f, Complex2f, Array3f, Bool]: ...

@runtime_checkable
class SphericalParticle(Particle, Protocol):
    """ Base for all smooth, curved geometries (Spheres, Oblates, Dropouts). Routed to Phasors. """
    pass

@runtime_checkable
class PrismParticle(Particle, Protocol):
    """ Base for all faceted geometries with flat planes. Routed to Photons. """
    def randomize_orientations(self, num_rays: int, rng): ...
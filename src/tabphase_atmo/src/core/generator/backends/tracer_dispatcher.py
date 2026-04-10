import numpy as np
import drjit as dr
from drjit.auto import Float
dr.set_log_level(dr.LogLevel.Error)

from .base import ScatteringBackend, ParticleSize

from .geometric.particles.base_particle import SphericalParticle, PrismParticle
from .geometric.particles.particle_factory import build_particle
# The Collectors
from .geometric.Collectors.photon_collector import PhotonCollectionSphere
from .geometric.Collectors.grid_collector import CollectionSphere
from .geometric.pipelines.pipeline_phasor import run_phasor_pipeline
from .geometric.pipelines.pipeline_photon import run_photon_pipeline


class DrJitRaytracerBackend(ScatteringBackend):
    name: str = "drjit_raytracer"

    def __init__(self, grid_res: int = 600, num_batches: int = 25, num_phi_bins: int = 1,
                 particle_shape: str = "sphere", sun_elevation_deg: float = 0.0): # <-- Catch it here
        self.grid_res = grid_res
        self.num_batches = num_batches
        self.num_phi_bins = num_phi_bins
        self.particle_shape = particle_shape
        self.sun_elevation_deg = sun_elevation_deg # <-- Save it

    def intensity_unpolarized(self, m: complex, wavelength_nm: float, mu: np.ndarray, size: ParticleSize) -> np.ndarray:
        ior_real_dr = dr.opaque(Float, float(m.real))
        ior_inv_dr = dr.opaque(Float, 1.0 / float(m.real))

        theta_requested = np.arccos(np.clip(mu, -1.0, 1.0))

        # <-- Pass it to the factory
        particle = build_particle(self.particle_shape, size, self.sun_elevation_deg)

        # --- PATH A: PHOTON PIPELINE (PRISMS) ---
        if isinstance(particle, PrismParticle):
            print(f"[Dispatcher] Habit '{self.particle_shape}' -> Photon Pipeline.")

            # Create a internal LINEAR theta grid for the simulation
            # We use your grid_res to stay efficient
            res = len(mu)
            theta_internal = np.linspace(0.0, np.pi, res)

            collector = PhotonCollectionSphere(mu_bins=np.cos(theta_internal), num_phi_bins=self.num_phi_bins)

            return run_photon_pipeline(
                particle, collector, theta_internal, theta_requested,
                ior_real_dr, ior_inv_dr, self.num_phi_bins
            )

        # --- PATH B: WAVEFRONT PIPELINE (SPHERES) ---
        elif isinstance(particle, SphericalParticle):
            # (Unchanged Sacred Path, using the 8192 internal bins)
            internal_bins = 8192
            theta_internal = np.linspace(0.0, np.pi, internal_bins)

            collector = CollectionSphere(mu_bins=np.cos(theta_internal),
                                         num_phi_bins=self.num_phi_bins,
                                         wavelength_nm=wavelength_nm)
            return run_phasor_pipeline(
                self, particle, collector, theta_internal, theta_requested,
                ior_real_dr, ior_inv_dr
            )
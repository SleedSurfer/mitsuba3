# shim package; real implementation lives under python/tools/atmo_phase
from python.atmospheric.tests.tools.atmo_phase.lut import read_atmphase_file, lookup_phase_reference

__all__ = ["read_atmphase_file", "lookup_phase_reference"]


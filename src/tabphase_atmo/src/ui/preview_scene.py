"""
preview_scene.py
Lightweight Mitsuba 3 scene generator for phase function preview rendering.

NOTE: The atmosphericphase plugin REQUIRES a binary .bin file path.
We cannot hot-swap phase data - must write to disk and reload scene.
"""
import mitsuba as mi
import numpy as np
from typing import Dict, Any
from pathlib import Path

# Note: Variant must be set by caller before importing this module


def create_preview_scene(
    phase_lut_path: str,
    resolution: tuple[int, int] = (512, 512),
    spp: int = 16
) -> Dict[str, Any]:
    """
    Create a minimal Mitsuba scene for phase function visualization.

    Scene: Cloud slab illuminated by directional sun, adapted from rainbow scene.

    Args:
        phase_lut_path: Path to .bin phase LUT file (REQUIRED by atmosphericphase plugin)
        resolution: Output image resolution (width, height)
        spp: Samples per pixel

    Returns:
        Mitsuba scene dictionary
    """
    phase_lut_path = str(Path(phase_lut_path).absolute())
    width, height = resolution

    sun_direction = [0, 0, -1]  # Sun from above

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'volpath',
            'max_depth': 2,  # Unlimited bounces
        },
        'sensor': {
            'type': 'perspective',
            'fov': 110.0,  # Wide FOV to catch scattering features
            'to_world': mi.ScalarTransform4f.look_at(
                origin=(0, 0, 3),
                target=(0, 0, 0),
                up=(0, 1, 0),
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': width,
                'height': height,
                'pixel_format': 'rgb',
                'rfilter': {'type': 'box'}
            }
        },
        # Directional light (The Sun)
        'sun': {
            'type': 'directional',
            'direction': sun_direction,
            'irradiance': {'type': 'rgb', 'value': 50.0/4},
        },
        # Cloud slab
        'cloud_slab': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.scale([15.0, 15.0, 0.5]),
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 0.1,
                'albedo': 0.98,
                'phase': {
                    'type': 'atmosphericphase',
                    'filename': phase_lut_path
                }
            }
        }
    }

    return scene_dict


def print_scene_tree(scene: mi.Scene):
    """
    Print the scene parameter tree for debugging.

    Args:
        scene: Loaded Mitsuba scene
    """
    params = mi.traverse(scene)

    print("\n" + "="*60)
    print("MITSUBA SCENE PARAMETER TREE")
    print("="*60)

    for key in sorted(params.keys()):
        value = params[key]
        value_type = type(value).__name__

        # Try to get shape if it's array-like
        shape_info = ""
        if hasattr(value, 'shape'):
            shape_info = f" shape={value.shape}"
        elif hasattr(value, '__len__'):
            try:
                shape_info = f" len={len(value)}"
            except:
                pass

        print(f"  {key:50s} : {value_type:20s}{shape_info}")

    print("="*60 + "\n")


def update_phase_data(scene: mi.Scene, phase_data: np.ndarray):
    """
    Hot-swap phase function data using mi.traverse().

    IMPORTANT: The new phase_data MUST match the dimensions of the initial dummy LUT
    (num_wavelengths, num_angles).

    Args:
        scene: Loaded Mitsuba scene
        phase_data: New phase data [num_wavelengths, num_angles]
    """
    import drjit as dr

    params = mi.traverse(scene)

    # Find the phase data key (should be cloud_slab.interior.phase.data or similar)
    data_key = None
    for key in params.keys():
        if 'phase' in key.lower() and 'data' in key.lower():
            data_key = key
            break
    
    if data_key is None:
        # Print all phase-related keys for debugging
        phase_keys = [k for k in params.keys() if 'phase' in k.lower()]
        raise RuntimeError(
            f"Could not find phase data buffer in scene.\n"
            f"Available phase-related keys: {phase_keys}"
        )

    print(f"✓ Hot-swapping phase data via: {data_key}")

    # Interleave: [angle][wavelength] order (as expected by plugin)
    phase_interleaved = phase_data.T  # Now [num_angles, num_wavelengths]
    flat_data = phase_interleaved.flatten().astype(np.float32)

    # Update Dr.Jit buffer
    params[data_key] = flat_data
    params.update()

    # Force evaluation
    dr.eval(params[data_key])

    print(f"✓ Updated phase data: {phase_data.shape} -> {flat_data.shape}")

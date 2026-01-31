import mitsuba as mi
from ..utils import get_phase_plugin


def get_scene(config):
    """
    Cornell Box with a spectral FogBox and a dielectric sphere.
    Refactored for the accumulation bench.
    """

    # --- PHASE & ATMOSPHERE ---
    #Using your 1.2um "rain" droplet settings
    phase_mist = get_phase_plugin(
        radius_mean=0.2,
        radius_std=0.03,
        note="glory_020_003"
    )
    # phase_mist = get_phase_plugin(
    #     radius_mean=0.0,
    #     radius_std=0.0,
    #     num_angles=0,
    #     num_wavelengths=0,
    #     note="0"
    # )

    # Helper to ingest raw XML matrices
    def T(flat_list):
        return mi.ScalarTransform4f(mi.ScalarMatrix4f(flat_list))

    # --- SCENE DICT ---
    return {
        'type': 'scene',
        'integrator': {
            'type': 'volpathmis',
            'max_depth': 1024,
            'rr_depth': 10
        },
        'sensor': {
            'type': 'perspective',
            'fov': 19.5,
            'to_world': T([
                -1, 0, 1.50996e-007, -1.05697e-006,
                0, 1, 0, 1,
                -1.50996e-007, 0, -1, 7,
                0, 0, 0, 1
            ]),
            'sampler': {
                'type': 'ldsampler',
                'sample_count': config['batch_size']
            },
            'film': {
                'type': 'hdrfilm',
                'width': config['res_w'],
                'height': config['res_h'],
                'pixel_format': 'rgb',
                'rfilter': {'type': 'box'}
            },
        },

        # --- MATERIALS ---
        'LeftWallBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.63, 0.065, 0.05]}},
        'RightWallBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.091]}},
        'WhiteBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}},
        'LightBSDF': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.0, 0.0, 0.0]}},

        'SphereBSDF': {
            'type': 'roughdielectric',
            'int_ior': 1.5,
            'ext_ior': 1.0,
            'distribution': 'ggx',
            'alpha': 0.01,
        },

        # --- GEOMETRY ---
        'Floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 0]).rotate([1, 0, 0], -90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'WhiteBSDF'}
        },
        'Ceiling': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 2, 0]).rotate([1, 0, 0], 90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'WhiteBSDF'}
        },
        'BackWall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 1, -1]).scale(1),
            'bsdf': {'type': 'ref', 'id': 'WhiteBSDF'}
        },
        'RightWall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([1, 1, 0]).rotate([0, 1, 0], -90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'RightWallBSDF'}
        },
        'LeftWall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([-1, 1, 0]).rotate([0, 1, 0], 90).scale(1),
            'bsdf': {'type': 'ref', 'id': 'LeftWallBSDF'}
        },

        'Sphere': {
            'type': 'sphere',
            'radius': 0.3,
            'center': [-0.22827, 1.2, 0.152505],
            'bsdf': {'type': 'ref', 'id': 'SphereBSDF'},
        },

        # --- LIGHT ---
        'Light': {
            'type': 'rectangle',
            'to_world': T([
                -0.0025, -1.91069e-015, 4.37114e-008, -0.005,
                -2.18557e-010, 2.18557e-008, -0.5, 1.98,
                0, -0.002, -8.74228e-011, -0.03,
                0, 0, 0, 1
            ]),
            'bsdf': {'type': 'ref', 'id': 'LightBSDF'},
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'spectrum',
                    'value': 541127.0
                }
            }
        },

        # --- VOLUMETRIC FOG BOX ---
        'FogBox': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.scale(2.5),
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 0.8,
                'albedo': 0.9,
                'sample_emitters': True,
                'phase': phase_mist
            }
        }
    }
import math
import mitsuba as mi

def create_directional_sun(sun_elevation_deg):
    elev_rad = math.radians(max(0.001, sun_elevation_deg))
    sun_pos_vector = [0.0, math.sin(elev_rad), math.cos(elev_rad)]
    travel_direction = [-x for x in sun_pos_vector]

    return {
        "type": "directional",
        "direction": travel_direction,
        "irradiance": {
            "type": "rgb",
            "value": [x/16 for x in [120.0, 110.0, 95.0]]
        }
    }

def create_sweeping_camera(sun_elev_deg, cam_azim_deg):
    elev_rad = math.radians(sun_elev_deg)
    yaw_rad = math.radians(cam_azim_deg - 180.0)

    target_dir = [
        math.cos(elev_rad) * math.sin(yaw_rad),
        math.sin(elev_rad),
        math.cos(elev_rad) * math.cos(yaw_rad)
    ]

    return mi.ScalarTransform4f.look_at(
        origin=[0.0, 0.0, 0.0],
        target=target_dir,
        up=[0.0, 1.0, 0.0]
    )

def build_scene(params):
    """
    The main pipeline. Takes the dumb numbers from runner.py and builds the Mitsuba dictionary.
    """
    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": params['max_depth'],
            "rr_depth": 2
        },

        #--- THE MEDIUM ---
        "my_atmo_medium": {
            "type": "homogeneous",
            "sigma_t": 0.001,
            "albedo": 0.98,
            "phase": params['phase_dict'] # <--- INJECTED DIRECTLY HERE
        },

        # --- THE CAMERA ---
        "sensor": {
            "type": "perspective",
            "focal_length": "16mm",
            "to_world": create_sweeping_camera(params['sun_elevation'], params['camera_azimuth']),
            "sampler": {
                "type": "independent",
                "sample_count": params['spp']
            },
            "medium": {
                "type": "ref",
                "id": "my_atmo_medium"
            },
            "film": {
                "type": "hdrfilm",
                "width": params['res_w'],
                "height": params['res_h'],
                "pixel_format": "rgb",
                "component_format": "float32"
            }
        },

        # --- THE BOUNDARY ---
        "atmosphere_boundary": {
            "type": "sphere",
            "radius": 400.0,
            "to_world": mi.ScalarTransform4f.scale(1.0),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "ref",
                "id": "my_atmo_medium"
            }
        },

        # --- EMITTERS ---
        "envmap": {
            "type": "envmap",
            "filename": "/home/speedlord/mitsuba3/src/tabphase_atmo/assets/envmaps/passendorf_snow_4k_sunless.exr",

        },

        "sun_emitter": create_directional_sun(params['sun_elevation'])
    }

    return scene_dict
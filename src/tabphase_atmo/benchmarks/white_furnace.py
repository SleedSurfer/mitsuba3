import mitsuba as mi
import numpy as np

# Load your rendered white furnace EXR
# Ensure you are pointing this to the output of your furnace scene
bmp = mi.Bitmap("/home/speedlord/mitsuba3/src/tabphase_atmo/outputs/white_furnace/preview_20_1280spp.exr")
img = np.array(bmp)

# Crop out the edges to only look at pure medium pixels.
# Assuming a 512x512 image where the sphere is dead center.
# Adjust these indices if your resolution or framing changes.
center_pixels = img[200:312, 200:312, :]

mean_radiance = np.mean(center_pixels)
variance = np.var(center_pixels)

print(f"Mean Radiance: {mean_radiance:.4f}")
print(f"Variance: {variance:.4f}")
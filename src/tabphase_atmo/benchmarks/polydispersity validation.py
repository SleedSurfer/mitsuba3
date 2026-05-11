import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import os


def validate_graphics(img_path_a, img_path_b):
    """
    Calculates similarity metrics for two images.
    Logic:
    1. Load images and convert to RGB.
    2. Ensure identical dimensions.
    3. Compute MSE, PSNR, and SSIM.
    """
    if not (os.path.exists(img_path_a) and os.path.exists(img_path_b)):
        print("❌ Error: One or both image paths are invalid. Check your filenames!")
        return

    # Load images
    img_a = Image.open(img_path_a).convert('RGB')
    img_b = Image.open(img_path_b).convert('RGB')

    # Quick OCD check: Dimensions must match for pixel-wise comparison
    if img_a.size != img_b.size:
        print(f"⚠️ Warning: Resizing image B from {img_b.size} to {img_a.size}")
        img_b = img_b.resize(img_a.size, Image.Resampling.LANCZOS)

    # Convert to numpy arrays
    arr_a = np.array(img_a)
    arr_b = np.array(img_b)

    # 1. MSE (Mean Squared Error)
    # Lower is better. 0 = identical.
    mse = np.mean((arr_a - arr_b) ** 2)

    # 2. PSNR (Peak Signal-to-Noise Ratio) in $dB$
    # Higher is better. >40dB is typically indistinguishable.
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    # 3. SSIM (Structural Similarity Index)
    # 1.0 = identical. This is the 'human-eye' metric.
    # We use channel_axis=2 because our arrays are (H, W, 3)
    s_index = ssim(arr_a, arr_b, channel_axis=2)

    print("-" * 30)
    print(f"📊 VALIDATION RESULTS")
    print("-" * 30)
    print(f"SSIM: {s_index:.4f} (Target: >0.98)")
    print(f"PSNR: {psnr:.2f} $dB$ (Target: >40.0)")
    print(f"MSE:  {mse:.4f} (Target: <5.0)")
    print("-" * 30)


if __name__ == "__main__":
    # Replace these with the actual filenames of your exported renders
    IMAGE_A = "/home/speedlord/mitsuba3/src/tabphase_atmo/benchmarks/polydispersity/brute.png"
    IMAGE_B = "/home/speedlord/mitsuba3/src/tabphase_atmo/benchmarks/polydispersity/post.png"

    validate_graphics(IMAGE_A, IMAGE_B)
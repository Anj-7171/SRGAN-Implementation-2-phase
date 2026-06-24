import os
import io
import numpy as np
from PIL import Image, ImageFilter

# -------------------------
# Paths
# -------------------------
input_dir = "data/test"
output_dir = "degraded"

os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Degradation Parameters
# -------------------------
SCALE_FACTOR = 4
BLUR_RADIUS = 2
JPEG_QUALITY = 20
NOISE_STD = 10

# -------------------------
# Process Images
# -------------------------
for img_name in os.listdir(input_dir):

    if not img_name.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp")
    ):
        continue

    img_path = os.path.join(input_dir, img_name)

    hr = Image.open(img_path).convert("RGB")

    # =====================================
    # Step 1: Gaussian Blur
    # =====================================
    degraded = hr.filter(
        ImageFilter.GaussianBlur(radius=BLUR_RADIUS)
    )

    # =====================================
    # Step 2: Bicubic Downsample x4
    # =====================================
    degraded = degraded.resize(
        (
            max(1, hr.width // SCALE_FACTOR),
            max(1, hr.height // SCALE_FACTOR),
        ),
        Image.BICUBIC,
    )

    # =====================================
    # Step 3: JPEG Compression
    # =====================================
    buffer = io.BytesIO()

    degraded.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
    )

    buffer.seek(0)

    degraded = Image.open(buffer).convert("RGB")

    # =====================================
    # Step 4: Add Gaussian Noise
    # =====================================
    degraded_np = np.array(degraded).astype(np.float32)

    noise = np.random.normal(
        0,
        NOISE_STD,
        degraded_np.shape
    )

    degraded_np = np.clip(
        degraded_np + noise,
        0,
        255
    ).astype(np.uint8)

    degraded = Image.fromarray(degraded_np)

    # =====================================
    # Upscale for Visualization
    # =====================================
    degraded_vis = degraded.resize(
        hr.size,
        Image.NEAREST
    )

    base_name = os.path.splitext(img_name)[0]

    # Save original
    hr.save(
        os.path.join(
            output_dir,
            f"{base_name}_original.png"
        )
    )

    # Save degraded
    degraded_vis.save(
        os.path.join(
            output_dir,
            f"{base_name}_degraded.png"
        )
    )

    # =====================================
    # Side-by-side comparison
    # =====================================
    comparison = Image.new(
        "RGB",
        (hr.width * 2, hr.height)
    )

    comparison.paste(hr, (0, 0))
    comparison.paste(degraded_vis, (hr.width, 0))

    comparison.save(
        os.path.join(
            output_dir,
            f"{base_name}_comparison.png"
        )
    )

    print(f"Processed: {img_name}")

print("\nDone!")
print(f"Saved results in '{output_dir}'")
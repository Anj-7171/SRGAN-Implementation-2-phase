import os
import torch
from PIL import Image
from torchvision import transforms

from model import Generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load model
# -------------------------
model = Generator().to(DEVICE)

checkpoint = "checkpoints/pretrain_best.pth"

model.load_state_dict(
    torch.load(checkpoint, map_location=DEVICE)
)

model.eval()

# -------------------------
# Transforms
# -------------------------
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# -------------------------
# Paths
# -------------------------
input_dir = "data/test"
output_dir = "results/test"

os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Inference
# -------------------------
for img_name in os.listdir(input_dir):

    if not img_name.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp")
    ):
        continue

    img_path = os.path.join(input_dir, img_name)

    hr = Image.open(img_path).convert("RGB")

    # Create LR image (same degradation used in training)
    lr = hr.resize(
        (hr.width // 4, hr.height // 4),
        Image.BICUBIC
    )

    lr_tensor = to_tensor(lr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sr = model(lr_tensor)

    sr = sr.squeeze(0).cpu().clamp(0, 1)
    sr_img = to_pil(sr)

    # Upscale LR for visualization
    lr_vis = lr.resize(
        hr.size,
        Image.BICUBIC
    )

    base_name = os.path.splitext(img_name)[0]

    # -------------------------
    # Save individual images
    # -------------------------
    hr.save(
        os.path.join(
            output_dir,
            f"{base_name}_HR.png"
        )
    )

    lr_vis.save(
        os.path.join(
            output_dir,
            f"{base_name}_LR.png"
        )
    )

    sr_img.save(
        os.path.join(
            output_dir,
            f"{base_name}_SR.png"
        )
    )

    # -------------------------
    # Create comparison image
    # -------------------------
    comparison = Image.new(
        "RGB",
        (hr.width * 3, hr.height)
    )

    comparison.paste(hr, (0, 0))
    comparison.paste(lr_vis, (hr.width, 0))
    comparison.paste(sr_img, (2 * hr.width, 0))

    comparison.save(
        os.path.join(
            output_dir,
            f"{base_name}_comparison.png"
        )
    )

    print(f"Processed: {img_name}")

print("\nDone!")
print(f"Results saved to: {output_dir}")
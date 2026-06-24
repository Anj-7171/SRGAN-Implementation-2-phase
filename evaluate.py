import torch
import os
from model import Generator
from dataset import DIV2KDataset
from torch.utils.data import DataLoader
import torchvision
import config
from math import log10
from skimage.metrics import structural_similarity as ssim

G = Generator().to(config.DEVICE)
G.load_state_dict(torch.load("checkpoints/srgan_best.pth"))
G.eval()

loader = DataLoader(DIV2KDataset(config.DATA_DIR), batch_size=1)

psnr_list = []
ssim_list = []

os.makedirs("results", exist_ok=True)

with torch.no_grad():
    for i, (lr, hr) in enumerate(loader):
        lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

        sr = G(lr)
        bicubic = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic")

        # PSNR
        mse = torch.mean((sr - hr) ** 2)
        psnr = 10 * log10(1 / mse.item())
        psnr_list.append(psnr)

        # SSIM (convert to numpy)
        sr_img = sr.squeeze().cpu().permute(1,2,0).numpy()
        hr_img = hr.squeeze().cpu().permute(1,2,0).numpy()

        ssim_val = ssim(sr_img, hr_img, channel_axis=2)
        ssim_list.append(ssim_val)

        # save samples
        if i < 10:
            os.makedirs(f"results/img_{i}", exist_ok=True)

            torchvision.utils.save_image(lr, f"results/img_{i}/lr.png")
            torchvision.utils.save_image(sr, f"results/img_{i}/sr.png")
            torchvision.utils.save_image(hr, f"results/img_{i}/hr.png")
            torchvision.utils.save_image(bicubic, f"results/img_{i}/bicubic.png")

print("Avg PSNR:", sum(psnr_list)/len(psnr_list))
print("Avg SSIM:", sum(ssim_list)/len(ssim_list))
import torch
from model import Generator
from dataset import DIV2KDataset
from losses import pixel_loss
from torch.utils.data import DataLoader
import config
import os
import csv
from math import log10
from torchvision.utils import save_image

# -------------------
# Setup
# -------------------

G = Generator().to(config.DEVICE)
opt = torch.optim.Adam(G.parameters(), lr=config.LR)

train_loader = DataLoader(
    DIV2KDataset("data/train"),
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    DIV2KDataset("data/validation"),
    batch_size=1,
    shuffle=False
)

# directories
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs/pretrain_samples", exist_ok=True)

# -------------------
# CSV logging
# -------------------

log_file = "logs/pretrain_metrics.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "psnr"])

best_psnr = 0

# -------------------
# Training loop
# -------------------

for epoch in range(config.PRETRAIN_EPOCHS):

    # -------------------
    # Train
    # -------------------
    G.train()
    epoch_loss = 0

    for lr, hr in train_loader:
        lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

        sr = G(lr)
        loss = pixel_loss(sr, hr)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    # -------------------
    # Validation PSNR + sample saving
    # -------------------
    G.eval()
    psnr_total = 0

    with torch.no_grad():
        for idx, (lr, hr) in enumerate(val_loader):
            lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

            sr = G(lr)

            # -------------------
            # PSNR
            # -------------------
            mse = torch.mean((sr - hr) ** 2)
            psnr = 10 * log10(1 / mse.item())
            psnr_total += psnr

            # -------------------
            # SAVE SAMPLE IMAGES (only first batch)
            # -------------------
            if epoch % 5 == 0 and idx == 0:

                sr_save = torch.clamp(sr, 0, 1)

                save_image(lr, f"outputs/pretrain_samples/e{epoch}_lr.png")
                save_image(sr_save, f"outputs/pretrain_samples/e{epoch}_sr.png")
                save_image(hr, f"outputs/pretrain_samples/e{epoch}_hr.png")

    avg_loss = epoch_loss / len(train_loader)
    avg_psnr = psnr_total / len(val_loader)

    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f}")

    # -------------------
    # Save best model
    # -------------------
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(G.state_dict(),
                   f"{config.CHECKPOINT_DIR}/pretrain_best.pth")

    # -------------------
    # Save last checkpoint
    # -------------------
    torch.save(G.state_dict(),
               f"{config.CHECKPOINT_DIR}/pretrain_last.pth")

    # -------------------
    # Log CSV
    # -------------------
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_loss, avg_psnr])
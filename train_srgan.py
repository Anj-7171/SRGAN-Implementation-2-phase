import torch
from model import Generator, Discriminator
from losses import content_loss
import config

G = Generator().to(config.DEVICE)
D = Discriminator().to(config.DEVICE)

opt_G = torch.optim.Adam(G.parameters(), lr=config.LR)
opt_D = torch.optim.Adam(D.parameters(), lr=config.LR)

bce = torch.nn.BCEWithLogitsLoss()

G.load_state_dict(torch.load("checkpoints/pretrain_best.pth"))

for epoch in range(config.GAN_EPOCHS):
    for lr, hr in loader:
        lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

        sr = G(lr)

        real = torch.ones(hr.size(0), 1).to(config.DEVICE)
        fake = torch.zeros(hr.size(0), 1).to(config.DEVICE)

        # Discriminator
        loss_D = bce(D(hr), real) + bce(D(sr.detach()), fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Generator
        loss_G = content_loss(sr, hr) + 1e-3 * bce(D(sr), real)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(epoch, loss_G.item(), loss_D.item())
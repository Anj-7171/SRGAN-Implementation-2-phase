import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, scale=4):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
        self.prelu = nn.PReLU()

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(64) for _ in range(5)
        ])

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        # upsampling
        upsample = []
        for _ in range(scale // 2):
            upsample += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*upsample)

        self.out = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.res_blocks(x1)
        x2 = self.conv2(x2)
        x = x1 + x2
        x = self.upsample(x)
        return torch.tanh(self.out(x))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )

        self.net = nn.Sequential(
            block(3, 64, 1),
            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.net(x)
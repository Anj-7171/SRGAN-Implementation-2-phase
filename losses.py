import torch.nn as nn
import torchvision.models as models

mse = nn.MSELoss()

vgg = models.vgg19(pretrained=True).features[:36].eval()

for p in vgg.parameters():
    p.requires_grad = False


def content_loss(sr, hr):
    return mse(vgg(sr), vgg(hr))


def pixel_loss(sr, hr):
    return mse(sr, hr)
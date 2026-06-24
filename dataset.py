from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T
import random

class DIV2KDataset(Dataset):
    def __init__(self, root, scale=4, patch_size=96):
        self.root = root
        self.images = os.listdir(root)

        self.scale = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # -------------------------
        # Random crop HR patch
        # -------------------------
        w, h = img.size
        ps = self.patch_size

        if w < ps or h < ps:
            img = img.resize((ps, ps))

        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)

        hr = img.crop((x, y, x + ps, y + ps))

        # -------------------------
        # LR generation
        # -------------------------
        lr_size = ps // self.scale
        lr = hr.resize((lr_size, lr_size), Image.BICUBIC)

        transform = T.ToTensor()

        return transform(lr), transform(hr)
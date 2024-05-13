import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import os
import time

class ArtificialImageDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(root) if f.endswith('.tif')])
        self.labels = sorted([f for f in os.listdir(root) if f.endswith('.json')])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        label_path = os.path.join(self.root, self.labels[idx])
        
        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32) / 65535  # Normalize 16-bit image
        img = torch.from_numpy(img).permute(2, 0, 1)  # Convert to [C, H, W]

        with open(label_path, 'r') as file:
            target = json.load(file)
        target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

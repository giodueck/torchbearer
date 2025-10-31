import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        with rasterio.open(img_path, "r") as f:
            image = f.read(1)
        image = image.astype("float32")
        image *= 255.0/10000  # The maximum value a pixel can be is 10_000
        image = np.clip(image, min=0, max=255.0)
        image = image.astype("uint8")

        image = torch.from_numpy(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

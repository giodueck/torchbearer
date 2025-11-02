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
        # # Introduce a bias to reduce images labeled False,
        # # as they are 3 times as abundant as True
        # labels = [self.img_labels.iloc[i, 1] for i in range(len(self.img_labels))]
        # mask = np.random.rand(len(self.img_labels))
        # mask += 0.5
        # self.mask = mask.astype("uint8")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        with rasterio.open(img_path, "r") as f:
            image = f.read()
        image = image.astype("float32")
        image *= 1.0/10000  # The maximum value a pixel can be is 10_000
        image = np.clip(image, min=0, max=1.0)

        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        label = self.img_labels.iloc[idx, 1]
        return image, label

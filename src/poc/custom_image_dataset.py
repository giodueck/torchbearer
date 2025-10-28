import os
import rasterio
import torch
from torch.utils.data import Dataset
import pandas as pd


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        with rasterio.open(img_path, "r") as f:
            image = torch.from_numpy(f.read(1))
        label = self.img_labels.iloc[idx, 1]
        return image, label

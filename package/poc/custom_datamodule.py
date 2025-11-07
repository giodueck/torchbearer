import lightning.pytorch as pl
from custom_image_dataset import CustomImageDataset
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        """
        Data setup, done once for all GPUs.
        Can be downloading data or instead call a custom dataset from setup, skipping this step.
        """
        pass

    def setup(self, stage: str = None):
        """
        Data setup, done once per GPU.
        """
        transforms = Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ])
        train_set_full = CustomImageDataset(
            "data/labels.csv", "data/", transforms)
        train_set_size = int(len(train_set_full) * 0.7)
        valid_set_size = int(len(train_set_full) * 0.2)
        test_set_size = len(train_set_full) - train_set_size - valid_set_size
        generator = torch.Generator().manual_seed(1234567)
        self.train, self.validate, self.test = random_split(
            train_set_full, [train_set_size, valid_set_size, test_set_size], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4, shuffle=False)

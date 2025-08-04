from collections.abc import Callable, Iterable
from pathlib import Path
import os

import kornia.augmentation as K
import numpy as np
import rasterio as rio
import torch
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader

from torch import nn
import torch.nn.functional as F

from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples, utils
from torchgeo.samplers import RandomGeoSampler, Units
from torchgeo.transforms import indices
from torchgeo.datamodules import GeoDataModule
from torchvision import transforms
import lightning as L

from torchvision.models.segmentation import deeplabv3_resnet50


class SurfaceWaterDataModule(GeoDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, num_workers: int = 1):
        super().__init__(dataset_class=RasterDataset, num_workers=num_workers)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_channels = 9
        self.num_classes = 2

    def prepare_data(self):
        root = Path(self.data_dir)
        utils.download_and_extract_archive(
            'https://hf.co/datasets/cordmaur/earth_surface_water/resolve/main/earth_surface_water.zip',
            root,
        )
        root = root / 'dset-s2'

        train_imgs = RasterDataset(
            paths=(root / 'tra_scene').as_posix(), crs='epsg:3395', res=10, transforms=self.scale
        )
        train_msks = RasterDataset(
            paths=(root / 'tra_truth').as_posix(), crs='epsg:3395', res=10
        )

        valid_imgs = RasterDataset(
            paths=(root / 'val_scene').as_posix(), crs='epsg:3395', res=10, transforms=self.scale
        )
        valid_msks = RasterDataset(
            paths=(root / 'val_truth').as_posix(), crs='epsg:3395', res=10
        )

        # Calculate the statistics (Mean and std) for the dataset
        mean, std = self.calc_statistics(train_imgs)

        # Please, note that we will create spectral indices using the raw (non-normalized) data. Then, when normalizing, the sensors will have more channels (the indices) that should not be normalized.
        # To solve this, we will add the indices to the 0's to the mean vector and 1's to the std vectors
        mean = np.concat([mean, [0, 0, 0]])
        std = np.concat([std, [1, 1, 1]])

        norm = K.Normalize(mean=mean, std=std)

        tfms = torch.nn.Sequential(
            indices.AppendNDWI(index_green=1, index_nir=3),
            indices.AppendNDWI(index_green=1, index_nir=5),
            indices.AppendNDVI(index_nir=3, index_red=2),
            norm,
        )

        train_imgs['image'] = tfms(train_imgs['image'])
        valid_imgs['image'] = tfms(valid_imgs['image'])

        # IMPORTANT
        train_msks.is_image = False
        valid_msks.is_image = False

        self.train_dset = train_imgs & train_msks
        self.valid_dset = valid_imgs & valid_msks
        self.train_imgs = train_imgs
        self.valid_imgs = valid_imgs

    def setup(self, stage: str):
        self.train_sampler = RandomGeoSampler(
            self.train_imgs, size=512, length=130, units=Units.PIXELS)
        self.valid_sampler = RandomGeoSampler(
            self.valid_imgs, size=512, length=64, units=Units.PIXELS)

        self.training_dataloader = DataLoader(
            self.train_dset, sampler=self.train_sampler, batch_size=2, collate_fn=stack_samples
        )
        self.valid_dataloader = DataLoader(
            self.valid_dset, sampler=self.valid_sampler, batch_size=2, collate_fn=stack_samples
        )

    def train_dataloader(self):
        return self.training_dataloader

    def val_dataloader(self):
        return self.valid_dataloader

    # TODO actual test dataloader
    def test_dataloader(self):
        return self.training_dataloader

    # TODO actual predict dataloader
    def predict_dataloader(self):
        return self.valid_dataloader

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

    def scale(self, item: dict):
        item['image'] = item['image'] / 10000
        return item

    # Data Standardization and Spectral Indices
    def calc_statistics(self, dset: RasterDataset):
        """
        Calculate the statistics (mean and std) for the entire dataset
        Warning: This is an approximation. The correct value should take into account the
        mean for the whole dataset for computing individual stds.
        For correctness I suggest checking: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """

        # To avoid loading the entire dataset in memory, we will loop through each img
        # The filenames will be retrieved from the dataset's rtree index
        files = [
            item.object for item in dset.index.intersection(dset.index.bounds, objects=True)
        ]

        # Resetting statistics
        accum_mean = 0
        accum_std = 0

        for file in files:
            img = rio.open(file).read() / 10000  # type: ignore
            accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
            accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

        # at the end, we shall have 2 vectors with length n=chnls
        # we will average them considering the number of images
        return accum_mean / len(files), accum_std / len(files)


class SurfaceWaterEncoder(nn.Module):
    def __init__(self):
        super(SurfaceWaterEncoder, self).__init__()
        self.resnet = deeplabv3_resnet50(weights=None, num_classes=2)
        backbone = self.resnet.get_submodule('backbone')
        conv = nn.modules.conv.Conv2d(
            in_channels=9,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        backbone.register_module('conv1', conv)

    def forward(self, x):
        return self.resnet(x)['out']


class SurfaceWaterLitModule(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        x = batch['image']
        print(f'{type(x)}')
        y = batch['mask']
        y_hat = self.encoder(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        y_hat = self.encoder(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001, weight_decay=0.01)
        return optimizer


resnet = SurfaceWaterEncoder()
model = SurfaceWaterLitModule(resnet)
data_module = SurfaceWaterDataModule(data_dir=Path("/root/data"), batch_size=8, num_workers=7)
trainer = L.Trainer(min_epochs=1, max_epochs=10)
trainer.fit(model=model, datamodule=data_module)

import torch
from torch.utils.data import DataLoader
from ..datasets import Sentinel2_60m, LabelDataset
from ..config.products import PRODUCTS
from torchgeo.datamodules.geo import GeoDataModule
from torchgeo.datasets import IntersectionDataset, random_grid_cell_assignment, stack_samples
import kornia.augmentation as K
from kornia.constants import DataKey, Resample
from torchgeo.samplers.utils import _to_tuple
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler, RandomGeoSampler
import matplotlib.pyplot as plt
import os
from typing import Sequence


class Sentinel2_60mDataModule(GeoDataModule):
    def __init__(
        self,
        batch_size=16,
        patch_size=64,
        length=None,
        num_workers=0,
        sentinel_path='data',
        sentinel_products=PRODUCTS,
        mask_path='masks',
        bands: Sequence[str] | None = None,
    ):
        super().__init__(IntersectionDataset, batch_size, patch_size, length, num_workers)

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
            extra_args={
                DataKey.MASK: {'resample': Resample.NEAREST,
                               'align_corners': None}
            },
        )

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=None,
            keepdim=True,
        )

        self.sentinel_path = sentinel_path
        self.sentinel_products = sentinel_products
        self.mask_path = mask_path
        self.bands = bands

    def prepare_data(self):
        """
        Data setup, done once for all GPUs.
        Can be downloading data or instead call a custom dataset from setup, skipping this step.
        """
        self.sentinel2 = Sentinel2_60m(
            self.sentinel_path, products=self.sentinel_products, bands=self.bands)
        self.label = LabelDataset(self.mask_path)
        self.dataset = self.sentinel2 & self.label

    def setup(self, stage: str):
        """
        Data setup, done once per GPU.
        """
        generator = torch.Generator().manual_seed(torch.Generator().initial_seed())

        self.train_dataset, self.val_dataset, self.test_dataset = random_grid_cell_assignment(
            self.dataset, [0.7, 0.2, 0.1], grid_size=8, generator=generator)

        if stage in ['fit']:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length)
            self.train_sampler = RandomGeoSampler(
                self.train_dataset, self.patch_size)
        if stage in ['fit', 'validate']:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size)
        if stage in ['test']:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=self.num_workers, collate_fn=stack_samples)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, sampler=self.val_sampler, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=stack_samples)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, sampler=self.test_sampler, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=stack_samples)

    def plot(self, sample, filename: str | None = None):
        if 'output' in sample.keys():
            figure, axes = plt.subplots(ncols=2, nrows=2)
            axes = axes.flatten()
        else:
            figure, axes = plt.subplots(ncols=2)

        plt.axis('on')
        img_fig = self.sentinel2.plot(sample, axes[0])
        label_fig = self.label.plot(sample, axes[1])
        if 'output' in sample.keys():
            output = sample['output'].cpu().squeeze()
            output_fig = axes[2].imshow(output, cmap='Blues')
            pred = (output > 0.5).float()
            pred_fig = axes[3].imshow(pred, cmap='Blues')

        if filename is not None:
            os.makedirs(filename.rsplit('/', 1)[0], exist_ok=True)
            plt.savefig(filename)
            print(filename)
        else:
            plt.show()
        plt.close(figure)


def createSentinel2_60mDataModule(params: dict):
    """
    Returns a configured Sentinel2_60mDataModule.

    If params contains a parameter for Sentinel2_60mDataModule, it is used,
    otherwise it is set to a default value. Inexistant parameters are ignored.
    """
    datamodule = Sentinel2_60mDataModule(
        batch_size=params.get('batch_size', 3),
        patch_size=params.get('patch_size', 128),
        length=params.get('length', 800),
        num_workers=params.get('num_workers', 6),
        sentinel_path=params.get('sentinel_path', 'data'),
        sentinel_products=params.get('sentinel_products', PRODUCTS),
        mask_path=params.get('mask_path', 'masks'),
        bands=params.get('bands', None))
    return datamodule

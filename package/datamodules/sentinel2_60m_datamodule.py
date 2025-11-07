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


class Sentinel2_60mDataModule(GeoDataModule):
    def __init__(self, batch_size=16, patch_size=64, length=None, num_workers=0, sentinel_path='data', sentinel_products=PRODUCTS, mask_path='masks'):
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

        self.sentinel_path = sentinel_path
        self.sentinel_products = sentinel_products
        self.mask_path = mask_path

    def prepare_data(self):
        """
        Data setup, done once for all GPUs.
        Can be downloading data or instead call a custom dataset from setup, skipping this step.
        """
        self.sentinel2 = Sentinel2_60m(
            self.sentinel_path, products=self.sentinel_products)
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

    def plot(self, sample):
        _, axes = plt.subplots(ncols=2)
        img_fig = self.sentinel2.plot(sample, axes[0])
        label_fig = self.label.plot(sample, axes[1])
        plt.axis('off')
        plt.show()

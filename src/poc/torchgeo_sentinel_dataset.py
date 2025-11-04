import torch
from torchgeo.datasets import RasterDataset, VectorDataset, stack_samples, unbind_samples
from torchgeo.samplers import RandomGeoSampler
from sys import argv
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.data import DataLoader


class Sentinel2(RasterDataset):
    filename_glob = "T*_B01_60m.jp2"
    filename_regex = r"^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])"
    date_format = '%Y%m%dT%H%M%S'
    is_image = True
    separate_files = True
    all_bands = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                 'B07', 'B8A', 'B09', 'B11', 'B12')
    rgb_bands = ('B04', 'B03', 'B02')

    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample['image'][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig


class LabelDataset(RasterDataset):
    """
    Create a mask image using a shapefile layer in QGIS and creating a raster from it using the "vector to raster" tool.
    Save this layer as GeoTiff.
    """

    filename_glob = "T*_mask.tif"
    filename_regex = r"^.{6}_(?P<date>\d{8}T\d{6})_mask"
    date_format = '%Y%m%dT%H%M%S'
    is_image = False

    def plot(self, sample):
        # Reorder and rescale the image
        mask = sample['mask']

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap='Blues')

        return fig


if __name__ == "__main__":
    img = Sentinel2(argv[1])
    mask = LabelDataset(argv[1])
    ds = img & mask
    g = torch.Generator().manual_seed(1)
    sampler = RandomGeoSampler(ds, size=256, length=3, generator=g)
    dataloader = DataLoader(ds, sampler=sampler, collate_fn=stack_samples)

    for batch in dataloader:
        sample = unbind_samples(batch)[0]
        img.plot(sample)
        mask.plot(sample)
        plt.axis('off')
        plt.show()

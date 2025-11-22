import torch
import lightning.pytorch as pl
from torchgeo.datasets import unbind_samples
from sys import argv
import pathlib
import rasterio
from rasterio.io import MemoryFile
import os
from pathlib import Path

from . import datamodules
from . import models
from .config import configparser


def create_rasterio_dataset(data, crs, transform):
    # Receives a 2D array, a transform and a crs to create a rasterio dataset
    memfile = MemoryFile()
    dataset = memfile.open(driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, crs=crs,
                           transform=transform, dtype=data.dtype)
    dataset.write(data, 1)

    return dataset


# Args:
# 1. path/to/checkpoint
# 2. path/to/version/logs which contains the hparams.yaml and config.yaml files
# 3. config override, to plot a different dataset than the train dataset
if __name__ == "__main__":
    default_config = configparser.defaultConfig()[0]

    checkpoint = argv[1]
    path = pathlib.PosixPath(argv[2])
    hparams_file = path / "hparams.yaml"
    config_file = path / "config.yaml"
    conf = default_config | configparser.parseConfig(config_file)[0]
    model = models.model_classes[conf['model']].load_from_checkpoint(
        argv[1], hparams_file=hparams_file)

    # For overrides of e.g. the dataset to plot
    # Should be compatible with original model, of course
    conf |= configparser.parseConfig(argv[3])[0]

    pl.seed_everything(conf['seed'])
    datamodule = datamodules.datamodules[conf['datamodule']](
        conf['datamodule_params'])

    # Plot sample testing images
    # If the dataset has a None mask_path, the whole dataset is treated as the test dataset
    datamodule.prepare_data()
    datamodule.setup('test')

    os.makedirs('data/work', exist_ok=True)
    path = Path('data/work')
    for f in path.glob('*.tif'):
        os.remove(f)
    i = 0
    bound_set = set()
    for batch in datamodule.test_dataloader():
        with torch.no_grad():
            image_tensor = (torch.FloatTensor(batch['image'])).to(
                torch.device('cuda'))
            batch['output'] = model(image_tensor)

        for sample in unbind_samples(batch):
            bounds = sample['bounds']
            bound_set.add(bounds)
            shape = sample['output'].cpu().numpy().shape
            # west, south, east, north, width, height
            transform = rasterio.transform.from_bounds(
                bounds.minx,
                bounds.miny,
                bounds.maxx,
                bounds.maxy,
                shape[1],
                shape[2])
            ds = create_rasterio_dataset(
                sample['output'].cpu().numpy().squeeze().clip(0.3, 1.0), sample['crs'], transform)
            out_meta = ds.meta.copy()
            out_meta.update(
                {
                    'driver': 'GTiff',
                    'height': shape[1],
                    'width': shape[2],
                    'transform': transform,
                    'count': 1,
                    'dtype': rasterio.uint8,
                }
            )
            with rasterio.open(f'data/work/patch_{i}.tif', 'w', **out_meta) as dst:
                arr = ds.read(1) * 255
                arr = arr.astype(rasterio.uint8)
                dst.write(arr, 1)
                i += 1
                print(i, end='\r')
    print()

    path = Path('.')
    dest, output_transform = rasterio.merge.merge(list(path.glob('data/work/*.tif')), method='max')

    with rasterio.open('data/work/patch_0.tif') as src:
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": dest.shape[1],
                "width": dest.shape[2],
                "transform": output_transform
            }
        )

    with rasterio.open("data/merged.tif", "w", **out_meta) as fp:
        fp.write(dest)

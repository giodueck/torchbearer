import torch
import lightning.pytorch as pl
from torchgeo.datasets import unbind_samples
from sys import argv
import pathlib
import rasterio
from rasterio.io import MemoryFile
import os
import yaml

from . import datamodules
from . import models
from .config import configparser


def create_rasterio_dataset(data, crs, transform):
    # Receives a 2D array, a transform and a crs to create a rasterio dataset
    memfile = MemoryFile()
    dataset = memfile.open(driver='GTiff',
                           height=data.shape[0],
                           width=data.shape[1],
                           count=1,
                           crs=crs,
                           transform=transform,
                           dtype=data.dtype)
    dataset.write(data, 1)

    return dataset


def create_inferrence_mask(config: dict, dst_path: str):
    pl.seed_everything(conf['seed'])
    datamodule = datamodules.datamodules[conf['datamodule']](
        conf['datamodule_params'])

    # Plot sample testing images
    # If the dataset has a None mask_path, the whole dataset is treated as the test dataset
    datamodule.prepare_data()
    datamodule.setup('test')

    i = 0
    bound_set = set()
    tiles = []
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
                sample['output'].cpu().numpy().squeeze().clip(0.3, 1.0),
                sample['crs'],
                transform
            )
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

            memfile = MemoryFile()
            tile = memfile.open(driver=out_meta['driver'],
                                height=out_meta['height'],
                                width=out_meta['width'],
                                count=out_meta['count'],
                                crs=out_meta['crs'],
                                transform=out_meta['transform'],
                                dtype=out_meta['dtype'])

            arr = ds.read(1) * 255
            arr = arr.astype(rasterio.uint8)
            tile.write(arr, 1)
            i += 1
            tiles.append(tile)
            print(i, end='\r')
    print()

    dest, output_transform = rasterio.merge.merge(tiles, method='max')

    out_meta.update(
        {
            "driver": "GTiff",
            "height": dest.shape[1],
            "width": dest.shape[2],
            "transform": output_transform
        }
    )

    with rasterio.open(dst_path, "w", **out_meta) as fp:
        fp.write(dest)


# Args:
# 1. ensemble config, with version path and checkpoint filename
if __name__ == "__main__":
    default_config = configparser.defaultConfig()[0]
    ensemble_config = configparser.parseConfig(argv[1])[0]

    output_path = pathlib.PosixPath(ensemble_config['output']['path'])
    os.makedirs(output_path, exist_ok=True)

    print(f'=> Ensemble config used:\n{yaml.dump(ensemble_config, None)}\n')

    outputs = []
    for model in ensemble_config['ensemble']:

        version_name = model['version_path'].split('/')[-1]
        print(f'=> Creating inferrence masks for version: {
              version_name.split('_')[-1]
              }')

        path = pathlib.PosixPath(model['version_path'])
        checkpoint = path / model['checkpoint']
        hparams_file = path / "hparams.yaml"
        config_file = path / "config.yaml"

        conf = default_config | configparser.parseConfig(config_file)[0]
        model = models.model_classes[conf['model']].load_from_checkpoint(
            checkpoint,
            hparams_file=hparams_file
        )

        # For overrides of e.g. the dataset to plot
        # Should be compatible with original model, of course
        conf['datamodule_params'] |= ensemble_config['datamodule_params']
        product_name = ensemble_config['datamodule_params']['sentinel_products']

        for stride in ensemble_config['strides']:
            conf['datamodule_params']['stride'] = stride

            output_name = output_path / f"{product_name}_{version_name}_{stride}.tif"
            create_inferrence_mask(conf, output_name)
            print(f'==> Created: {output_name}')
            outputs.append(output_name)

    print('=> Merging outputs')
    # Average all predictions
    dest, output_transform = rasterio.merge.merge(
        outputs,
        method='sum',
        dtype=rasterio.uint32
    )
    dest //= len(outputs)
    dest = dest.astype(rasterio.uint8)

    with rasterio.open(outputs[0]) as src:
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": dest.shape[1],
                "width": dest.shape[2],
                "transform": output_transform
            }
        )

    merged_output = output_path / \
        f"{ensemble_config['output']['merged_filename']}.tif"
    with rasterio.open(merged_output, "w", **out_meta) as fp:
        fp.write(dest)

    print(f'=> Created: {merged_output}')

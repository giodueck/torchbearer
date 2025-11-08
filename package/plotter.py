import torch
import lightning.pytorch as pl
from torchgeo.datasets import unbind_samples
from .datamodules import Sentinel2_60mDataModule
from .models import UNet
from .config.products import PRODUCTS
from sys import argv


if __name__ == "__main__":
    if (len(argv) > 1):
        do_predict = True
    else:
        do_predict = False

    pl.seed_everything(3)
    datamodule = Sentinel2_60mDataModule(batch_size=6, patch_size=96, length=None,
                                         num_workers=4, sentinel_path='data',
                                         sentinel_products=PRODUCTS, mask_path='masks')

    # Plot sample testing images
    datamodule.prepare_data()
    datamodule.setup('test')

    if do_predict:
        checkpoint = argv[1]
        hparams_file = argv[2]
        model = UNet.load_from_checkpoint(argv[1], hparams_file=hparams_file)

    for i, batch in enumerate(datamodule.test_dataloader()):
        if do_predict:
            with torch.no_grad():
                image_tensor = (torch.FloatTensor(batch['image'])).to(torch.device('cuda'))
                batch['output'] = model(image_tensor)
        sample = unbind_samples(batch)[0]
        datamodule.plot(sample, filename=f'data/plots/plot_{i}')
    exit(0)
